import os, re, cv2, time
from typing import List, Tuple, Optional
import numpy as np
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI, APIError, RateLimitError

#initial steps:

def ms_to_ts(ms: int) -> str:
    ms = max(0, ms)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms,    60_000)
    s, ms = divmod(ms,     1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def clean_text(t: str) -> str:
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"[-–—]\s*$", "", t)
    return re.sub(r"\s+", " ", t).strip()

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def to_srt(segments: List[Tuple[int,int,str]]) -> str:
    rows = []
    for i, (s, e, txt) in enumerate(segments, 1):
        if not txt: 
            continue
        rows += [str(i), f"{ms_to_ts(s)} --> {ms_to_ts(e)}", txt, ""]
    return "\n".join(rows).strip() + "\n"

# OCR_helpers

def preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    gray = cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    invert = gray[int(gray.shape[0]*0.25):].mean() < 128
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY, 31, 15
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def tiny_thumb(gray: np.ndarray, max_side=96) -> np.ndarray:
    h, w = gray.shape[:2]
    s = max(1, int(round(max(h, w) / max_side)))
    return cv2.resize(gray, (w//s, h//s), interpolation=cv2.INTER_AREA)

def changed(prev_tn: Optional[np.ndarray], curr_tn: np.ndarray, thr=6.0) -> bool:
    if prev_tn is None: 
        return True
    return float(np.mean(cv2.absdiff(prev_tn, curr_tn))) >= thr

def rapid_text(ocr: RapidOCR, crop: np.ndarray) -> str:
    result, _ = ocr(crop)
    if not result: 
        return ""
    parts = []
    for item in result:
        text = ""
        if isinstance(item, (list, tuple)):
            if item and isinstance(item[0], (list, tuple)):
                text = str(item[1]) if len(item) > 1 else ""
            else:
                text = str(item[0])
        elif isinstance(item, str):
            text = item
        if text and not re.fullmatch(r"\s*\[\[.*\]\]\s*", text):
            parts.append(text)
    out = clean_text(" ".join(parts))
    if out and not re.search(r"[A-Za-z\u0600-\u06FF]", out):
        return ""
    return out

# ROI

def auto_roi(video_path: str, ocr: RapidOCR, probe_sec=8) -> Tuple[int,int,int,int]:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, fps * probe_sec))
    stride = max(1, int(fps // 2))  # ~2 fps
    y_start = int(H * 0.55)
    heights = [0.20, 0.26, 0.32]
    y_candidates = list(range(y_start, H - 10, max(1, (H - y_start)//4)))[:4]

    best = None
    f = 0
    while True:
        if not cap.grab(): break
        f += 1
        if f % stride: 
            continue
        ok, frame = cap.retrieve()
        if not ok: 
            continue

        for hr in heights:
            rh = int(H * hr)
            for y0 in y_candidates:
                y0 = max(0, min(y0, H - rh))
                y1 = y0 + rh
                crop = preprocess(frame[y0:y1, :])
                txt = rapid_text(ocr, crop)
                score = len(txt) 
                if not best or score > best[0]:
                    best = (score, 0, y0, W, y1)
        if f >= max_frames:
            break

    cap.release()
    if not best:
        rh = int(H * 0.28)
        return (0, H - rh, W, H)
    _, x0, y0, x1, y1 = best
    return (x0, y0, x1, y1)

# OCR main loop

def extract_segments(
    video_path: str,
    manual_rect: Optional[Tuple[int,int,int,int]] = None,
    center_crop: Optional[Tuple[float,float]] = None,
    auto=True,
    sample_stride=3,
    stability_frames=2,
    sim_threshold=0.85,
    min_show_ms=350,
) -> List[Tuple[int,int,str]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # OCR
    try:
        ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    except TypeError:
        try:
            ocr = RapidOCR(use_cuda=True)
        except TypeError:
            ocr = RapidOCR()

    # ROI
    if manual_rect:
        x0, y0, x1, y1 = manual_rect
    elif center_crop:
        wr, hr = max(0.05, min(1.0, center_crop[0])), max(0.05, min(1.0, center_crop[1]))
        cw, ch = int(W*wr), int(H*hr)
        x0, x1 = (W - cw)//2, (W - cw)//2 + cw
        y0, y1 = (H - ch)//2, (H - ch)//2 + ch
    elif auto:
        x0, y0, x1, y1 = auto_roi(video_path, ocr)
    else:
        rh = int(H * 0.28)
        x0, y0, x1, y1 = 0, H - rh, W, H

    segments: List[Tuple[int,int,str]] = []
    prev_tn = None
    last_text = ""              
    candidate = ""              
    open_start_ms = None
    same_cnt = 0
    diff_cnt = 0
    last_ocr_ms = -10_000

    f = 0
    while True:
        if not cap.grab(): 
            break
        f += 1
        if f % sample_stride: 
            continue
        ok, frame = cap.retrieve()
        if not ok: 
            continue

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        tn = tiny_thumb(gray)
        now_ms = int((f / fps) * 1000)
        time_due = (now_ms - last_ocr_ms) >= 600 

        if not changed(prev_tn, tn) and not time_due:
            prev_tn = tn
            continue

        prev_tn = tn
        last_ocr_ms = now_ms

        txt = rapid_text(ocr, preprocess(roi))
        t_ms = int((f / fps) * 1000)

        if last_text and txt and similar(last_text, txt) >= 0.98:
            diff_cnt = 0
            continue

        if not txt:
            if last_text:
                diff_cnt += 1
                if diff_cnt >= stability_frames:
                    end = t_ms
                    if open_start_ms is not None and end - open_start_ms >= min_show_ms:
                        segments.append((open_start_ms, end, last_text))
                    last_text, open_start_ms = "", None
                    same_cnt = diff_cnt = 0
            else:
                same_cnt = diff_cnt = 0
            continue

        if not last_text:
            if candidate and similar(candidate, txt) >= sim_threshold:
                same_cnt += 1
            else:
                candidate, same_cnt = txt, 1
            if same_cnt >= stability_frames:
                last_text = candidate
                open_start_ms = t_ms
                same_cnt = diff_cnt = 0
        else:
            if similar(last_text, txt) >= sim_threshold:
                diff_cnt = 0
            else:
                if candidate and similar(candidate, txt) >= sim_threshold:
                    same_cnt += 1
                else:
                    candidate, same_cnt = txt, 1
                if same_cnt >= stability_frames:
                    end = t_ms
                    if open_start_ms is not None and end - open_start_ms >= min_show_ms:
                        segments.append((open_start_ms, end, last_text))
                    last_text = candidate
                    open_start_ms = t_ms
                    same_cnt = diff_cnt = 0

    cap.release()

    if last_text and open_start_ms is not None:
        end = int((f / fps) * 1000)
        if end - open_start_ms >= min_show_ms:
            segments.append((open_start_ms, end, last_text))

    # merge near-duplicates with tiny gaps
    merged: List[Tuple[int,int,str]] = []
    GAP = 150
    for s, e, txt in segments:
        if merged and txt == merged[-1][2] and s - merged[-1][1] <= GAP:
            ps, pe, ptxt = merged[-1]
            merged[-1] = (ps, e, ptxt)
        else:
            merged.append((s, e, txt))
    return merged

# translation

TRANSLATION_GUIDE_FA = """You are a professional translator.
You will receive one SRT block (index line, time line, then 1–3 text lines).
Rules:
- Do not touch the index or the time line.
- Replace only the text lines with fluent Persian.
- Keep the same number of text lines.
- No notes, parentheses, or glosses.
Return only the modified block.
"""

def retry(fn, *args, retries=5, backoff=2, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except (APIError, RateLimitError) as e:
            if i == retries - 1:
                raise
            time.sleep(backoff ** i)

def split_blocks(srt_text: str) -> List[str]:
    return re.split(r"\n\s*\n", srt_text.strip())

def translate_block(client: OpenAI, block: str, model: str) -> str:
    prompt = f"{TRANSLATION_GUIDE_FA}\n\nSRT BLOCK:\n{block}\n"
    resp = retry(client.responses.create, model=model, input=prompt)
    return resp.output_text.strip()

def translate_srt(client: OpenAI, srt_text: str, model="gpt-4o-mini", workers=8) -> str:
    blocks = split_blocks(srt_text)
    out = [""] * len(blocks)

    def task(i_b):
        i, b = i_b
        return i, translate_block(client, b, model)

    with ThreadPoolExecutor(max_workers=max(1, min(32, workers))) as ex:
        futures = [ex.submit(task, (i, b)) for i, b in enumerate(blocks)]
        for fut in as_completed(futures):
            i, t = fut.result()
            out[i] = t
    return "\n\n".join(out)

# --------------- simple CLI-ish entry ---------------

class Args:
    video_path = r"C:\path\to\video.mp4"
    manual_rect = None          # e.g. "800,800,1200,1080"
    center_crop = None          # e.g. "0.6x0.35"
    auto_roi = True

    sample_stride = 3
    stability_frames = 2
    sim_threshold = 0.85
    min_show_ms = 350

    translate = True
    llm = "gpt-4o-mini"
    workers = 8

def parse_rect(rect: Optional[str]) -> Optional[Tuple[int,int,int,int]]:
    if not rect: return None
    x0, y0, x1, y1 = [int(v) for v in rect.split(",")]
    return (x0, y0, x1, y1)

def parse_center(cc: Optional[str]) -> Optional[Tuple[float,float]]:
    if not cc: return None
    wr, hr = cc.lower().split("x")
    return (float(wr), float(hr))

def main():
    a = Args
    if not os.path.isfile(a.video_path):
        raise FileNotFoundError(a.video_path)

    segments = extract_segments(
        a.video_path,
        manual_rect=parse_rect(a.manual_rect),
        center_crop=parse_center(a.center_crop),
        auto=a.auto_roi,
        sample_stride=a.sample_stride,
        stability_frames=a.stability_frames,
        sim_threshold=a.sim_threshold,
        min_show_ms=a.min_show_ms,
    )

    base, _ = os.path.splitext(a.video_path)
    en_path = base + ".ocr.en.srt"
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(to_srt(segments))
    print(f"[ok] EN SRT: {en_path}")

    if not a.translate:
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()

    fa_text = translate_srt(client, to_srt(segments), model=a.llm, workers=a.workers)
    fa_path = base + ".ocr.fa.srt"
    with open(fa_path, "w", encoding="utf-8") as f:
        f.write(fa_text.replace("\r\n", "\n"))
    print(f"[ok] FA SRT: {fa_path}")

if __name__ == "__main__":
    main()
