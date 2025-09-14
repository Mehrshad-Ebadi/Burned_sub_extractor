# extract_subs_ocr.py
# 1) Detect & crop subtitle region (auto bottom band, or center-crop, or manual rect)
# 2) OCR subtitles (RapidOCR, prefers GPU via onnxruntime-gpu), build timings -> EN SRT
# 3) Translate EN SRT -> FA SRT (parallel) [optional]
# 4) (NEW) Optional progress readout in seconds during OCR

import os, re, cv2, time, argparse
from difflib import SequenceMatcher
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidocr_onnxruntime import RapidOCR
import onnxruntime as ort
from openai import OpenAI
import numpy as np
from openai import APIError, RateLimitError

# ---------------- Utilities ----------------
def tiny_thumb(gray, max_side=96):
    h, w = gray.shape[:2]
    s = max(1, int(round(max(h, w) / max_side)))
    return cv2.resize(gray, (w//s, h//s), interpolation=cv2.INTER_AREA)

def changed_enough(prev_tn, curr_tn, thr=6.0):
    # mean absolute difference on tiny thumbnails
    if prev_tn is None:
        return True
    return float(np.mean(cv2.absdiff(prev_tn, curr_tn))) >= thr

def ms_to_ts(ms: int) -> str:
    if ms < 0: ms = 0
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000;    ms %= 60_000
    s = ms // 1_000;     ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def clean_ocr_text(txt: str) -> str:
    t = txt.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"[-–—]\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def split_srt_blocks(srt_text: str) -> List[str]:
    return re.split(r"\n\s*\n", srt_text.strip())

def retry(func, *args, retries=5, backoff=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except (APIError, RateLimitError) as e:
            if i == retries - 1:
                raise
            wait = backoff ** i
            print(f"[warn] API error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

# ------------- OCR helpers & ROI detection -------------

def preprocess_crop(crop,
                    denoise_binarize=True,
                    scale=2.0,
                    use_adaptive=True,
                    auto_invert=True):
    if not denoise_binarize:
        return crop

    # 1) Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 2) Contrast boost (CLAHE is great for subtitles)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3) Upscale for sharper edges
    if scale and scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 4) Light denoise that preserves edges
    gray = cv2.medianBlur(gray, 3)  # try 3; if still noisy, 5

    # 5) Decide polarity (white text on dark bg -> invert for binary)
    invert = False
    if auto_invert:
        band = gray[int(gray.shape[0]*0.25):, :]  # bottom part where text likely is
        invert = band.mean() < 128  # dark background -> invert thresholding

    # 6) Threshold
    if use_adaptive:
        bw = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            31, 15  # tune: blockSize odd 21–41, C 5–15
        )
    else:
        _, bw = cv2.threshold(
            gray, 0, 255,
            (cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY) | cv2.THRESH_OTSU
        )

    # 7) Gentle morphology to connect broken strokes
    kernel = np.ones((2,2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    # 8) Return in BGR (RapidOCR accepts BGR)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def rapidocr_text_and_score(ocr: RapidOCR, crop) -> Tuple[str, float]:
    """
    RapidOCR returns something like:
      [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text", score
    but some wrappers may return ["text", score] or similar.
    This function detects the format and extracts (text, score) robustly.
    """
    result, _ = ocr(crop)
    if not result:
        return "", 0.0

    text_parts, score_sum = [], 0.0
    for item in result:
        text = ""
        score = 0.0

        try:
            # Common RapidOCR format: [box, text, score]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # If first element looks like a box (list/tuple of points), use item[1] as text
                if isinstance(item[0], (list, tuple)):
                    text = str(item[1]) if len(item) > 1 else ""
                    # score is usually at index 2 or last
                    if len(item) > 2 and isinstance(item[2], (float, int)):
                        score = float(item[2])
                    elif isinstance(item[-1], (float, int)):
                        score = float(item[-1])
                else:
                    # Alternative format: [text, score, ...]
                    text = str(item[0])
                    if isinstance(item[1], (float, int)):
                        score = float(item[1])
                    elif isinstance(item[-1], (float, int)):
                        score = float(item[-1])
            elif isinstance(item, str):
                text = item
            else:
                text = ""
        except Exception:
            text, score = "", 0.0

        # Skip obvious non-text like pure coordinate arrays accidentally stringified
        if text and not re.fullmatch(r"\s*\[\[.*\]\]\s*", text):
            text_parts.append(text)
            score_sum += max(0.0, score)

    return clean_ocr_text(" ".join(text_parts)), score_sum

def auto_detect_roi(video_path: str,
                    ocr: RapidOCR,
                    probe_seconds: int = 15,
                    roi_scan_steps: int = 4,
                    height_choices: Optional[List[float]] = None,
                    denoise_binarize: bool = True) -> Tuple[int,int,int,int]:
    """Scan the lower part of the frame for the band with best OCR confidence."""
    if height_choices is None:
        height_choices = [0.20, 0.24, 0.28, 0.32]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, fps * probe_seconds))
    stride = max(1, int(fps // 2))  # ~2 fps sampling

    bottom_zone_start = int(height * 0.55)
    if roi_scan_steps <= 1:
        y_tops = [bottom_zone_start]
    else:
        step = max(1, (height - bottom_zone_start) // roi_scan_steps)
        y_tops = list(range(bottom_zone_start, height - 10, step))[:roi_scan_steps]

    best = None
    frame_idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue

        for h_ratio in height_choices:
            roi_h = int(height * h_ratio)
            for y0 in y_tops:
                y0 = max(0, min(y0, height - roi_h))
                y1 = y0 + roi_h
                x0, x1 = 0, width
                pad = 8  # try 6–12
                yy0 = max(0, y0 - pad); yy1 = min(frame.shape[0], y1 + pad)
                xx0 = max(0, x0);       xx1 = min(frame.shape[1], x1)
                crop = frame[yy0:yy1, xx0:xx1]
                crop = preprocess_crop(crop, denoise_binarize)
                txt, score = rapidocr_text_and_score(ocr, crop)
                score_w = score * (1.0 + 0.02 * len(txt))
                if not best or score_w > best[0]:
                    best = (score_w, x0, y0, x1, y1)

        if frame_idx >= max_frames:
            break

    cap.release()
    if not best:
        # fallback: bottom 28%
        roi_h = int(height * 0.28)
        return (0, height - roi_h, width, height)
    _, x0, y0, x1, y1 = best
    return (x0, y0, x1, y1)

# ------------- Main OCR pipeline -------------

def extract_subs_from_video(
    video_path: str,
    # ROI modes (priority: manual_rect > center_crop > auto_roi > ratios fallback)
    manual_rect: Optional[Tuple[int,int,int,int]] = None,  # x0,y0,x1,y1 pixels
    center_crop: Optional[Tuple[float,float]] = None,      # width_ratio, height_ratio around center
    auto_roi: bool = False,
    bottom_ratio: Optional[float] = None,
    height_ratio: Optional[float] = None,
    # Timing & robustness
    sample_stride: int = 2,
    stability_frames: int = 2,
    sim_threshold: float = 0.80,
    min_show_ms: int = 500,
    denoise_binarize: bool = True,
    probe_seconds: int = 15,
    roi_scan_steps: int = 4,
    # NEW: progress reporting
    show_progress: bool = False,
    progress_interval: float = 1.0
) -> List[Tuple[int,int,str]]:
    """Return list of (start_ms, end_ms, text)."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration_s = float(total_frames) / float(fps) if total_frames > 0 else 0.0

    print("[info] onnxruntime providers:", ort.get_available_providers())
    #ocr = RapidOCR()  # will pick CUDA if onnxruntime-gpu is installed
    print("[info] ORT available providers:", ort.get_available_providers())
    try:
        ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    except TypeError:
        try:
            ocr = RapidOCR(use_cuda=True)  # older RapidOCR versions
        except TypeError:
            ocr = RapidOCR()


    # ---- Resolve ROI
    if manual_rect:
        x0, y0, x1, y1 = manual_rect
        x0 = max(0, min(x0, width-1)); x1 = max(1, min(x1, width))
        y0 = max(0, min(y0, height-1)); y1 = max(1, min(y1, height))
    elif center_crop:
        wr, hr = center_crop
        wr = max(0.05, min(1.0, wr))
        hr = max(0.05, min(1.0, hr))
        cw, ch = int(width*wr), int(height*hr)
        x0 = max(0, (width - cw)//2); x1 = min(width, x0 + cw)
        y0 = max(0, (height - ch)//2); y1 = min(height, y0 + ch)
    elif auto_roi or bottom_ratio is None or height_ratio is None:
        print("[info] Auto-detecting subtitle band…")
        x0, y0, x1, y1 = auto_detect_roi(
            video_path, ocr,
            probe_seconds=probe_seconds,
            roi_scan_steps=roi_scan_steps,
            denoise_binarize=denoise_binarize
        )
    else:
        roi_h = int(height * height_ratio)
        y1_base = height
        y0 = max(0, y1_base - int(height * bottom_ratio) - roi_h)
        y1 = min(height, y0 + roi_h)
        x0, x1 = 0, width

    print(f"[info] Video: {width}x{height} @ {fps:.2f} fps, frames={total_frames}")
    print(f"[info] OCR ROI: x={x0}:{x1}, y={y0}:{y1}, h={y1-y0}")

    # Timing state
    current_text = ""
    last_confirmed = ""
    open_start_ms: Optional[int] = None
    pending_same = pending_diff = 0

    segments: List[Tuple[int,int,str]] = []

    def run_ocr(crop):
        txt, _ = rapidocr_text_and_score(ocr, crop)
        return clean_ocr_text(txt)

    frame_idx = 0
    last_progress_report_s = -1.0  # progress ticker baseline

    while True:
        ok = cap.grab()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        # --- change gate before heavy OCR ---
        crop_raw = frame[y0:y1, x0:x1]
        gray_small = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)
        tn = tiny_thumb(gray_small)

        now_ms = int((frame_idx / fps) * 1000)
        force_every_ms = 600  # always run OCR at least every 0.6s
        time_due = (now_ms - globals().get("last_ocr_ms", -9999)) >= force_every_ms

        if not changed_enough(globals().get("prev_tn", None), tn) and not time_due:
            # no big visual change → skip OCR this frame
            globals()["prev_tn"] = tn
            continue

        globals()["prev_tn"] = tn
        globals()["last_ocr_ms"] = now_ms

        # --- only do heavy OCR if change detected or timeout ---
        crop = preprocess_crop(crop_raw, denoise_binarize)
        text = run_ocr(crop)
        
        if text and not re.search(r"[A-Za-z\u0600-\u06FF]", text):
            text = ""

        t_ms = int((frame_idx / fps) * 1000)

        # --- progress line (in seconds) ---
        if show_progress:
            t_s = t_ms / 1000.0
            should_print = (
                last_progress_report_s < 0 or
                (t_s - last_progress_report_s) >= progress_interval or
                int(t_s) != int(last_progress_report_s)
            )
            if should_print:
                if total_duration_s > 0:
                    pct = min(1.0, t_s / total_duration_s)
                    print(f"\r[progress] {t_s:,.1f}s / {total_duration_s:,.1f}s ({pct:.2%})", end="", flush=True)
                else:
                    print(f"\r[progress] {t_s:,.1f}s", end="", flush=True)
                last_progress_report_s = t_s
        
        if last_confirmed and text and similar(last_confirmed, text) >= 0.98:
            pending_diff = 0              # prevent accidental closure
            # (optional) current_text = last_confirmed
            continue


        if not text:
            if last_confirmed:
                pending_diff += 1
                if pending_diff >= stability_frames:
                    end_ms = t_ms
                    if open_start_ms is not None and end_ms - open_start_ms >= min_show_ms:
                        segments.append((open_start_ms, end_ms, last_confirmed))
                    last_confirmed = ""; open_start_ms = None
                    pending_same = pending_diff = 0
            else:
                pending_same = pending_diff = 0
            continue

        if not last_confirmed:
            if current_text and similar(current_text, text) >= sim_threshold:
                pending_same += 1
            else:
                current_text = text
                pending_same = 1
            if pending_same >= stability_frames:
                last_confirmed = current_text
                open_start_ms = t_ms
                pending_same = pending_diff = 0
        else:
            if similar(last_confirmed, text) >= sim_threshold:
                pending_diff = 0
            else:
                if current_text and similar(current_text, text) >= sim_threshold:
                    pending_same += 1
                else:
                    current_text = text
                    pending_same = 1
                if pending_same >= stability_frames:
                    end_ms = t_ms
                    if open_start_ms is not None and end_ms - open_start_ms >= min_show_ms:
                        segments.append((open_start_ms, end_ms, last_confirmed))
                    last_confirmed = current_text
                    open_start_ms = t_ms
                    pending_same = pending_diff = 0

    if last_confirmed and open_start_ms is not None:
        end_ms = int((frame_idx / fps) * 1000)
        if end_ms - open_start_ms >= min_show_ms:
            segments.append((open_start_ms, end_ms, last_confirmed))

    # finish progress line
    if show_progress:
        if total_duration_s > 0:
            print(f"\r[progress] {total_duration_s:,.1f}s / {total_duration_s:,.1f}s (100.00%)")
        else:
            print("\r[progress] done")

    cap.release()

    # Merge adjacent identical with tiny gaps
    merged: List[Tuple[int,int,str]] = []
    gap_ms = 150
    for s, e, txt in segments:
        if merged and txt == merged[-1][2] and s - merged[-1][1] <= gap_ms:
            ps, pe, ptxt = merged[-1]
            merged[-1] = (ps, e, ptxt)
        else:
            merged.append((s, e, txt))
    return merged

# ------------- SRT & Translation -------------

def segments_to_srt(segments: List[Tuple[int,int,str]]) -> str:
    out = []
    for i, (s, e, txt) in enumerate(segments, start=1):
        if not txt: continue
        out.append(f"{i}")
        out.append(f"{ms_to_ts(s)} --> {ms_to_ts(e)}")
        out.append(txt)
        out.append("")
    return "\n".join(out).strip() + "\n"

TRANSLATION_GUIDE_FA_ONLY = """You are a professional translator.
You will receive one SRT block (index line, timecode line, and 1–3 subtitle text lines).
Your job:
1) Do NOT modify the index line or the timecode line(s).
2) Replace ONLY the subtitle text line(s) with a fluent, natural Persian (Farsi) translation.
3) Do NOT include any source-language words in parentheses, footnotes, or glosses.
4) Preserve the number of subtitle text lines (same line breaks).
5) Keep numbers, timing cues, and punctuation sensible in Persian.
Return only the modified SRT block.
"""

def translate_block_fa_only(client: OpenAI, block: str, llm: str) -> str:
    prompt = f"""{TRANSLATION_GUIDE_FA_ONLY}

SRT BLOCK:
{block}
"""
    resp = retry(client.responses.create, model=llm, input=prompt)
    return resp.output_text.strip()

def translate_srt_parallel(client: OpenAI, srt_text: str, llm: str, max_workers: int = 8) -> str:
    blocks = split_srt_blocks(srt_text)
    if not blocks: return ""
    results = [""] * len(blocks)

    def worker(i_b):
        i, b = i_b
        return i, translate_block_fa_only(client, b, llm)

    done = 0
    max_workers = max(1, min(32, max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, (i, b)) for i, b in enumerate(blocks)]
        for fut in as_completed(futures):
            i, out = fut.result()
            results[i] = out
            done += 1
            if done % 25 == 0 or done == len(blocks):
                print(f"[info] Translated {done}/{len(blocks)} blocks…")
    return "\n\n".join(results)

# ---------------- CLI ----------------


class Args:
    video_path= "C:\\Users\\ebadi\\Videos\\channel\\transcriptor\\22.mp4"
    auto_roi=False
    # Manual ROI controls
    bottom_ratio = None          # distance from bottom (0-1), used with height_ratio
    height_ratio = None          # relative height (0-1) of ROI
    center_crop = None           # Center crop as WIDTHxHEIGHT ratios, e.g., 0.6x0.35
    manual_rect = "800, 800, 1200, 1080"           # Manual rect as x0,y0,x1,y1 in pixels

    # Auto-ROI probe controls
    probe_seconds = 4           # Seconds to probe for auto-ROI
    roi_scan_steps = 4           # Vertical steps to scan in bottom zone

    # OCR timing robustness
    sample_stride = 6            # Process every Nth frame
    stability_frames = 1         # Frames to confirm appearance/change
    sim_threshold = 0.90         # Similarity threshold for same text
    min_show_ms = 1            # Minimum on-screen duration to keep

    # Progress (NEW)
    show_progress = False        # Show running progress in seconds
    progress_interval = 1.0      # Seconds between progress updates

    # Translation
    no_translate = False         # Skip translation stage
    llm = "gpt-4o-mini"          # OpenAI model for translation (e.g., gpt-4o, gpt-4o-mini)
    translate_workers = 8        # Parallel translation workers

    

def main():
    # ap = argparse.ArgumentParser(description="OCR burned-in subtitles (GPU OCR if available), then translate to Farsi.")
    # ap.add_argument("video_path", help="Path to the video file (mp4, mkv, etc.)")

    # # ROI control (priority: manual > center > auto > ratios)
    # ap.add_argument("--auto-roi", action="store_true", help="Auto-detect subtitle band near bottom.")
    # ap.add_argument("--bottom-ratio", type=float, default=None, help="Manual: distance from bottom (0-1), used with --height-ratio.")
    # ap.add_argument("--height-ratio", type=float, default=None, help="Manual: relative height (0-1) of ROI.")
    # ap.add_argument("--center-crop", type=str, default=None, help="Center crop as WIDTHxHEIGHT ratios, e.g., 0.6x0.35")
    # ap.add_argument("--manual-rect", type=str, default=None, help="Manual rect as x0,y0,x1,y1 in pixels.")

    # # Auto-ROI probe controls
    # ap.add_argument("--probe-seconds", type=int, default=15, help="Seconds to probe for auto-ROI.")
    # ap.add_argument("--roi-scan-steps", type=int, default=4, help="Vertical steps to scan in bottom zone.")

    # # OCR timing robustness
    # ap.add_argument("--sample-stride", type=int, default=2, help="Process every Nth frame.")
    # ap.add_argument("--stability-frames", type=int, default=2, help="Frames to confirm appearance/change.")
    # ap.add_argument("--sim-threshold", type=float, default=0.90, help="Similarity threshold for same text.")
    # ap.add_argument("--min-show-ms", type=int, default=500, help="Minimum on-screen duration to keep.")

    # # Progress (NEW)
    # ap.add_argument("--show-progress", action="store_true", help="Show running progress in seconds.")
    # ap.add_argument("--progress-interval", type=float, default=1.0, help="Seconds between progress updates.")

    # # Translation
    # ap.add_argument("--no-translate", action="store_true", help="Skip translation stage.")
    # ap.add_argument("--llm", default="gpt-4o-mini", help="OpenAI model for translation (e.g., gpt-4o, gpt-4o-mini).")
    # ap.add_argument("--translate-workers", type=int, default=8, help="Parallel translation workers.")

    # args = ap.parse_args()

    args = Args
    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(args.video_path)

    # Parse ROI flags
    manual_rect = None
    center_crop = None
    if args.manual_rect:
        try:
            x0,y0,x1,y1 = [int(v) for v in args.manual_rect.split(",")]
            manual_rect = (x0,y0,x1,y1)
        except Exception:
            raise ValueError("Invalid --manual-rect. Use: x0,y0,x1,y1")
    elif args.center_crop:
        try:
            wr, hr = args.center_crop.lower().split("x")
            center_crop = (float(wr), float(hr))
        except Exception:
            raise ValueError("Invalid --center-crop. Use: 0.6x0.35 (width_ratio x height_ratio)")

    # Step 1: OCR + timings
    print("[step 1] Extracting subtitles via OCR…")
    segments = extract_subs_from_video(
        args.video_path,
        manual_rect=manual_rect,
        center_crop=center_crop,
        auto_roi=args.auto_roi,
        bottom_ratio=args.bottom_ratio,
        height_ratio=args.height_ratio,
        sample_stride=args.sample_stride,
        stability_frames=args.stability_frames,
        sim_threshold=args.sim_threshold,
        min_show_ms=args.min_show_ms,
        denoise_binarize=True,
        probe_seconds=args.probe_seconds,
        roi_scan_steps=args.roi_scan_steps,
        show_progress=args.show_progress,
        progress_interval=args.progress_interval
    )
    print(f"[ok] Found {len(segments)} subtitle segments.")

    # Save EN SRT
    base, _ = os.path.splitext(args.video_path)
    en_path = base + ".ocr.en.srt"
    en_srt = segments_to_srt(segments)
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(en_srt)
    print(f"[ok] English (OCR) SRT saved: {en_path}")

    if args.no_translate:
        return

    # Step 2: Translate to FA
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI()

    print(f"[step 2] Translating to Farsi with {args.llm} using {args.translate_workers} workers…")
    fa_srt = translate_srt_parallel(client, en_srt, args.llm, max_workers=args.translate_workers)
    fa_path = base + ".ocr.fa.srt"
    with open(fa_path, "w", encoding="utf-8") as f:
        f.write(fa_srt.replace("\r\n", "\n"))
    print(f"[ok] Persian SRT saved: {fa_path}")




if __name__ == "__main__":

    main()