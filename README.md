title: Subtitle OCR & Translation

description: >
  This script extracts hard-coded subtitles from videos using RapidOCR, then
  (optionally) translates them into Persian (Farsi) with OpenAI.
  Workflow: Video → OCR → English SRT → Persian SRT

features:
  - Auto-detects subtitle region (or set manually).
  - Cleans frames before OCR for better accuracy.
  - Groups lines with stability logic to avoid flicker.
  - Saves standard .srt subtitle files.
  - Optional: parallel translation into Persian (OpenAI).

requirements:
  python: "3.9+"
  external_tools:
    - FFmpeg (sometimes needed for OpenCV video support)
  pip_packages:
    - opencv-python
    - rapidocr-onnxruntime
    - openai
    - numpy

setup:
  steps:
    - Clone this repo
    - Install requirements with pip
    - Place your video file (e.g. video.mp4)
    - (Optional) set OpenAI API key for translation:
      - Linux/macOS: export OPENAI_API_KEY=sk-...
      - Windows (PowerShell): setx OPENAI_API_KEY "sk-..."

usage:
  edit_args_class:
    video_path: "C:\\path\\to\\video.mp4"
    manual_rect: null          # "800,800,1200,1080" for manual region
    center_crop: null          # "0.6x0.35" for width x height ratios
    auto_roi: true             # auto-detect subtitle band
    translate: true            # set false to skip Persian translation
  run:
    command: python main.py
  output_files:
    - video.ocr.en.srt (English OCR subtitles)
    - video.ocr.fa.srt (Persian translation, if enabled)

roi_options:
  auto: "default, scans bottom part of video"
  manual: "set manual_rect = 'x0,y0,x1,y1'"
  center_crop: "set center_crop = 'width_ratioxheight_ratio', e.g. '0.6x0.35'"
  fallback: "auto_roi = true if nothing else specified"

notes:
  - Translation prompt is optimized for Persian but can be adapted to other languages.
  - Accuracy is good for personal/fan use, not a pro subtitling pipeline.
  - Speed depends on video length and CPU/GPU.
  - Works with mp4/mkv and most formats OpenCV supports.

roadmap:
  - Add proper CLI arguments instead of editing Args directly
  - Support config file
  - Add more translation language presets

license: MIT
