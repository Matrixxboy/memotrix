---
title: "Images and media"
---

# Use case: images, audio, and video

## Images

Requires `pip install "memotrix[images]"`. Extensions: PNG, JPEG, WebP, BMP, GIF, TIFF, SVG, HEIC.

- `describe_images=True` (default): `describe_image` in `utils/ai_integration.py`. If no vision provider is configured, captions fall back to local/heuristic text, not a guaranteed cloud vision model.
- `describe_images=False`: placeholder description (`Image file {name}`).
- SVG conversion uses `cairosvg` if installed; otherwise `MissingDependencyError`.
- HEIC uses `pillow-heif`.

PDF/DOCX embedded images can be counted in `add`’s `extracted_images`.

## Audio

Requires `pip install "memotrix[audio]"` and **ffmpeg** on PATH. `AudioExtractor` uses `faster-whisper` (default model size `"base"`), converting to 16 kHz mono WAV.

`memory.add("call.mp3", generate_srt=True)` may write an SRT beside the file.

## Video

Same Whisper + ffmpeg path: audio is stripped from the container, then transcribed. Extensions: mp4, mkv, avi, mov, webm, m4v, wmv.

## Limitations

- Transcription quality and latency depend on Whisper model size and CPU/GPU — not measured in this DocBook unless you run them.
- Video does not index visual frames as a separate embedding space; it indexes the **transcript** (and any summary the extractor attaches).
