import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from memotrix.filetypes.document.base import BaseExtractor
from memotrix.utils.ai_integration import get_ai_router
from memotrix.utils.outputSturcture import build_document


class VideoExtractor(BaseExtractor):
    supported_extensions = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv")

    def __init__(self, model_size: str = "base", generate_srt: bool = False):
        self.model_size = model_size
        self.generate_srt = generate_srt
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def _extract_audio(self, video_path: Path) -> Path:
        """Extracts 16kHz mono audio from the video using ffmpeg."""
        import hashlib
        path_hash = hashlib.md5(f"{video_path.absolute()}_{video_path.stat().st_size}".encode()).hexdigest()
        audio_path = Path(tempfile.gettempdir()) / f"audio_{path_hash}.wav"
        if audio_path.exists():
            return audio_path

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(audio_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return audio_path

    def _summarize(self, full_transcript: str) -> str:
        if not full_transcript.strip():
            return ""
        prompt = f"Provide a brief summary of this video transcript for a semantic search engine:\n\n{full_transcript}"
        try:
            return get_ai_router().generate_text(prompt)
        except Exception:
            # Fallback if AI provider is not configured
            return ""

    def _format_srt(self, transcripts: List[Dict[str, Any]]) -> str:
        """Converts transcript segments into SRT subtitle format."""
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, segment in enumerate(transcripts, start=1):
            start = format_timestamp(segment["start_time"])
            end = format_timestamp(segment["end_time"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(segment["text"])
            lines.append("")
        return "\n".join(lines)

    def extract(self, path: Path) -> Any:
        path = Path(path)
        audio_path = None
        try:
            # 1. Extract audio
            audio_path = self._extract_audio(path)

            # 2. Transcribe with VAD filter to improve efficiency and reduce hallucinations
            model = self._get_model()
            segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)

            transcripts: List[Dict[str, Any]] = []
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                transcripts.append({
                    "text": text,
                    "start_time": segment.start,
                    "end_time": segment.end,
                })
                text_parts.append(text)

            full_transcript = " ".join(text_parts)

            if self.generate_srt and transcripts:
                srt_content = self._format_srt(transcripts)
                srt_path = path.with_suffix(".srt")
                srt_path.write_text(srt_content, encoding="utf-8")

            # 3. Summarize (truncate for long videos to prevent context limits)
            summary = self._summarize(full_transcript[:4000])

            # 4. Build document
            metadata = {
                "file_type": "video",
                "kind": "video",
                "language": info.language,
                "duration": info.duration,
            }

            return build_document(
                path,
                text=full_transcript,
                extra_metadata=metadata,
                language=info.language,
                summary=summary,
                transcripts=transcripts,
            )
        finally:
            # Clean up the large audio file after extraction
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    pass