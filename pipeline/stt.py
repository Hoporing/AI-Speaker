import numpy as np
from faster_whisper import WhisperModel


class STT:
    def __init__(self, model_path: str, device: str = "cpu", language: str = "ko"):
        self.model = WhisperModel(model_path, device=device, compute_type="int8", local_files_only=True)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.language)
        return " ".join(seg.text for seg in segments).strip()
