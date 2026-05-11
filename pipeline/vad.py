import torch
import numpy as np


class VAD:
    def __init__(self, model_path: str, threshold: float = 0.5, sample_rate: int = 16000, min_rms: float = 0.01):
        # 로컬 경로에서 직접 로드 (인터넷 차단)
        self.model, _ = torch.hub.load(
            repo_or_dir=model_path,
            model="silero_vad",
            source='local',
            trust_repo=True,
        )
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_rms = min_rms

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        # 볼륨이 너무 낮으면 VAD 연산 없이 바로 False 반환 (노이즈 방지)
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms < self.min_rms:
            return False

        tensor = torch.FloatTensor(audio_chunk)
        confidence = self.model(tensor, self.sample_rate).item()
        return confidence > self.threshold
