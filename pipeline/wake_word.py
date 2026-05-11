import logging
import numpy as np
from openwakeword.model import Model

logger = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(self, model_name: str = "hey_jarvis", threshold: float = 0.5):
        self.model = Model(
            wakeword_models=[model_name], inference_framework="onnx")
        self.threshold = threshold

    def detect(self, audio_chunk: np.ndarray) -> bool:
        # float32 -> int16 변환 (인식에 필수)
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self.model.predict(audio_int16)

        if not prediction:
            return False

        score = max(prediction.values())

        if score > self.threshold:
            return True
        
        # 아깝게 실패한 경우만 로그 출력 (피드백용)
        if score > 0.3:
            logger.info(f"호출어와 비슷한 소리 감지 (신뢰도: {score:.2f}, 목표: {self.threshold})")

        return False

    def reset(self):
        """감지기의 내부 버퍼와 상태를 초기화합니다."""
        self.model.reset()
