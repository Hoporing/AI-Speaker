import os
import tempfile
import numpy as np
import soundfile as sf
from melo.api import TTS as MeloTTS


class TTS:
    def __init__(self, ckpt_path: str, config_path: str, bert_path: str, language: str = "KR", speed: float = 1.0):
        # BERT 로컬 경로 강제 설정
        import melo.text.korean
        import melo.text.japanese_bert
        melo.text.korean.model_id = bert_path
        melo.text.japanese_bert.model_id = bert_path
        
        self.model = MeloTTS(
            language=language, 
            device="cpu", 
            use_hf=False, 
            ckpt_path=ckpt_path, 
            config_path=config_path
        )
        self.speaker_id = self.model.hps.data.spk2id[language]
        self.speed = speed

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            self.model.tts_to_file(text, self.speaker_id, tmp_path, speed=self.speed)
            audio, sample_rate = sf.read(tmp_path)
        finally:
            os.unlink(tmp_path)
        return audio, sample_rate
