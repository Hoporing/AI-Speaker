import os
import sys
import warnings
import logging

# 1. 모든 Python 경고 및 라이브러리 알림 무시 (최우선)
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 2. 라이브러리 로깅 제어
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from audio.input import AudioInput
from audio.output import AudioOutput
from pipeline.llm import LLM
from pipeline.stt import STT
from pipeline.tts import TTS
from pipeline.vad import VAD
from pipeline.wake_word import WakeWordDetector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def record_until_silence(
    audio_input: AudioInput,
    vad: VAD,
    sample_rate: int,
    min_silence_ms: int,
    chunk_size: int,
    wait_for_speech_timeout: float = 5.0,
    min_duration: float = 0.5,
    min_avg_volume: float = 0.01
) -> np.ndarray:
    start_time = time.time()
    silence_limit = max(
        1, int(min_silence_ms / (chunk_size / sample_rate * 1000)))

    logger.info("음성 대기 시작")
    
    while True:
        chunks = []
        speech_started = False
        silence_count = 0
        
        while True:
            chunk = audio_input.read(timeout=0.1)
            if chunk is None:
                if not speech_started and (time.time() - start_time) > wait_for_speech_timeout:
                    return np.array([], dtype=np.float32)
                continue

            chunks.append(chunk)

            if vad.is_speech(chunk):
                if not speech_started:
                    speech_started = True
                    logger.info("사용자 음성 감지 성공! 녹음 시작")
                silence_count = 0
            else:
                if speech_started:
                    silence_count += 1
                    if silence_count >= silence_limit:
                        break
                elif (time.time() - start_time) > wait_for_speech_timeout:
                    return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        if len(audio) == 0:
            return audio

        duration = len(audio) / sample_rate
        avg_volume = np.sqrt(np.mean(audio**2))

        if duration >= min_duration and avg_volume >= min_avg_volume:
            return audio
        
        # 소음 판단 시 재대기
        logger.info(f"소음으로 판단 (기간: {duration:.1f}s, 볼륨: {avg_volume:.4f}). 다시 대기 시작")
        audio_input.clear()
        if (time.time() - start_time) > wait_for_speech_timeout:
            return np.array([], dtype=np.float32)


_MAX_BUFFER_CHARS = 50
_MIN_TTS_CHARS = 10


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。,，\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def should_flush(buffer: str) -> bool:
    return bool(re.search(r"[.!?。,，\n]", buffer)) or len(buffer) >= _MAX_BUFFER_CHARS


def _synthesize_task(tts, text):
    if not text.strip():
        return None
    logger.info(f"TTS 합성 시작: {text[:20]}")
    return tts.synthesize(text)


def main():
    config = load_config()

    logger.info("모델을 로딩 중입니다")

    audio_input = AudioInput(
        sample_rate=config["audio"]["sample_rate"],
        chunk_size=config["audio"]["chunk_size"],
        device=config["audio"]["input_device"],
    )
    audio_output = AudioOutput(device=config["audio"]["output_device"])
    wake_word = WakeWordDetector(
        model_name=config["wake_word"]["model"],
        threshold=config["wake_word"]["threshold"],
    )
    vad = VAD(
        model_path=config["vad"]["model_path"],
        threshold=config["vad"]["threshold"],
        sample_rate=config["audio"]["sample_rate"],
        min_rms=config["vad"].get("min_rms", 0.01),
    )
    stt = STT(
        model_path=config["stt"]["model_path"],
        device=config["stt"]["device"],
        language=config["stt"]["language"],
    )
    llm = LLM(
        model_path=config["llm"]["model_path"],
        n_ctx=config["llm"]["n_ctx"],
        n_gpu_layers=config["llm"]["n_gpu_layers"],
        max_tokens=config["llm"]["max_tokens"],
        temperature=config["llm"]["temperature"],
    )
    tts = TTS(
        ckpt_path=config["tts"]["ckpt_path"],
        config_path=config["tts"]["config_path"],
        bert_path=config["tts"]["bert_path"],
        language=config["tts"]["language"],
        speed=config["tts"]["speed"],
    )

    executor = ThreadPoolExecutor(max_workers=2)

    logger.info("준비 완료. 호출어 대기 시작")
    # 초기 시작 시에는 자연스럽게 버퍼가 채워지도록 reset 생략 또는 시작 후 호출
    audio_input.start()

    try:
        while True:
            # [단계 1] 호출어 대기 루프
            chunk = audio_input.read(timeout=1.0)
            if chunk is None:
                continue

            if not wake_word.detect(chunk):
                continue

            logger.info("호출어 감지! 사용자 요청 대기 시작")
            audio_input.clear()

            # [단계 2] 연속 대화 세션 시작
            while True:
                audio = record_until_silence(
                    audio_input,
                    vad,
                    config["audio"]["sample_rate"],
                    config["vad"]["min_silence_ms"],
                    config["audio"]["chunk_size"],
                    wait_for_speech_timeout=config["audio"].get("wait_for_speech_timeout", 5.0),
                    min_duration=config["audio"].get("min_duration", 0.5),
                    min_avg_volume=config["audio"].get("min_avg_volume", 0.01)
                )

                if len(audio) == 0:
                    logger.info("대화 자동 종료")
                    break

                logger.info("음성 인식 중")
                text = stt.transcribe(audio)
                if not text:
                    logger.info("인식된 텍스트 없음. 세션 종료")
                    break
                logger.info(f"인식 결과: {text}")

                logger.info("응답 생성 중")
                buffer = ""
                pending = ""
                futures = []
                for token in llm.generate(text):
                    buffer += token
                    if should_flush(buffer):
                        sentences = split_sentences(buffer)
                        if len(sentences) > 1:
                            for s in sentences[:-1]:
                                pending += s + " "
                            buffer = sentences[-1]
                        else:
                            pending += buffer
                            buffer = ""

                        if len(pending) >= _MIN_TTS_CHARS:
                            to_speak = pending.strip()
                            futures.append(executor.submit(
                                _synthesize_task, tts, to_speak))
                            pending = ""

                remaining = (pending + buffer).strip()
                if remaining:
                    futures.append(executor.submit(
                        _synthesize_task, tts, remaining))

                # 합성 완료 대기 및 재생
                for i, f in enumerate(futures):
                    try:
                        result = f.result()
                        if result:
                            audio_out, sr = result
                            audio_output.play(audio_out, sr)
                    except Exception as e:
                        logger.error(f"문장 {i+1} 합성 오류: {e}")

                if futures:
                    audio_output.wait_until_done()

                # 재생 후 짧은 대기 및 마이크 정리 (연속 대화 준비)
                time.sleep(0.5)
                audio_input.clear()
                logger.info("사용자 대화 기다리는 중")

            # 세션 종료 후 호출어 대기 상태로 복귀
            time.sleep(1.0) # 중복 감지 방지를 위해 대기 시간 증가
            audio_input.clear()
            wake_word.reset()
            logger.info("준비 완료. 호출어 대기 시작")

    except KeyboardInterrupt:
        logger.info("프로그램 종료")
        executor.shutdown(wait=False)
    finally:
        audio_input.stop()


if __name__ == "__main__":
    main()
