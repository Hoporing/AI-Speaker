import re
import yaml
import soundfile as sf
from pipeline.llm import LLM
from pipeline.tts import TTS

_MAX_BUFFER_CHARS = 50
_MIN_TTS_CHARS = 10


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。,，\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def should_flush(buffer: str) -> bool:
    return bool(re.search(r"[.!?。,，\n]", buffer)) or len(buffer) >= _MAX_BUFFER_CHARS


def main():
    config = load_config()

    print("모델 로딩 중...")
    llm = LLM(
        model_path=config["llm"]["model_path"],
        n_ctx=config["llm"]["n_ctx"],
        n_gpu_layers=config["llm"]["n_gpu_layers"],
        max_tokens=config["llm"]["max_tokens"],
        temperature=config["llm"]["temperature"],
    )
    tts = TTS(
        language=config["tts"]["language"],
        speed=config["tts"]["speed"],
    )
    print("준비 완료. 'quit' 입력 시 종료.\n")

    chunk_index = 0

    while True:
        user_input = input("입력 > ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        print("응답 > ", end="", flush=True)
        buffer = ""
        pending = ""
        chunk_index = 0

        for token in llm.generate(user_input):
            print(token, end="", flush=True)
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
                    audio, sr = tts.synthesize(to_speak)
                    path = f"test_output_{chunk_index}.wav"
                    sf.write(path, audio, sr)
                    print(f"\n  [TTS #{chunk_index}] {to_speak[:30]}{'...' if len(to_speak) > 30 else ''} → {path}", flush=True)
                    print("응답 > ", end="", flush=True)
                    chunk_index += 1
                    pending = ""

        remaining = (pending + buffer).strip()
        if remaining:
            audio, sr = tts.synthesize(remaining)
            path = f"test_output_{chunk_index}.wav"
            sf.write(path, audio, sr)
            print(f"\n  [TTS #{chunk_index}] {remaining[:30]}{'...' if len(remaining) > 30 else ''} → {path}", flush=True)

        print()


if __name__ == "__main__":
    main()
