![Banner](assets/Yunsu_Hwang_Banner.png)

# AI-Speaker

Raspberry Pi 5 기반 Local AI Voice Assistant Pipeline.  
Wake Word 감지부터 TTS 응답까지 모든 처리를 On-Device로 수행합니다.

---

## Demo

![Demo](assets/demo.gif)

---

## Wi-Fi Setting (Captive Portal)

최초 Booting 시 `AI_SPEAKER_SETUP` AP에 접속하면 Captive Portal로 자동 연결됩니다.

![Wi-Fi Setup](assets/captive_1.PNG)

![Wi-Fi Connecting](assets/captive_2.PNG)

---

## Architecture

```
Microphone Input
      │
      ▼
[Wake Word Detector] ── 호출어 미감지 시 대기
      │ 호출어 감지
      ▼
[VAD - Silero VAD] ── 음성 구간 감지 + Noise Filter
      │
      ▼
[STT - Whisper] ── 음성 → Text 변환
      │
      ▼
[LLM - llama.cpp] ── 응답 생성 (Token Streaming)
      │
      ▼
[TTS - MeloTTS] ── Text → 음성 합성 (병렬 처리)
      │
      ▼
Speaker Output
```

---

## Features

- **Wake Word 감지** — 호출어 인식 후 대화 Session 시작, 음성으로 준비 완료 알림
- **VAD** — Silero VAD 기반 음성/소음 구분, 침묵 감지 시 Session 자동 종료
- **STT** — Faster-Whisper Local Inference (한국어)
- **LLM** — llama.cpp 기반 Qwen2.5 Local Inference (OpenBLAS 가속)
- **Streaming TTS** — Token Streaming → 문장 단위 분할 → ThreadPoolExecutor 병렬 합성 → 순차 재생
- **종료 명령** — "종료", "꺼줘" 등 음성 명령으로 System Shutdown
- **Memory** — ChromaDB 기반 대화 내용 Vector 저장 및 검색
- **Web Search Tool** — DuckDuckGo 검색 Tool 연동
- **Captive Portal** — 최초 Booting 시 AP 모드로 Wi-Fi 설정 지원

---

## Tech Stack

| 항목        | 내용                                           |
| --------- | -------------------------------------------- |
| Platform  | Raspberry Pi 5 (Raspberry Pi OS Lite 64-bit) |
| Language  | Python 3.10                                  |
| Wake Word | openwakeword                                 |
| VAD       | Silero VAD                                   |
| STT       | Faster-Whisper                               |
| LLM       | llama-cpp-python (Qwen2.5-3B-Instruct GGUF)  |
| TTS       | MeloTTS (한국어)                                |
| Memory    | ChromaDB                                     |
| 검색        | DuckDuckGo Search (ddgs)                     |
| 병렬처리      | ThreadPoolExecutor                           |

---

## Getting Started (Raspberry Pi 5)

**1. System Packages**
```bash
sudo apt update
sudo apt install -y portaudio19-dev libsndfile1-dev ffmpeg python3.10 python3.10-dev python3.10-venv build-essential cmake
```

**2. Python Packages**
```bash
pip install sounddevice numpy openwakeword silero-vad faster-whisper ddgs chromadb
```

**3. llama-cpp-python (ARM64)**
```bash
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

**4. MeloTTS**
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS && pip install -e .
python -m unidic download
```

**5. Model Download**
```bash
python -c "import openwakeword; openwakeword.utils.download_models()"
huggingface-cli download bartowski/Qwen2.5-3B-Instruct-GGUF Qwen2.5-3B-Instruct-Q4_K_M.gguf --local-dir ./models
```

---

## Settings

`config.yaml.example`을 참고하여 `config.yaml`을 작성합니다.

---

## Execute

```bash
python main.py
```

---

## License

본 Project Source Code는 [MIT License](LICENSE)를 따릅니다.

사용된 Open Source Library:

| Library          | License    |
| ---------------- | ---------- |
| MeloTTS          | MIT        |
| Faster-Whisper   | MIT        |
| Silero VAD       | MIT        |
| llama-cpp-python | MIT        |
| openwakeword     | Apache 2.0 |
| ChromaDB         | Apache 2.0 |
