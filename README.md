# speaker

완전한 로컬 AI 음성 Assistant Pipeline.  
호출어 감지부터 음성 합성까지 모든 처리를 로컬에서 수행합니다.

---

## 실행 화면

![Demo](docs/demo.gif)
<!-- Demo GIF 또는 Screenshot 추가 후 위 경로 업데이트 -->

---

## Pipeline 구조

```
Microphone Input
      │
      ▼
[Wake Word Detector] ── 호출어 미감지 시 대기
      │ 호출어 감지
      ▼
[VAD - Silero VAD] ── 음성 구간 감지 + Noise 필터
      │
      ▼
[STT - Whisper] ── 음성 → 텍스트 변환
      │
      ▼
[LLM - llama.cpp] ── 응답 생성 (Token Streaming)
      │
      ▼
[TTS - MeloTTS] ── 텍스트 → 음성 합성 (병렬 처리)
      │
      ▼
Speaker Output
```

---

## 주요 기능

- **Wake Word 감지** — 호출어 인식 후 대화 Session 시작
- **VAD** — Silero VAD 기반 음성/소음 구분, 침묵 감지 시 자동 종료
- **STT** — Whisper 로컬 추론 (한국어 지원)
- **LLM** — llama.cpp 기반 로컬 LLM (GPU Layer 설정 가능)
- **Streaming TTS** — LLM Token Streaming → 문장 단위 분할 → ThreadPoolExecutor 병렬 합성 → 순차 재생
- **Memory / Tool** — 대화 기억 저장, 검색 Tool 연동

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| 언어 | Python 3.12+ |
| Wake Word | 커스텀 Detector |
| VAD | Silero VAD |
| STT | Whisper (로컬) |
| LLM | llama.cpp (로컬) |
| TTS | MeloTTS (한국어) |
| 병렬처리 | ThreadPoolExecutor |

---

## 설치

```bash
git clone https://github.com/Hoporing/speaker.git
cd speaker

# 의존성 설치
pip install torch torchaudio
pip install git+https://github.com/myshell-ai/MeloTTS.git
pip install openai-whisper
pip install llama-cpp-python
pip install PyAudio numpy pyyaml
```

---

## 설정

`config.yaml`을 작성합니다. (`config.yaml.example` 참고)

```yaml
audio:
  sample_rate: 16000
  chunk_size: 512
  input_device: null   # null이면 기본 Device
  output_device: null

wake_word:
  model: "your_wake_word_model"
  threshold: 0.5

vad:
  model_path: "models/silero_vad.onnx"
  threshold: 0.5

stt:
  model_path: "models/whisper/medium"
  device: "cuda"
  language: "ko"

llm:
  model_path: "models/llm/your_model.gguf"
  n_ctx: 4096
  n_gpu_layers: 35
  max_tokens: 512
  temperature: 0.7

tts:
  ckpt_path: "models/melo/checkpoint.pth"
  config_path: "models/melo/config.json"
  bert_path: "models/melo/bert"
  language: "KR"
  speed: 1.0
```

---

## 실행

```bash
python main.py
```

---

## License

본 프로젝트 소스코드는 [MIT License](LICENSE)를 따릅니다.

사용된 오픈소스 라이브러리:

| Library | License |
|---------|---------|
| MeloTTS | MIT |
| Whisper (OpenAI) | MIT |
| Silero VAD | MIT |
| llama-cpp-python | MIT |
