from pipeline.stt import STT

stt = STT(model_size="small", device="cpu", language="ko")

wav_path = "test_output.wav"
print(f"파일 인식 중: {wav_path}")

result = stt.transcribe(wav_path)
print(f"인식 결과: {result}")
