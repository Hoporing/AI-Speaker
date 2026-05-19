import queue
import math
import numpy as np
import sounddevice as sd
from scipy import signal

class AudioInput:
    def __init__(self, sample_rate=16000, chunk_size=512, device=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._queue = queue.Queue()
        self._stream = None

        device_info = sd.query_devices(device, 'input')
        self._native_rate = int(device_info['default_samplerate'])
        self._native_chunk = round(chunk_size * self._native_rate / self.sample_rate)

    def _callback(self, indata, frames, time, status):
        audio = indata[:, 0].copy()
        if self._native_rate != self.sample_rate:
            audio = signal.resample(audio, self.chunk_size)
        self._queue.put(audio)

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self._native_rate,
            channels=1,
            dtype="float32",
            blocksize=self._native_chunk,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def read(self, timeout=1.0):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        while not self._queue.empty():
            self._queue.get_nowait()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()