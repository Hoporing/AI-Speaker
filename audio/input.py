import queue
import numpy as np
import sounddevice as sd


class AudioInput:
    def __init__(self, sample_rate=16000, chunk_size=1280, device=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time, status):
        self._queue.put(indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
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
