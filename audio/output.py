import queue
import threading
import numpy as np
import sounddevice as sd
from scipy import signal


class AudioOutput:
    def __init__(self, device=None):
        self.device = device
        self.queue = queue.Queue()
        self._stop_event = threading.Event()

        device_info = sd.query_devices(device, 'output')
        self._native_rate = int(device_info['default_samplerate'])

        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:
                    break

                audio, sr = item
                try:
                    if sr != self._native_rate:
                        resampled_len = int(len(audio) * self._native_rate / sr)
                        audio = signal.resample(audio, resampled_len)
                    sd.play(audio, self._native_rate, device=self.device)
                    sd.wait()
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Playback error: {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Audio thread error: {e}")
                continue

    def play(self, audio: np.ndarray, sample_rate: int):
        self.queue.put((audio, sample_rate))

    def wait_until_done(self):
        self.queue.join()

    def stop(self):
        self._stop_event.set()
        self._thread.join()