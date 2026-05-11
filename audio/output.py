import queue
import threading
import numpy as np
import sounddevice as sd


class AudioOutput:
    def __init__(self, device=None):
        self.device = device
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def _play_loop(self):
        while not self._stop_event.is_set():
            try:
                # 큐에서 오디오 데이터를 가져옴
                item = self.queue.get(timeout=0.1)
                if item is None:
                    break

                audio, sr = item
                try:
                    sd.play(audio, sr, device=self.device)
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
        """오디오 데이터를 재생 큐에 넣습니다. (Non-blocking)"""
        self.queue.put((audio, sample_rate))

    def wait_until_done(self):
        """현재 큐에 있는 모든 오디오가 재생될 때까지 기다립니다."""
        self.queue.join()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
