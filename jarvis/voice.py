"""
Push-to-talk voice: record while the other hand pinches, transcribe locally
with Whisper (mlx-whisper, runs on the Apple GPU), then either run a command
from jarvis.toml or type the text into the focused field.
"""

import queue
import threading

import numpy as np

SAMPLE_RATE = 16000
MIN_SPEECH_S = 0.3
MIN_RMS = 0.004     # quieter than this is silence; Whisper hallucinates on silence
MAX_RECORD_S = 30


class Voice:
    def __init__(self, model, on_text, on_status=print):
        """on_text(text) is called on the worker thread with each transcript."""
        self.model = model
        self.on_text = on_text
        self.on_status = on_status
        self.stream = None
        self.chunks = []
        self.recording = False
        self.ready = False
        self.jobs = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------- recording ----------------
    def start(self):
        if self.recording:
            return
        import sounddevice as sd
        self.chunks = []
        try:
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                         dtype="float32", callback=self._on_audio)
            self.stream.start()
        except Exception as e:
            self.on_status(f"mic error: {e}")
            self.stream = None
            return
        self.recording = True

    def _on_audio(self, data, frames, t, status):
        if len(self.chunks) * frames < MAX_RECORD_S * SAMPLE_RATE:
            self.chunks.append(data[:, 0].copy())

    def stop(self):
        """Stop recording and queue the clip. Returns False if it was discarded."""
        if not self.recording:
            return False
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        if len(audio) < MIN_SPEECH_S * SAMPLE_RATE or np.sqrt(np.mean(audio ** 2)) < MIN_RMS:
            return False
        self.jobs.put(audio)
        return True

    def cancel(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.stream = None

    @property
    def busy(self):
        return not self.jobs.empty() or self._transcribing

    # ---------------- transcription ----------------
    _transcribing = False

    def _worker(self):
        import mlx_whisper
        # Warm up: downloads the model on first run and compiles kernels, so
        # the first real utterance isn't slow.
        try:
            mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, np.float32), path_or_hf_repo=self.model)
            self.ready = True
        except Exception as e:
            self.on_status(f"whisper failed to load: {e}")
            return
        while True:
            audio = self.jobs.get()
            self._transcribing = True
            try:
                r = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model,
                                           condition_on_previous_text=False,
                                           language="en")
                segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.6]
                text = " ".join(s["text"].strip() for s in segs).strip()
                if text:
                    self.on_text(text)
            except Exception as e:
                self.on_status(f"transcribe error: {e}")
            finally:
                self._transcribing = False
