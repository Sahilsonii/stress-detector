"""Shared live keystroke-timing capture + feature extraction.

Produces the exact same 5-dim feature vector (hold_mean, hold_std,
flight_mean, flight_std, total_duration) that train_keystroke_rf.py computes
from the CMU DSL-StrongPasswordData, so a model trained there can be applied
to live typing here without a schema mismatch. Used by:
  - keystroke_timing_logger.py (standalone session capture for the paired
    facial+keystroke fusion evaluation)
  - combined_stress_monitor.py / modern_stress_monitor.py (live fusion UI)
"""

from pynput import keyboard
import numpy as np

FEATURE_NAMES = ["hold_mean", "hold_std", "flight_mean", "flight_std", "total_duration"]


class LiveKeystrokeTimer:
    """Records per-key down/up timestamps and derives the same hold/flight/
    duration features used to train the keystroke Random Forest."""

    def __init__(self):
        self._down_times = {}
        self._events = []  # (key_char, down_ts, up_ts)
        self.listener = None

    def start(self):
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None

    def _on_press(self, key):
        import time
        k = getattr(key, "char", None) or str(key)
        if k not in self._down_times:
            self._down_times[k] = time.time()

    def _on_release(self, key):
        import time
        k = getattr(key, "char", None) or str(key)
        down_ts = self._down_times.pop(k, None)
        if down_ts is not None:
            self._events.append((k, down_ts, time.time()))

    def reset(self):
        self._down_times.clear()
        self._events.clear()

    def extract_features(self):
        """Returns a (1, 5) feature vector, or None if fewer than 2 keys were
        captured (not enough to compute flight/duration statistics)."""
        events = sorted(self._events, key=lambda e: e[1])
        if len(events) < 2:
            return None
        holds = np.array([up - down for _, down, up in events])
        flights = np.array([
            events[i][1] - events[i - 1][2] for i in range(1, len(events))
        ])
        dd = np.array([
            events[i][1] - events[i - 1][1] for i in range(1, len(events))
        ])
        total_duration = float(dd.sum())
        feats = np.array([[holds.mean(), holds.std(), flights.mean(), flights.std(), total_duration]])
        return feats

    def event_count(self):
        return len(self._events)
