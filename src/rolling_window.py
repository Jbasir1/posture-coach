"""
Rolling window evaluator — the "Wisdom" layer from the proposal.

Takes a stream of per-frame binary posture labels (True = bad posture) and
decides when to fire an alert:
  - If >= BAD_FRACTION_THRESHOLD of the frames in the last WINDOW_SECONDS
    are "bad", the state becomes BAD and an alert fires.
  - Once in BAD state, we don't re-alert every frame — we wait for
    RESET_GOOD_SECONDS of continuous good posture before re-arming.

This mirrors proposal section 3 (Wisdom) and the Mitigation 2 fallback
(stretching the window from 10s to 30s if false alerts are a problem) —
just change WINDOW_SECONDS below.
"""

import time
from collections import deque


class RollingWindowEvaluator:
    def __init__(
        self,
        window_seconds: float = 10.0,
        bad_fraction_threshold: float = 0.70,
        reset_good_seconds: float = 3.0,
    ):
        self.window_seconds = window_seconds
        self.bad_fraction_threshold = bad_fraction_threshold
        self.reset_good_seconds = reset_good_seconds

        # Each entry: (timestamp, is_bad: bool)
        self._history = deque()

        self._alerting = False
        self._good_streak_start = None

    def update(self, is_bad: bool, now: float | None = None) -> bool:
        """
        Feed one frame's posture label in. Returns True exactly on the
        frame where a NEW alert should fire (edge-triggered, not level).
        """
        if now is None:
            now = time.time()

        self._history.append((now, is_bad))
        self._trim(now)

        # Track continuous good streak, used to reset out of alerting state.
        if is_bad:
            self._good_streak_start = None
        else:
            if self._good_streak_start is None:
                self._good_streak_start = now
            elif (now - self._good_streak_start) >= self.reset_good_seconds:
                self._alerting = False

        if self._alerting:
            return False  # already alerting, don't re-fire every frame

        bad_fraction = self._bad_fraction()
        if bad_fraction >= self.bad_fraction_threshold and self._window_is_full(now):
            self._alerting = True
            return True

        return False

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _bad_fraction(self) -> float:
        if not self._history:
            return 0.0
        bad_count = sum(1 for _, is_bad in self._history if is_bad)
        return bad_count / len(self._history)

    def _window_is_full(self, now: float) -> bool:
        """Require at least window_seconds of real data before alerting,
        so we don't fire after just 1 bad frame at startup."""
        if not self._history:
            return False
        oldest_ts = self._history[0][0]
        return (now - oldest_ts) >= (self.window_seconds * 0.9)

    @property
    def is_alerting(self) -> bool:
        return self._alerting
