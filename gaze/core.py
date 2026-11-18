from collections import defaultdict
from dataclasses import dataclass, asdict
import math
from statistics import mean, median

DIRECTIONS = ("left", "right", "up", "down", "towards camera")


def timestamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and non-negative")
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"


@dataclass(frozen=True)
class Observation:
    direction: str = "indeterminate"
    status: str = "not observable"
    confidence: float = 0.0
    reason: str = ""

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.status != "valid" and self.direction != "indeterminate":
            raise ValueError("Unavailable observations cannot contain a gaze estimate")
        if self.status == "valid" and self.direction not in DIRECTIONS:
            raise ValueError("Unknown gaze direction")


@dataclass
class Interval:
    start: float
    end: float
    direction: str
    status: str
    confidence: float
    reason: str

    @property
    def duration(self):
        return self.end - self.start

    def record(self, session_id, index):
        return dict(observation_id=index, session_id=session_id,
                    timestamp_start=timestamp(self.start), timestamp_end=timestamp(self.end),
                    start_seconds=self.start, end_seconds=self.end,
                    direction_or_target=self.direction, duration=self.duration,
                    confidence=round(self.confidence, 4),
                    confidence_label="High" if self.confidence >= .8 else "Medium" if self.confidence >= .6 else "Low",
                    tracking_status=self.status, quality_reason=self.reason)


class Timeline:
    """Conservative dwell filter: unconfirmed changes remain unavailable.

    Bad frames are never bridged into a valid direction. Confirmed transitions
    are backdated to the candidate's first frame. Inputs cover [start, end).
    """
    def __init__(self, dwell=.15):
        self.dwell = dwell
        self.intervals = []
        self.pending = []
        self.stable = None
        self.last_end = 0.0

    def _append(self, start, end, obs):
        if end <= start:
            return
        prev = self.intervals[-1] if self.intervals else None
        if prev and (prev.direction, prev.status, prev.reason) == (obs.direction, obs.status, obs.reason) and abs(prev.end-start) < 1e-6:
            prev.confidence = (prev.confidence*prev.duration + obs.confidence*(end-start))/(end-prev.start)
            prev.end = end
        else:
            self.intervals.append(Interval(start, end, obs.direction, obs.status, obs.confidence, obs.reason))

    def _flush(self, confirmed=False):
        for start, end, obs in self.pending:
            self._append(start, end, obs if confirmed else Observation(reason="direction transition unconfirmed"))
        self.pending.clear()

    def add(self, start, end, obs):
        if not all(math.isfinite(x) for x in (start, end)) or start < self.last_end-1e-6 or end <= start:
            raise ValueError("Frame times must increase without overlap")
        if start > self.last_end + 1e-6:
            self._flush()
            self._append(self.last_end, start, Observation(reason="capture gap"))
            self.stable = None
        self.last_end = end
        if obs.status != "valid":
            self._flush()
            self._append(start, end, obs)
            self.stable = None
            return
        if obs.direction == self.stable:
            self._flush()
            self._append(start, end, obs)
            return
        if self.pending and self.pending[-1][2].direction != obs.direction:
            self._flush()
        self.pending.append((start, end, obs))
        if end-self.pending[0][0] >= self.dwell:
            self.stable = obs.direction
            self._flush(confirmed=True)

    def finish(self):
        self._flush()
        return self.intervals


def summarize(intervals):
    totals = defaultdict(float)
    valid = [i for i in intervals if i.status == "valid"]
    for i in valid:
        totals[i.direction] += i.duration
    analysable = sum(totals.values())
    total = sum(i.duration for i in intervals)
    # Unknown gaps break continuity; no inferred shifts across missing data.
    shifts = [dict(timestamp=timestamp(b.start), start_seconds=b.start,
                   from_direction=a.direction, to_direction=b.direction)
              for a, b in zip(intervals, intervals[1:])
              if a.status == b.status == "valid" and a.direction != b.direction]
    return dict(total_duration=total, analysable_duration=analysable,
                unavailable_duration=total-analysable,
                directions={d: dict(seconds=totals[d], percent=100*totals[d]/analysable if analysable else 0)
                            for d in DIRECTIONS},
                gaze_shift_count=len(shifts), gaze_shifts=shifts,
                average_valid_interval=mean(i.duration for i in valid) if valid else 0,
                median_valid_interval=median(i.duration for i in valid) if valid else 0,
                confidence_note="Heuristic tracking quality, not calibrated probability of gaze correctness")
