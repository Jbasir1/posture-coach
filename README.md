# Real Time Posture Coach

Edge-device posture monitor: webcam → MediaPipe Pose → posture classifier →
rolling-window evaluator → local alert. Runs entirely on-device, no cloud.

This repo currently implements the **Minimum Viable Result (MVR)** from the
proposal: a terminal app that prints "Good" / "Bad" posture at ~10fps using
an angle-threshold rule, and beeps after 10 straight seconds of bad posture.

## Status

- [x] MediaPipe Pose wired up to live webcam feed
- [x] Angle-threshold baseline classifier (torso + neck angle vs. vertical)
- [x] Rolling 10s window evaluator with 3s good-streak reset
- [x] Local beep alert
- [ ] Data collection (Week 7)
- [ ] Labeled dataset + feature extraction script (Week 7–8)
- [ ] Random Forest / MLP classifier trained on real data (Week 10)
- [ ] Latency profiling (Week 12)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

A preview window shows the pose skeleton and current status. Press `q` in
that window (or Ctrl+C in the terminal) to quit. Terminal output looks like:

```
[14:32:07] status=GOOD torso=  4.2 deg  neck=  8.1 deg
[14:32:07] status=BAD  torso= 22.7 deg  neck= 31.4 deg
>>> ALERT: sustained bad posture — sit up! <<<
```

## Project structure

```
posture-coach/
├── src/
│   ├── main.py                # capture loop, ties everything together
│   ├── posture_classifier.py  # angle-threshold baseline (MVR model)
│   ├── rolling_window.py      # 10s/70% rolling window + 3s reset (Wisdom layer)
│   └── alert.py                # cross-platform beep
├── requirements.txt
└── README.md
```

## Tuning

- `TORSO_ANGLE_THRESHOLD_DEG` / `NECK_ANGLE_THRESHOLD_DEG` in
  `posture_classifier.py` — lower these if the system misses real slouching,
  raise them if it fires too often.
- `window_seconds` / `bad_fraction_threshold` in `RollingWindowEvaluator`
  (set in `main.py`) — proposal's Mitigation 2 suggests widening from 10s to
  30s if false alerts become annoying.

## Notes

- `model_complexity=0` in `main.py` picks MediaPipe's fastest pose model,
  which matters for the <15ms/frame latency target in the proposal.
- Landmarks below `MIN_LANDMARK_VISIBILITY` cause the frame to be dropped
  rather than classified — this is Mitigation 1 for Risk 1 (noisy keypoints
  from bad lighting/angles).
