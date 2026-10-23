"""
Baseline heuristic posture classifier — this is the MVR's
"basic binary model based on torso angles," i.e. a threshold rule,
NOT the Random Forest / MLP that comes later (proposal Week 10).

We use MediaPipe Pose landmarks:
  - 11, 12 = left/right shoulder
  - 23, 24 = left/right hip
  - 7, 8   = left/right ear

Two angles, each measured from vertical (0 degrees = perfectly upright):
  1. Torso angle: hip-midpoint -> shoulder-midpoint, vs. vertical.
  2. Neck angle: shoulder-midpoint -> ear-midpoint, vs. vertical.

If either angle exceeds its threshold, posture is classified "Bad"
(this folds the proposal's "Slouched" and "Forward Head" classes into
one binary label, matching the MVR's binary scope).
"""

import math
from dataclasses import dataclass



LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_EAR, RIGHT_EAR = 7, 8


TORSO_ANGLE_THRESHOLD_DEG = 15.0
NECK_ANGLE_THRESHOLD_DEG = 25.0


MIN_LANDMARK_VISIBILITY = 0.5


@dataclass
class PostureResult:
    is_bad: bool
    torso_angle_deg: float
    neck_angle_deg: float
    low_confidence: bool  


def _midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _angle_from_vertical_deg(top_point, bottom_point) -> float:
    """Angle between the vector (bottom -> top) and straight up, in degrees.
    0 = perfectly vertical/upright. Larger = more forward lean."""
    dx = top_point[0] - bottom_point[0]
    dy = bottom_point[1] - top_point[1]  
    angle_rad = math.atan2(abs(dx), abs(dy)) if dy != 0 else math.pi / 2
    return math.degrees(angle_rad)


def classify(landmarks) -> PostureResult:
    """
    landmarks: the .pose_landmarks.landmark list from MediaPipe Pose,
    where each item has .x, .y, .z, .visibility.
    """
    required = [
        landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER],
        landmarks[LEFT_HIP], landmarks[RIGHT_HIP],
        landmarks[LEFT_EAR], landmarks[RIGHT_EAR],
    ]
    if any(lm.visibility < MIN_LANDMARK_VISIBILITY for lm in required):
        return PostureResult(is_bad=False, torso_angle_deg=0.0,
                              neck_angle_deg=0.0, low_confidence=True)

    shoulder_mid = _midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    hip_mid = _midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    ear_mid = _midpoint(landmarks[LEFT_EAR], landmarks[RIGHT_EAR])

    torso_angle = _angle_from_vertical_deg(shoulder_mid, hip_mid)
    neck_angle = _angle_from_vertical_deg(ear_mid, shoulder_mid)

    is_bad = (
        torso_angle > TORSO_ANGLE_THRESHOLD_DEG
        or neck_angle > NECK_ANGLE_THRESHOLD_DEG
    )

    return PostureResult(
        is_bad=is_bad,
        torso_angle_deg=torso_angle,
        neck_angle_deg=neck_angle,
        low_confidence=False,
    )
