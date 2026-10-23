"""
Real Time Posture Coach — Minimum Viable Result (proposal section 5).

A terminal app that reads the webcam, runs MediaPipe Pose, classifies
posture with a simple angle threshold (see posture_classifier.py), prints
"Good" / "Bad" at ~10fps, and beeps once posture has been bad for
10 straight seconds (see rolling_window.py).

Run:
    python src/main.py

Press 'q' in the preview window, or Ctrl+C in the terminal, to quit.
"""

import time

import cv2
import mediapipe as mp

from posture_classifier import classify
from rolling_window import RollingWindowEvaluator
from alert import beep

TARGET_FPS = 10
FRAME_INTERVAL_SEC = 1.0 / TARGET_FPS

SHOW_PREVIEW_WINDOW = True  


def main() -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0). "
              "Check camera permissions / that no other app is using it.")
        return

    evaluator = RollingWindowEvaluator(
        window_seconds=10.0,
        bad_fraction_threshold=0.70,
        reset_good_seconds=3.0,
    )

    print(f"Posture Coach running at target {TARGET_FPS} fps. Ctrl+C to quit.\n")

    last_tick = 0.0
    try:
        with mp_pose.Pose(
            model_complexity=0,  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("WARNING: failed to read frame from webcam.")
                    break

                now = time.time()
                if now - last_tick < FRAME_INTERVAL_SEC:
                    # Throttle to ~10 fps so we don't burn CPU faster than needed.
                    if SHOW_PREVIEW_WINDOW:
                        cv2.imshow("Posture Coach", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue
                last_tick = now

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                status_text = "NO PERSON DETECTED"
                if results.pose_landmarks:
                    posture = classify(results.pose_landmarks.landmark)

                    if posture.low_confidence:
                        status_text = "LOW CONFIDENCE (frame dropped)"
                    else:
                        status_text = "BAD" if posture.is_bad else "GOOD"
                        fired = evaluator.update(posture.is_bad, now=now)
                        if fired:
                            print(">>> ALERT: sustained bad posture — sit up! <<<")
                            beep()

                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"status={status_text:4s} "
                            f"torso={posture.torso_angle_deg:5.1f} deg  "
                            f"neck={posture.neck_angle_deg:5.1f} deg"
                        )

                    if SHOW_PREVIEW_WINDOW:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                        )
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] {status_text}")

                if SHOW_PREVIEW_WINDOW:
                    cv2.putText(
                        frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                    cv2.imshow("Posture Coach", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("\nStopping Posture Coach.")
    finally:
        cap.release()
        if SHOW_PREVIEW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
