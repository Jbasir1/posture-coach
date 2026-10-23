"""
Local desktop alert — the final box in the proposal's system diagram
("Local Desktop Chime / Alert"). Cross-platform simple beep, no audio
files needed for the MVR.
"""

import sys


def beep() -> None:
    try:
        if sys.platform == "darwin":
            import os
            os.system("afplay /System/Library/Sounds/Ping.aiff")
        elif sys.platform.startswith("win"):
            import winsound
            winsound.Beep(880, 300)
        else:
            # Linux / fallback: terminal bell
            print("\a", end="", flush=True)
    except Exception:
        # Never let a broken audio backend crash the posture loop.
        print("\a", end="", flush=True)
