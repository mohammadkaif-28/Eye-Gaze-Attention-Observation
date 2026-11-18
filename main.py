import sys
from pathlib import Path

if __name__ == "__main__":
    try:
        from gaze.ui import run
        run(Path(__file__).resolve().parent)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run setup.ps1, then run.ps1.")
