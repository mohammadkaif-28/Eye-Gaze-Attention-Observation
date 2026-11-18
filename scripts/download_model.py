"""Explicit online setup only. Analysis never downloads assets."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def main():
    destination = Path(__file__).resolve().parents[1]/"models"/"face_landmarker.task"
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        if temporary.stat().st_size < 1000000 or not zipfile.is_zipfile(temporary):
            raise ValueError("Downloaded model is not a valid task bundle")
        temporary.replace(destination)
        print(f"Model ready: {destination}\nSHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Model setup failed: {exc}. Check your internet connection and retry.")
