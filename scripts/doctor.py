"""Report setup readiness without opening the camera or downloading anything."""
import importlib
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
failed=False
print("Python:",sys.version.split()[0])
for name in ("numpy","cv2","mediapipe","PySide6","av","torch","torchvision","scipy","PIL"):
    try:
        module=importlib.import_module(name)
        print(name,"OK",getattr(module,"__version__",""))
    except Exception as exc:
        failed=True; print(name,"MISSING/ERROR",str(exc))
for path in (root/"models"/"face_landmarker.task",root/"data"/"processed"/"train.csv"):
    print(path.name,"READY" if path.is_file() else "NOT YET PRESENT")
sys.exit(1 if failed else 0)
