"""Validate official metadata, preserve splits, prepare matching face crops."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.io import loadmat


def prepare(root, output, limit=0):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty: choose a new folder to avoid mixing datasets")
    metadata = root / "metadata.mat"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing {metadata}. Extract the official dataset first.")
    data = loadmat(metadata, squeeze_me=True)
    required = ("recordings", "recording", "frame", "person_identity", "gaze_dir", "split",
                "person_head_bbox", "person_face_bbox", "person_eye_left_bbox", "person_eye_right_bbox")
    for key in required:
        if key not in data:
            raise ValueError(f"metadata.mat missing field: {key}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    names = np.atleast_1d(data["recordings"])
    count = len(np.atleast_1d(data["split"]))
    fields = ["path", "x", "y", "z", "recording", "identity", "frame"]
    rows = {"train": [], "val": [], "test": []}
    rejected = []
    seen = set()
    split_subjects = {}
    for i in range(count if not limit else min(count,limit)):
        try:
            split_id = int(np.atleast_1d(data["split"])[i])
            if split_id == 3:
                continue
            if split_id not in (0,1,2):
                raise ValueError("unknown split")
            split = ("train","val","test")[split_id]
            recording = str(names[int(np.atleast_1d(data["recording"])[i])])
            identity = int(np.atleast_1d(data["person_identity"])[i])
            frame = int(np.atleast_1d(data["frame"])[i])
            subject = (recording, identity)
            if subject in split_subjects and split_subjects[subject] != split:
                raise RuntimeError(f"Subject leakage between official splits: {subject}")
            split_subjects[subject] = split
            source = (root / "imgs" / recording / "head" / f"{identity:06}" / f"{frame:06}.jpg").resolve()
            if not source.is_relative_to(root):
                raise ValueError("unsafe image path")
            if source in seen:
                raise RuntimeError(f"Duplicate image in metadata: {source}")
            seen.add(source)
            gaze = np.asarray(data["gaze_dir"]).reshape(-1,3)[i].astype(float)
            norm = np.linalg.norm(gaze)
            if not np.isfinite(gaze).all() or norm < 1e-6:
                raise ValueError("invalid gaze vector")
            # Train the observable front-facing domain used by this app.
            if gaze[2] >= 0:
                raise ValueError("rear-facing gaze excluded from observable baseline")
            head = np.asarray(data["person_head_bbox"]).reshape(-1,4)[i]
            face = np.asarray(data["person_face_bbox"]).reshape(-1,4)[i]
            for key in ("person_eye_left_bbox", "person_eye_right_bbox"):
                box = np.asarray(data[key]).reshape(-1,4)[i]
                if not np.isfinite(box).all() or np.any(box < 0) or np.any(box[2:] <= 0):
                    raise ValueError("eyes not annotated as visible")
            if not np.isfinite(np.r_[head,face]).all() or np.any(head[2:] <= 0) or np.any(face < 0) or np.any(face[2:] <= 0):
                raise ValueError("invalid head/face crop")
            with Image.open(source) as original:
                original.load()
                image = original.convert("RGB")
                # Official boxes are normalized in the original full camera frame.
                xy = (face[:2]-head[:2])/head[2:]
                wh = face[2:]/head[2:]
                bounds = np.r_[xy, xy+wh] * [image.width,image.height,image.width,image.height]
                x0,y0,x1,y1 = bounds.round().astype(int)
                x0,y0 = max(0,x0),max(0,y0)
                x1,y1 = min(image.width,x1),min(image.height,y1)
                if min(x1-x0,y1-y0) < 24:
                    raise ValueError("face crop too small")
                destination = output / "images" / f"{i:08}.jpg"
                image.crop((x0,y0,x1,y1)).resize((128,128)).save(destination, quality=95)
            gaze /= norm
            rows[split].append(dict(zip(fields, [f"images/{i:08}.jpg", *gaze.tolist(),recording,identity,frame])))
        except (OSError,ValueError,IndexError) as exc:
            rejected.append({"index":i, "reason":str(exc)})
    for split, entries in rows.items():
        with (output/f"{split}.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
    report = dict(counts={k:len(v) for k,v in rows.items()}, rejected=rejected,
                  source=str(root), coordinates="gaze360", crop="official face box projected into head crop",
                  limitations="Frontal visible-eye subset; not the full Gaze360 benchmark. No random frame split.")
    (output/"preprocessing_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if not all(rows.values()):
        raise ValueError("One or more splits are empty. Inspect preprocessing_report.json; do not train yet.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--limit", type=int, default=0, help="Debug only; 0 processes all metadata")
    args = parser.parse_args()
    try:
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        report = prepare(args.root,args.output,args.limit)
        print(json.dumps(report["counts"],indent=2))
    except Exception as exc:
        sys.exit(f"Preprocessing failed: {exc}")
