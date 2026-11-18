from pathlib import Path
import time
import av
import cv2


def saved_frames(path):
    """Decode actual presentation timestamps, including variable-rate sources."""
    if not Path(path).is_file():
        raise FileNotFoundError("Video file does not exist")
    with av.open(str(path)) as container:
        streams = list(container.streams.video)
        if not streams:
            raise ValueError("This file has no video stream")
        stream = streams[0]
        origin = None
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video has missing timestamps. Convert it to a timestamped MP4 before analysis.")
            absolute = float(frame.pts * frame.time_base)
            if origin is None:
                origin = absolute
            start = absolute-origin
            image = frame.to_ndarray(format="bgr24")
            duration = float(frame.duration*frame.time_base) if frame.duration else None
            if previous:
                t, old, _ = previous
                if start <= t:
                    raise ValueError("Video timestamps are non-monotonic")
                yield t, start, old
            previous = start, image, duration
        if previous:
            start, image, duration = previous
            if not duration or duration <= 0:
                rate = float(stream.average_rate or 0)
                if rate <= 0:
                    raise ValueError("Cannot determine the final video frame duration")
                duration = 1/rate
            yield start, start+duration, image
        else:
            raise ValueError("Video contains no decodable frames")


def camera_frames(index, stopped):
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Check its index, Windows camera permission, and other apps using it.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        origin = None
        previous = None
        while not stopped():
            ok, image = cap.read()
            captured = time.monotonic()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames. The partial session can still be reviewed.")
            if origin is None:
                origin = captured
            start = captured-origin
            if previous is not None:
                t, old = previous
                yield t, min(start,t+.1), old
            previous = start,image
        if previous is not None:
            t, old = previous
            yield t,t+1/30,old
    finally:
        cap.release()


def resize(frame):
    h,w = frame.shape[:2]
    scale = min(1, 640/w, 480/h)
    return cv2.resize(frame, (round(w*scale),round(h*scale))) if scale < 1 else frame
