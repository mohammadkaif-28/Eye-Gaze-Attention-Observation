from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from .core import Observation


def classify(horizontal, vertical):
    """Unmirrored image coordinates: left/right refer to the viewer's image."""
    if abs(horizontal) < .12 and abs(vertical) < .16:
        return "towards camera"
    if abs(horizontal)/.12 >= abs(vertical)/.16:
        return "right" if horizontal > 0 else "left"
    return "down" if vertical > 0 else "up"


class Estimator:
    def __init__(self, model, checkpoint=None):
        if not Path(model).is_file():
            raise FileNotFoundError("Face model missing. Run python scripts/download_model.py while online, then restart.")
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        self.detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.CPU),
            running_mode=RunningMode.VIDEO, num_faces=2,
            min_face_detection_confidence=.65, min_face_presence_confidence=.65,
            min_tracking_confidence=.65))
        self.last_ms = -1
        self.predictor = None
        if checkpoint:
            from .learning import Predictor
            self.predictor = Predictor(checkpoint)

    def close(self):
        self.detector.close()

    def process(self, frame, seconds):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_ms = max(self.last_ms+1, round(seconds*1000))
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), self.last_ms)
        if not result.face_landmarks:
            return Observation(status="face not detected", reason="face missing or occluded"), frame
        if len(result.face_landmarks) != 1:
            return Observation(reason="multiple faces; isolate the intended participant"), frame
        points = np.array([(p.x*w, p.y*h) for p in result.face_landmarks[0]], dtype=np.float64)
        annotated = frame.copy()
        if len(points) < 478 or not np.isfinite(points).all():
            return Observation(reason="invalid eye landmarks"), annotated
        for i in (33, 133, 159, 145, 362, 263, 386, 374, 468, 473):
            cv2.circle(annotated, tuple(points[i].astype(int)), 2, (190, 220, 80), -1)
        if not np.isfinite(points).all() or len(points) < 478:
            return Observation(reason="invalid eye landmarks"), annotated
        eye_ids = [33, 133, 159, 145, 362, 263, 386, 374, 468, 473]
        eyes = points[eye_ids]
        if np.any(eyes[:, 0] < 0) or np.any(eyes[:, 0] >= w) or np.any(eyes[:, 1] < 0) or np.any(eyes[:, 1] >= h):
            return Observation(reason="eyes outside image"), annotated
        x0, y0 = np.maximum(points.min(axis=0).astype(int), 0)
        x1, y1 = np.minimum(points.max(axis=0).astype(int)+1, [w, h])
        face = gray[y0:y1, x0:x1]
        if face.size == 0 or x1-x0 < 80:
            return Observation(reason="face too small for iris observation"), annotated
        brightness = float(face.mean())
        sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())
        if brightness < 40 or brightness > 225 or sharpness < 35:
            return Observation(status="low confidence", reason="poor face lighting or blur"), annotated
        ratios = []
        for left, right, top, bottom, iris in [(33,133,159,145,468), (362,263,386,374,473)]:
            a, b = sorted((points[left], points[right]), key=lambda p: p[0])
            axis = b-a
            width = np.linalg.norm(axis)
            height = np.linalg.norm(points[bottom]-points[top])
            if width < 12 or height/width < .14:
                return Observation(reason="eyes closed / blink or eyes not observable"), annotated
            ex = axis/width
            ey = np.array([-ex[1], ex[0]])
            center = (a+b)/2
            rx = np.dot(points[iris]-center, ex)/width
            ry = np.dot(points[iris]-(points[top]+points[bottom])/2, ey)/height
            if abs(rx) > .5 or abs(ry) > .65:
                return Observation(reason="iris geometry unreliable / possible occlusion"), annotated
            # Saturation is only a quality flag, never a claim to detect glasses.
            lo = np.maximum(np.floor(np.minimum(a,b)-[3,height]).astype(int), 0)
            hi = np.minimum(np.ceil(np.maximum(a,b)+[3,height]).astype(int), [w,h])
            crop = gray[lo[1]:hi[1], lo[0]:hi[0]]
            if crop.size and (crop > 248).mean() > .2:
                return Observation(status="low confidence", reason="eye saturation / possible glare; review"), annotated
            ratios.append((rx, ry))
        if np.linalg.norm(np.subtract(*ratios)) > .35:
            return Observation(status="low confidence", reason="inconsistent eye estimates"), annotated
        # Generic face points in camera convention: x right, y down, z into face.
        model = np.array([(0,0,0),(0,330,65),(-225,-170,135),(225,-170,135),(-150,150,125),(150,150,125)], dtype=np.float64)
        pixels = points[[1,152,33,263,61,291]]
        camera = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
        ok, rotation, translation = cv2.solvePnP(model, pixels, camera, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or translation[2,0] <= 0:
            return Observation(reason="head pose unavailable"), annotated
        projected, _ = cv2.projectPoints(model, rotation, translation, camera, np.zeros(4))
        reprojection = np.linalg.norm(projected.reshape(-1,2)-pixels, axis=1).mean()/max(x1-x0,1)
        matrix, _ = cv2.Rodrigues(rotation)
        pitch, yaw, roll = cv2.RQDecomp3x3(matrix)[0]
        if not np.isfinite([pitch,yaw,roll,reprojection]).all() or abs(yaw)>40 or abs(pitch)>35 or abs(roll)>35 or reprojection>.12:
            return Observation(reason="head turned / pose unreliable"), annotated
        rx, ry = np.mean(ratios, axis=0)
        # Face forward is negative z in the generic head coordinate system.
        horizontal = float(rx-yaw/150)
        vertical = float(ry+pitch/150)
        quality = min(.9, .9-max(abs(yaw)/180, abs(pitch)/160), .65+max(0,sharpness-35)/500)
        if quality < .6:
            return Observation(status="low confidence", confidence=max(0,quality), reason="marginal pose / eye visibility"), annotated
        if self.predictor:
            direction = self.predictor.predict(frame[y0:y1, x0:x1])
            if direction == "indeterminate":
                return Observation(reason="trained gaze points outside observable frontal region"), annotated
            return Observation(direction, "valid", quality, "trained model; tracking quality is not model confidence"), annotated
        direction = classify(horizontal, vertical)
        # Do not label cancelling head/eye deviations as towards camera.
        if direction == "towards camera" and (abs(yaw)>12 or abs(pitch)>12 or abs(rx)>.12 or abs(ry)>.16):
            return Observation(reason="head and eye estimates disagree"), annotated
        return Observation(direction, "valid", quality, "heuristic estimate; verify against video"), annotated
