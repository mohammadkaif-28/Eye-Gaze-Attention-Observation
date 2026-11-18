"""Offline batch analysis without the desktop UI."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaze.core import Timeline
from gaze.storage import Store

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--consent-confirmed",action="store_true")
    p.add_argument("--session-id")
    p.add_argument("--checkpoint")
    p.add_argument("--output",default="observations.json")
    args=p.parse_args()
    if not args.consent_confirmed: raise ValueError("Confirm authorised consent using --consent-confirmed")
    from gaze.video import saved_frames, resize
    from gaze.vision import Estimator
    root=Path(__file__).resolve().parents[1]
    session=args.session_id or "S-"+uuid.uuid4().hex
    store=Store(root/"sessions"/"observations.sqlite3")
    estimator=None
    try:
        if session in [row[0] for row in store.sessions()]: raise ValueError("Session ID already exists")
        estimator=Estimator(root/"models"/"face_landmarker.task",args.checkpoint)
        timeline=Timeline()
        for start,end,frame in saved_frames(args.video):
            observation,_=estimator.process(resize(frame),start)
            timeline.add(start,end,observation)
        store.save(session,dict(source=str(Path(args.video).resolve()),review_video=str(Path(args.video).resolve()),
            consent_confirmed=True,created_utc=datetime.now(timezone.utc).isoformat(),checkpoint=args.checkpoint),timeline.finish())
        store.export(session,args.output)
        print(f"Saved session {session}; exported {args.output}")
    finally:
        if estimator: estimator.close()
        store.close()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit("Analysis cancelled; no completed session saved")
    except Exception as exc: sys.exit(f"Analysis failed: {exc}")
