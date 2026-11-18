import json
import tempfile
import unittest
from pathlib import Path
from gaze.core import Observation, Timeline, summarize, timestamp
from gaze.storage import Store


class TimelineTests(unittest.TestCase):
    def test_missing_time_never_counts_as_gaze_or_shift(self):
        t=Timeline(dwell=0)
        t.add(0,1,Observation("left","valid",.8))
        t.add(2,3,Observation("right","valid",.8))
        result=summarize(t.finish())
        self.assertEqual(result["analysable_duration"],2)
        self.assertEqual(result["unavailable_duration"],1)
        self.assertEqual(result["directions"]["left"]["percent"],50)
        self.assertEqual(result["gaze_shift_count"],0)

    def test_blink_breaks_continuity(self):
        t=Timeline(dwell=0)
        t.add(0,1,Observation("left","valid",.8))
        t.add(1,1.2,Observation(reason="blink"))
        t.add(1.2,2,Observation("right","valid",.8))
        self.assertEqual(summarize(t.finish())["gaze_shift_count"],0)

    def test_confirmed_shift_timestamp(self):
        t=Timeline(.15)
        t.add(0,1,Observation("left","valid",.8))
        t.add(1,1.1,Observation("right","valid",.8))
        t.add(1.1,1.2,Observation("right","valid",.8))
        result=summarize(t.finish())
        self.assertEqual(result["gaze_shifts"][0]["timestamp"],"00:00:01.000")

    def test_flicker_not_fabricated(self):
        t=Timeline(.2)
        t.add(0,.1,Observation("left","valid",.8))
        self.assertEqual(summarize(t.finish())["analysable_duration"],0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Observation("left","low confidence",.3)
        with self.assertRaises(ValueError): Observation(confidence=float("nan"))
        t=Timeline()
        t.add(0,1,Observation())
        with self.assertRaises(ValueError): t.add(.5,2,Observation())
        with self.assertRaises(ValueError): timestamp(-1)
        self.assertEqual(timestamp(3599.9996),"01:00:00.000")

    def test_confidence_duration_weighted(self):
        t=Timeline(0)
        t.add(0,1,Observation("left","valid",.6))
        t.add(1,4,Observation("left","valid",1))
        self.assertAlmostEqual(t.finish()[0].confidence,.9)

    def test_empty_summary(self):
        self.assertEqual(summarize([])["analysable_duration"],0)


class StorageTests(unittest.TestCase):
    def test_review_separate_export_and_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/"sessions.db")
            t=Timeline(0); t.add(0,1,Observation("left","valid",.8))
            store.save("shared-id",{"source":"test"},t.finish())
            store.review("shared-id",0,"corrected","right","Manual review")
            self.assertEqual(store.load("shared-id")[1][0].direction,"left")
            out=Path(folder)/"out.json"; store.export("shared-id",out)
            result=json.loads(out.read_text())
            self.assertEqual(result["therapist_reviews"][0]["corrected_direction"],"right")
            self.assertEqual(result["machine_observations"][0]["direction_or_target"],"left")
            store.delete("shared-id")
            self.assertEqual(store.reviews("shared-id"),[])
            store.close()

if __name__ == "__main__": unittest.main()
