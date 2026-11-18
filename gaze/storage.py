import csv
import json
import os
from pathlib import Path
import sqlite3
from .core import Interval, summarize


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, metadata TEXT NOT NULL, intervals TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS reviews (
          session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          observation_id INTEGER NOT NULL, decision TEXT NOT NULL,
          corrected_direction TEXT NOT NULL, note TEXT NOT NULL,
          PRIMARY KEY(session_id, observation_id));
        ''')

    def save(self, session_id, metadata, intervals):
        from dataclasses import asdict
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                            (session_id, json.dumps(metadata), json.dumps([asdict(i) for i in intervals])))

    def sessions(self):
        return self.db.execute("SELECT id FROM sessions ORDER BY rowid DESC").fetchall()

    def load(self, session_id):
        row = self.db.execute("SELECT metadata, intervals FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError("Session no longer exists")
        return json.loads(row[0]), [Interval(**i) for i in json.loads(row[1])]

    def review(self, session_id, index, decision, corrected, note):
        _, intervals = self.load(session_id)
        if not 0 <= index < len(intervals):
            raise ValueError("Invalid observation")
        if decision not in ("accepted", "corrected", "dismissed", "annotated"):
            raise ValueError("Invalid review decision")
        from .core import DIRECTIONS
        if decision == "corrected" and corrected not in (*DIRECTIONS, "indeterminate"):
            raise ValueError("Select a corrected direction")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
                            (session_id, index, decision, corrected, note))

    def reviews(self, session_id):
        return [dict(observation_id=r[0], decision=r[1], corrected_direction=r[2], note=r[3])
                for r in self.db.execute("SELECT observation_id, decision, corrected_direction, note FROM reviews WHERE session_id=?", (session_id,))]

    def delete(self, session_id):
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def export(self, session_id, path):
        metadata, intervals = self.load(session_id)
        records = [i.record(session_id, n) for n, i in enumerate(intervals)]
        path = Path(path)
        temp = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="") as f:
                if path.suffix.lower() == ".csv":
                    fields = list(records[0]) if records else ["session_id", "timestamp_start", "timestamp_end", "direction_or_target", "duration", "confidence", "tracking_status"]
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)
                else:
                    json.dump(dict(session_id=session_id, metadata=metadata, machine_observations=records,
                                   summary=summarize(intervals), therapist_reviews=self.reviews(session_id)), f, indent=2)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def close(self):
        self.db.close()
