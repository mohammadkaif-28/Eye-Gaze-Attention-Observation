from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import time
import traceback
import uuid
import av
import cv2
from PySide6 import QtCore, QtGui, QtWidgets as W
from .core import DIRECTIONS, Observation, Timeline, summarize, timestamp
from .storage import Store
from .video import saved_frames, camera_frames, resize
from .vision import Estimator


class Worker(QtCore.QThread):
    preview = QtCore.Signal(object,str)
    result = QtCore.Signal(object,object,str)

    def __init__(self, root, source, checkpoint, recording):
        super().__init__()
        self.root,self.source,self.checkpoint,self.recording=root,source,checkpoint,recording

    def run(self):
        timeline=Timeline()
        estimator=container=frames=None
        error=""
        metadata=dict(source=str(self.source),created_utc=datetime.now(timezone.utc).isoformat(),
            estimator="trained gaze model" if self.checkpoint else "landmark heuristic",
            checkpoint=str(self.checkpoint or ""),consent_confirmed=True,
            direction_convention="unmirrored viewer image",optional_targets="not configured")
        try:
            estimator=Estimator(self.root/"models"/"face_landmarker.task",self.checkpoint)
            live=isinstance(self.source,int)
            frames=camera_frames(self.source,self.isInterruptionRequested) if live else saved_frames(self.source)
            if live:
                container=av.open(str(self.recording),mode="w")
                stream=container.add_stream("mpeg4",rate=30)
                stream.width,stream.height=640,480
                stream.pix_fmt="yuv420p"
                stream.time_base=Fraction(1,1000)
                stream.codec_context.time_base=Fraction(1,1000)
                metadata["review_video"]=str(self.recording)
            else:
                metadata["review_video"]=str(Path(self.source).resolve())
            last_preview=-1
            for start,end,frame in frames:
                if self.isInterruptionRequested(): break
                frame=resize(frame)
                obs,annotated=estimator.process(frame,start)
                if live:
                    encoded=av.VideoFrame.from_ndarray(cv2.resize(frame,(640,480)),format="bgr24")
                    encoded.pts=round(start*1000); encoded.time_base=Fraction(1,1000)
                    for packet in stream.encode(encoded): container.mux(packet)
                timeline.add(start,end,obs)
                if start-last_preview >= .1:
                    self.preview.emit(annotated.copy(),f"{timestamp(start)} | {obs.direction} | {obs.status} | {obs.reason}")
                    last_preview=start
            metadata["stopped_by_user"]=self.isInterruptionRequested()
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            metadata["error"]=error
            (self.root/"sessions").mkdir(exist_ok=True)
            with (self.root/"sessions"/"errors.log").open("a",encoding="utf-8") as log:
                log.write(traceback.format_exc()+"\n")
        finally:
            for obj in (frames,estimator):
                if obj:
                    try: obj.close()
                    except Exception: pass
            if container:
                try:
                    for packet in stream.encode(): container.mux(packet)
                    container.close()
                except Exception as exc:
                    error += f" Recording finalization failed: {exc}"
            self.result.emit(timeline.finish(),metadata,error)


class Window(W.QMainWindow):
    def __init__(self,root):
        super().__init__()
        self.root=root
        self.store=Store(root/"sessions"/"observations.sqlite3")
        self.worker=None; self.session=None; self.intervals=[]; self.review_frames=None
        self.setWindowTitle("Eye Gaze · Observation & Review")
        self.resize(1200,850)
        main=W.QWidget(); self.setCentralWidget(main); layout=W.QVBoxLayout(main)
        title=W.QLabel("Eye Gaze & Visual Attention — observation workspace")
        title.setStyleSheet("font-size:22px;font-weight:600;color:#176c74;padding:8px")
        layout.addWidget(title)
        layout.addWidget(W.QLabel("Estimated direction is not an attention score, eye-contact measure or diagnosis. Verify observations against video."))
        self.consent=W.QCheckBox("Parent/guardian consent confirmed; I am authorised to process this session")
        layout.addWidget(self.consent)
        controls=W.QHBoxLayout(); layout.addLayout(controls)
        self.session_id=W.QLineEdit(); self.session_id.setPlaceholderText("Shared session ID (optional; otherwise generated)")
        controls.addWidget(self.session_id)
        self.camera_index=W.QSpinBox(); self.camera_index.setRange(0,10)
        controls.addWidget(W.QLabel("Camera")); controls.addWidget(self.camera_index)
        self.camera_button=W.QPushButton("Start webcam + local recording")
        self.video_button=W.QPushButton("Analyse video")
        self.stop_button=W.QPushButton("Stop"); self.stop_button.setEnabled(False)
        for b in (self.camera_button,self.video_button,self.stop_button): controls.addWidget(b)
        self.camera_button.clicked.connect(lambda:self.start(self.camera_index.value()))
        self.video_button.clicked.connect(self.choose_video)
        self.stop_button.clicked.connect(self.stop)
        modelrow=W.QHBoxLayout(); layout.addLayout(modelrow)
        self.checkpoint=W.QLineEdit(); self.checkpoint.setPlaceholderText("Optional trained best.pt; blank uses landmark baseline")
        modelrow.addWidget(self.checkpoint)
        pick=W.QPushButton("Select trained model"); modelrow.addWidget(pick)
        pick.clicked.connect(self.pick_model)
        self.model_button=pick
        split=W.QSplitter(); layout.addWidget(split,1)
        self.preview=W.QLabel("Load a recorded video or start a consented session")
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480,300)
        self.preview.setStyleSheet("background:#12252b;color:#e0eff0;border-radius:8px")
        split.addWidget(self.preview)
        self.summary=W.QTextBrowser(); self.summary.setMinimumWidth(300); split.addWidget(self.summary)
        self.status=W.QLabel("Ready · Offline analysis after model installation")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        row=W.QHBoxLayout(); layout.addLayout(row)
        self.history=W.QComboBox(); row.addWidget(self.history,1)
        for label,callback in (("Open session",self.open_session),("Play / pause review",self.toggle_play),
                               ("Export JSON",lambda:self.export("json")),("Export CSV",lambda:self.export("csv")),
                               ("Delete session",self.delete_session)):
            button=W.QPushButton(label); button.clicked.connect(callback); row.addWidget(button)
        self.table=W.QTableWidget(0,7)
        self.table.setHorizontalHeaderLabels(["Start","End","Machine direction","Tracking","Quality","Reason","Therapist review (separate)"])
        self.table.setSelectionBehavior(W.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(W.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.ResizeMode.ResizeToContents)
        self.table.cellClicked.connect(self.seek)
        layout.addWidget(self.table,1)
        notes=W.QHBoxLayout(); layout.addLayout(notes)
        self.decision=W.QComboBox(); self.decision.addItems(["annotated","accepted","corrected","dismissed"])
        self.corrected=W.QComboBox(); self.corrected.addItems(["indeterminate",*DIRECTIONS])
        self.note=W.QLineEdit(); self.note.setPlaceholderText("Therapist note: machine observations remain unchanged")
        save=W.QPushButton("Save review"); save.clicked.connect(self.save_review)
        for widget in (self.decision,self.corrected,self.note,save): notes.addWidget(widget)
        self.timer=QtCore.QTimer(self); self.timer.setSingleShot(True); self.timer.timeout.connect(self.next_frame)
        self.refresh_history()

    def notify(self,exc):
        W.QMessageBox.warning(self,"Observation workspace",str(exc))

    def refresh_history(self):
        self.history.clear(); self.history.addItems([r[0] for r in self.store.sessions()])

    def pick_model(self):
        path,_=W.QFileDialog.getOpenFileName(self,"Select project checkpoint",str(self.root/"runs"),"Checkpoint (*.pt)")
        if path: self.checkpoint.setText(path)

    def choose_video(self):
        if not self.consent.isChecked(): return self.notify("Confirm consent before opening a video.")
        path,_=W.QFileDialog.getOpenFileName(self,"Select video","","Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)")
        if path: self.start(path)

    def start(self,source):
        if not self.consent.isChecked(): return self.notify("Confirm consent before starting.")
        if self.worker and self.worker.isRunning(): return
        self.pause_review()
        session=self.session_id.text().strip() or "S-"+datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
        if session in [r[0] for r in self.store.sessions()]: return self.notify("Session ID already exists. Choose a unique shared ID.")
        self.session=session
        recording=self.root/"sessions"/(uuid.uuid4().hex+".mp4")
        self.worker=Worker(self.root,source,self.checkpoint.text().strip() or None,recording)
        self.worker.preview.connect(self.show_frame)
        self.worker.result.connect(self.completed)
        self.worker.finished.connect(lambda:self.busy(False))
        self.busy(True); self.worker.start()

    def busy(self,active):
        for widget in (self.camera_button,self.video_button,self.model_button,self.checkpoint,self.session_id): widget.setEnabled(not active)
        self.stop_button.setEnabled(active)

    def stop(self):
        if self.worker: self.worker.requestInterruption()

    @QtCore.Slot(object,str)
    def show_frame(self,frame,status=""):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.preview.size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.SmoothTransformation))
        if status: self.status.setText(status)

    @QtCore.Slot(object,object,str)
    def completed(self,intervals,metadata,error):
        try:
            if intervals:
                self.store.save(self.session,metadata,intervals)
                self.refresh_history(); self.history.setCurrentText(self.session)
                self.display(self.session)
                self.status.setText("Partial session saved" if error else "Session saved · click an interval to review")
            if error: self.notify(error)
        except Exception as exc: self.notify(f"Could not save session: {exc}")

    def open_session(self):
        if self.worker and self.worker.isRunning(): return self.notify("Stop analysis before opening another session.")
        if self.history.currentText():
            try: self.display(self.history.currentText())
            except Exception as exc: self.notify(exc)

    def display(self,session):
        self.pause_review()
        self.metadata,self.intervals=self.store.load(session); self.session=session
        report=summarize(self.intervals)
        lines=["MACHINE SESSION SUMMARY",f"Session: {session}",f"Total: {report['total_duration']:.2f}s",
               f"Analysable: {report['analysable_duration']:.2f}s",f"Unavailable: {report['unavailable_duration']:.2f}s",
               f"Gaze shifts: {report['gaze_shift_count']}",f"Mean valid interval: {report['average_valid_interval']:.2f}s",
               f"Median valid interval: {report['median_valid_interval']:.2f}s","", "Percent of analysable time:"]
        lines += [f"{d}: {v['seconds']:.2f}s ({v['percent']:.1f}%)" for d,v in report["directions"].items()]
        lines += ["",report["confidence_note"],"", "Targets: not calibrated; person/object attribution unavailable."]
        self.summary.setPlainText("\n".join(lines))
        reviews={r["observation_id"]:r for r in self.store.reviews(session)}
        self.table.setRowCount(len(self.intervals))
        for n,interval in enumerate(self.intervals):
            record=interval.record(session,n)
            values=[record["timestamp_start"],record["timestamp_end"],interval.direction,interval.status,
                    f"{record['confidence_label']} ({interval.confidence:.2f})",interval.reason,
                    json.dumps(reviews[n],ensure_ascii=False) if n in reviews else "Not reviewed"]
            for col,value in enumerate(values): self.table.setItem(n,col,W.QTableWidgetItem(value))

    def pause_review(self):
        self.timer.stop()
        if self.review_frames:
            self.review_frames.close(); self.review_frames=None

    def seek(self,row,column=0):
        if self.worker and self.worker.isRunning(): return
        self.pause_review()
        if not 0 <= row < len(self.intervals): return
        try:
            target=self.intervals[row].start
            self.review_frames=saved_frames(self.metadata["review_video"])
            for start,end,frame in self.review_frames:
                if end>target:
                    self.show_frame(frame,f"Review {timestamp(start)} · source video")
                    break
            self.playback_position=target
        except Exception as exc:
            self.pause_review(); self.notify(f"Review video unavailable: {exc}. Keep the source video at its original path.")

    def toggle_play(self):
        if self.worker and self.worker.isRunning(): return
        if self.timer.isActive(): self.timer.stop(); return
        if not self.review_frames:
            self.seek(max(0,self.table.currentRow()))
        if self.review_frames: self.next_frame()

    def next_frame(self):
        try:
            start,end,frame=next(self.review_frames)
            self.show_frame(frame,f"Review {timestamp(start)}")
            self.timer.start(max(1,round((end-start)*1000)))
        except StopIteration: self.pause_review()
        except Exception as exc:
            self.pause_review(); self.notify(exc)

    def save_review(self):
        if not self.session or self.table.currentRow()<0: return self.notify("Select an interval first.")
        if self.worker and self.worker.isRunning(): return self.notify("Stop analysis before reviewing.")
        try:
            self.store.review(self.session,self.table.currentRow(),self.decision.currentText(),self.corrected.currentText(),self.note.text())
            self.display(self.session); self.note.clear()
        except Exception as exc: self.notify(exc)

    def export(self,extension):
        if not self.session: return self.notify("Open a saved session first.")
        if self.worker and self.worker.isRunning(): return self.notify("Stop analysis before exporting.")
        path,_=W.QFileDialog.getSaveFileName(self,"Export observations",f"observations.{extension}",f"{extension.upper()} (*.{extension})")
        if path:
            try:
                self.store.export(self.session,path)
                self.status.setText("Exported. JSON includes separate therapist reviews; CSV contains machine observations.")
            except Exception as exc: self.notify(exc)

    def delete_session(self):
        if self.worker and self.worker.isRunning(): return
        if not self.session: return
        if W.QMessageBox.question(self,"Delete session","Delete observations and reviews? Source/recorded video remains on disk; manage its retention separately.") != W.QMessageBox.StandardButton.Yes: return
        try:
            self.store.delete(self.session); self.session=None; self.intervals=[]
            self.pause_review(); self.table.setRowCount(0); self.summary.clear(); self.refresh_history()
        except Exception as exc: self.notify(exc)

    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.stop(); self.status.setText("Stopping safely. Close again after the session is saved."); event.ignore(); return
        self.pause_review(); self.store.close(); event.accept()


def run(root):
    app=W.QApplication.instance() or W.QApplication([])
    app.setStyle("Fusion")
    window=Window(root); window.show()
    app.exec()
