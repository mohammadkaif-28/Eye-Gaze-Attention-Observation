# Integration acceptance checklist

Not executed in the build environment: MediaPipe, PySide6, PyAV and the face model
could not be installed because dependency download DNS resolution failed.

1. Run setup on Python 3.12; doctor must import every dependency and find the face asset.
2. Start the app offline. Confirm consent is required for both video and camera.
3. Analyse an authorised front-facing test video with left/right/up/down/camera looks.
   Check the viewer-coordinate convention against manual labels.
4. Include blink, hand occlusion, dark/blurred frames, offscreen face, strong head turn,
   reflective glasses and a second face. Inspect false-valid outputs, not just crashes.
5. Verify summary duration equals analysable + unavailable. Direction percentages
   must sum to 100 when analysable time is positive, otherwise all be zero.
6. Compare timeline boundaries to video timestamps, including a variable-frame-rate
   video, long capture stalls and recording start/end. No shifts across unknown gaps.
7. Click early and late intervals; verify matching evidence and play/pause. Test missing
   source file, invalid codec, empty video and malformed checkpoint error messages.
8. Record a consented webcam session, stop, reopen and review it. Disconnect the camera
   during capture: the UI must remain responsive and preserve any completed intervals.
9. Correct/dismiss/annotate rows. Export JSON and verify separate therapist records;
   original machine estimates and machine summary must remain unchanged.
10. Reopen the app and saved session. Test a duplicate shared session ID and export to
    a read-only path. Confirm graceful errors and intact saved observations.
11. Process the real Gaze360 dataset, inspect all rejection counts, train and evaluate.
    Load the resulting best.pt into the app and compare it with the heuristic baseline.
12. Independently label representative therapy videos. Report direction confusion,
    valid/invalid detection errors, timestamp error, tracking coverage and latency.
    Gaze360 angular error alone does not validate behaviour in this population/setting.

Optional target-region mapping remains outside this implementation. Do not label
coarse directions as a person, toy or screen without a separate calibrated method.
