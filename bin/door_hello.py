#!/usr/bin/env python3
"""
Detect a new person entering camera view AND the word "hello" spoken on mic
within TIME_WINDOW seconds. Offline: YOLOv8n for person, Vosk for ASR.

macOS: grant Camera & Microphone permissions on first run.
"""

import time
import queue
import threading
from collections import deque

import cv2
import numpy as np
from rich import print
from ultralytics import YOLO

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------- Config ----------
CAMERA_INDEX = 0          # built-in cam usually 0
CONF_THRES = 0.35         # person conf threshold
FRAME_SKIP = 2            # run detector every N frames (reduce CPU)
TIME_WINDOW = 7.0         # seconds to pair entry + "hello"
MIN_PERSIST_NEW = 3       # frames of person presence before counting as "entry"
ASR_SAMPLE_RATE = 16000   # Vosk model default
HELLO_WORDS = {"hello", "hi", "hey"}  # accept any of these
# ----------------------------

def load_asr_model():
    """
    Use Vosk small English model. You must download a model once.
    Simplest path: 'vosk-model-small-en-us-0.15' folder in project root.
    """
    import os
    # Try small model folder in CWD; add your path if stored elsewhere.
    candidates = [
        "vosk-model-small-en-us-0.15",
        "vosk-model-en-us-0.22"
    ]
    for c in candidates:
        if os.path.isdir(c):
            print(f"[green]Using Vosk model: {c}[/green]")
            return Model(c)
    raise RuntimeError(
        "Vosk model not found. Download one small English model, unzip into project:\n"
        "  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
        "  unzip vosk-model-small-en-us-0.15.zip\n"
    )

def asr_listener(event_q: queue.Queue, stop_ev: threading.Event):
    """
    Stream mic audio -> Vosk -> push 'hello' events with timestamps.
    """
    model = load_asr_model()
    rec = KaldiRecognizer(model, ASR_SAMPLE_RATE)
    rec.SetWords(False)

    def audio_cb(indata, frames, timeinfo, status):
        if status:
            # Drop status noise; Vosk is resilient
            pass
        if stop_ev.is_set():
            raise sd.CallbackStop()
        if rec.AcceptWaveform(indata.tobytes()):
            txt = rec.Result()
        else:
            txt = rec.PartialResult()
        # Very simple parse: look for keywords lowercase
        low = txt.lower()
        if any(w in low for w in HELLO_WORDS):
            event_q.put(("hello", time.time()))

    with sd.RawInputStream(
        samplerate=ASR_SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_cb
    ):
        while not stop_ev.is_set():
            sd.sleep(100)

def main():
    # Prepare event channels
    asr_events = queue.Queue()
    stop_ev = threading.Event()
    t_asr = threading.Thread(target=asr_listener, args=(asr_events, stop_ev), daemon=True)
    t_asr.start()

    # Load YOLOv8n (auto-download on first use)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    print("[cyan]Camera opened. Press 'q' to quit.[/cyan]")
    prev_person_present = False
    person_presence_count = 0
    frame_idx = 0

    # Deques for event times
    entries = deque(maxlen=32)
    hellos  = deque(maxlen=32)

    def pair_and_report():
        # Pair latest entry with any hello within TIME_WINDOW seconds
        now = time.time()
        # purge old events
        while entries and now - entries[0] > TIME_WINDOW:
            entries.popleft()
        while hellos and now - hellos[0] > TIME_WINDOW:
            hellos.popleft()
        # if we have both, and |t_hello - t_entry| <= TIME_WINDOW → success
        for te in list(entries):
            for th in list(hellos):
                if abs(th - te) <= TIME_WINDOW:
                    # Report once per matched pair, then clear to avoid repeats
                    print(f"[bold green]✅ DETECTED:[/bold green] person came in & said hello "
                          f"(Δt={abs(th-te):.1f}s)")
                    # Optional: add your hook here (notify, save clip, trigger event)
                    # Clear matched to avoid duplicate reporting
                    try: entries.remove(te)
                    except ValueError: pass
                    try: hellos.remove(th)
                    except ValueError: pass
                    return

    try:
        while True:
            # Drain ASR events fast
            try:
                while True:
                    kind, ts = asr_events.get_nowait()
                    if kind == "hello":
                        hellos.append(ts)
                        print(f"[yellow]Heard hello @ {time.strftime('%H:%M:%S')}[/yellow]")
            except queue.Empty:
                pass

            ret, frame = cap.read()
            if not ret:
                print("[red]Frame grab failed[/red]")
                break

            frame_idx += 1
            # Run detector every FRAME_SKIP frames
            person_present = prev_person_present
            if frame_idx % FRAME_SKIP == 0:
                results = model.predict(source=frame, verbose=False, classes=[0], conf=CONF_THRES)
                # classes=[0] => 'person' in COCO
                det = results[0]
                person_present = (len(det.boxes) > 0)

                # Draw boxes (debug)
                for b in det.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person: {len(det.boxes)}"
                cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Entry logic: transition from no person → person, persisted a few frames
            if person_present:
                person_presence_count = min(person_presence_count + 1, 1000)
            else:
                person_presence_count = 0

            if not prev_person_present and person_present and person_presence_count >= MIN_PERSIST_NEW:
                ts = time.time()
                entries.append(ts)
                print(f"[magenta]Entry detected @ {time.strftime('%H:%M:%S')}[/magenta]")

            prev_person_present = person_present

            # Pairing
            pair_and_report()

            # Show window
            cv2.imshow("DoorHello (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        stop_ev.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
