#!/usr/bin/env python3
"""
DoorHello — camera+mic monitor with on-device person detect (YOLOv8n),
keyword ASR ("hello" via Vosk), face recognition, and speech (pyttsx3).

Fixes:
  • Face rec: use cv2.cvtColor for RGB (contiguous) to avoid dlib TypeError.
  • Guard face-recognition errors so they can't crash the loop.
  • TTS on main thread; clean engine shutdown to avoid AUHAL -50 noise.
  • Entry trigger fires once when presence streak crosses threshold.

Deps: ultralytics, opencv-python, numpy, rich, sounddevice, vosk, pyttsx3,
      face_recognition (with dlib).

Vosk model: unzip one of these into current dir:
    vosk-model-small-en-us-0.15
    vosk-model-en-us-0.22

macOS: pyttsx3 uses the **system default** output device; your sounddevice
speaker selection doesn't affect TTS. Optional fallback: set USE_OSX_SAY=1.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import queue
import random
import threading
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rich import print
from ultralytics import YOLO

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3

try:
    import face_recognition
except ImportError as exc:
    raise ImportError(
        "The 'face_recognition' package is required for identifying known visitors. "
        "Install it with 'pip install face_recognition'."
    ) from exc

# ---------- Config ----------
CAMERA_INDEX = 0
MIC_DEVICE_INDEX: int | None = None
SPEAKER_DEVICE_INDEX: int | None = None
CONF_THRES = 0.35
TIME_WINDOW = 7.0
MIN_PERSIST_NEW = 3
ASR_SAMPLE_RATE = 16000
HELLO_WORDS = {"hello", "hi", "hey"}
FACE_MATCH_THRESHOLD = 0.45
THANK_YOU_PHRASES = ("thank you rosy", "thank you rosy")
THANK_YOU_PAUSE_SECONDS = 10.0
RESUME_PHRASES = ("resume rosy", "resume rosy")


REGISTRY_PATH = Path(__file__).resolve().parent.parent / "logs" / "known_people.json"
VISITOR_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "visitors.log"

KOANS = [
    "What is the sound of one hand clapping?",
    "If you meet the Buddha on the road, kill him.",
    "The instant you speak about a thing, you miss the mark.",
    "When you can do nothing, what can you do?",
    "Where can you go to escape your own footprints?",
    "Not knowing is most intimate.",
    "Who is it that now hears this sound?",
    "What was your original face before your parents were born?",
    "A single instant is eternity; eternity is the now.",
    "When the many are reduced to one, to what is the one reduced?",
]

# In‑memory registry
registry_lock = threading.Lock()
known_names: list[str] = []
known_encodings: list[np.ndarray] = []
enrollment_requests: "queue.Queue[np.ndarray | None]" = queue.Queue()

# TTS (main-thread drained)
tts_engine: pyttsx3.Engine | None = None
tts_queue: deque[str] = deque()
# ----------------------------


# ---------------- TTS ----------------
def init_tts_engine() -> pyttsx3.Engine:
    global tts_engine
    if tts_engine is None:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
        except Exception:
            pass
        tts_engine = engine
    return tts_engine

def _mac_list_video_device_names() -> dict[int, str]:
    """
    Return {opencv_index: device_name} on macOS via AVFoundation.
    Best effort mapping: OpenCV uses avfoundation device ordering.
    """
    names = {}
    try:
        if platform.system() != "Darwin":
            return names
        # Lazy import to avoid hard dependency if not installed
        from AVFoundation import AVCaptureDevice
        from AVFoundation import AVMediaTypeVideo
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        if not devices:
            return names
        # OpenCV avfoundation generally enumerates as 0..N in same order.
        for i, dev in enumerate(list(devices)):
            label = dev.localizedName() or "Camera"
            names[i] = str(label)
    except Exception:
        pass
    return names

def _speak_mac_say(text: str, blocking: bool = True) -> None:
    if not text:
        return
    cmd = ["/usr/bin/say", text]
    try:
        if blocking:
            subprocess.run(cmd, check=False)
        else:
            subprocess.Popen(cmd)
    except Exception:
        pass


def speak_text(message: str, *, blocking: bool = False) -> None:
    if not message:
        return
    if sys.platform == "darwin" and os.getenv("USE_OSX_SAY") == "1":
        _speak_mac_say(message, blocking=blocking)
        return
    if blocking:
        engine = init_tts_engine()
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception:
            if sys.platform == "darwin":
                _speak_mac_say(message, blocking=True)
    else:
        tts_queue.append(message)


def drain_tts_queue() -> None:
    if not tts_queue:
        return
    if sys.platform == "darwin" and os.getenv("USE_OSX_SAY") == "1":
        while tts_queue:
            _speak_mac_say(tts_queue.popleft(), blocking=True)
        return
    engine = init_tts_engine()
    try:
        while tts_queue:
            engine.say(tts_queue.popleft())
        engine.runAndWait()
    except Exception:
        if sys.platform == "darwin":
            while tts_queue:
                _speak_mac_say(tts_queue.popleft(), blocking=True)


def shutdown_tts() -> None:
    """Best-effort cleanup to avoid AUHAL errors at interpreter shutdown."""
    try:
        drain_tts_queue()
    except Exception:
        pass
    try:
        if tts_engine is not None:
            tts_engine.stop()
    except Exception:
        pass
# -------------- end TTS --------------



def describe_camera(index: int) -> dict[str, str | int | None]:
    """Best-effort camera metadata lookup for display purposes."""
    name: str | None = None
    ip: str | None = None

    # macOS path: AVFoundation
    mac_names = _mac_list_video_device_names()
    if index in mac_names:
        name = mac_names[index]

    # Linux path: v4l sysfs
    sysfs_name = Path(f"/sys/class/video4linux/video{index}/name")
    if sysfs_name.exists():
        try:
            sysfs_value = sysfs_name.read_text(encoding="utf-8").strip()
            if sysfs_value:
                name = sysfs_value
        except OSError:
            pass

    # Optional IP via env
    for env_name in (f"CAMERA_{index}_IP", f"CAMERA{index}_IP", f"CAM{index}_IP", f"CAMERA_{index}_ADDR"):
        v = os.getenv(env_name)
        if v:
            ip = v.strip()
            break

    return {"index": index, "name": name, "ip": ip}


def detect_cameras(max_index: int = 10) -> list[dict[str, str | int | None]]:
    found: list[dict[str, str | int | None]] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                found.append(describe_camera(idx))
        else:
            if cap is not None:
                cap.release()
    return found


def describe_audio_devices() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to query audio devices: {exc}") from exc

    hostapis = sd.query_hostapis()
    microphones: list[dict[str, str]] = []
    speakers: list[dict[str, str]] = []

    for idx, dev in enumerate(devices):
        hostapi_name = ""
        if hostapis and 0 <= dev.get("hostapi", -1) < len(hostapis):
            hostapi_name = hostapis[dev["hostapi"]]["name"]
        label = f"{dev['name']} ({hostapi_name})" if hostapi_name else dev["name"]
        entry = {"index": str(idx), "label": label}
        if dev.get("max_input_channels", 0) > 0:
            microphones.append(entry)
        if dev.get("max_output_channels", 0) > 0:
            speakers.append(entry)
    return microphones, speakers


def prompt_choice(options: list[dict[str, str]], title: str) -> int:
    print(f"[bold cyan]{title}[/bold cyan]")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt['label']} (index {opt['index']})")

    default_idx = 1
    while True:
        try:
            raw = input(f"Select {title.lower()} [default {default_idx}]: ").strip()
        except EOFError:
            raw = ""
        if not raw:
            choice = default_idx
        else:
            if not raw.isdigit():
                print("[red]Please enter a number.[/red]")
                continue
            choice = int(raw)
        if 1 <= choice <= len(options):
            selection = int(options[choice - 1]["index"])
            print(f"[green]Selected {options[choice - 1]['label']}[/green]")
            return selection
        print("[red]Invalid selection. Try again.[/red]")


def test_camera_device(index: int) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened.")
    ret, _ = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Camera index {index} did not return frames.")
    print("[green]Camera test passed.[/green]")


def test_microphone_device(index: int) -> None:
    sd.check_input_settings(device=index, samplerate=ASR_SAMPLE_RATE, channels=1)
    with sd.InputStream(device=index, channels=1, samplerate=ASR_SAMPLE_RATE) as stream:
        stream.read(1)
    print("[green]Microphone test passed.[/green]")


def test_speaker_device(index: int) -> None:
    sd.check_output_settings(device=index, samplerate=ASR_SAMPLE_RATE, channels=1)
    duration = 0.35
    t = np.linspace(0, duration, int(ASR_SAMPLE_RATE * duration), False)
    tone = 0.2 * np.sin(2 * np.pi * 880 * t)
    sd.play(tone, samplerate=ASR_SAMPLE_RATE, device=index)
    sd.wait()
    print("[green]Speaker test passed (played confirmation tone).[/green]")


def configure_io_devices() -> tuple[int, int, int]:
    cameras = detect_cameras()
    if not cameras:
        raise RuntimeError("No usable cameras detected. Ensure at least one camera is connected.")
    camera_opts = []
    for cam in cameras:
        idx = int(cam["index"])
        details: list[str] = []
        name = cam.get("name")
        if name:
            details.append(str(name))
        ip = cam.get("ip")
        if ip:
            details.append(f"IP: {ip}")
        label = f"Camera {idx}"
        if details:
            label = f"{label} — {', '.join(details)}"
        camera_opts.append({"index": str(idx), "label": label})
    camera_index = prompt_choice(camera_opts, "Available Cameras")
    test_camera_device(camera_index)

    microphones, speakers = describe_audio_devices()
    if not microphones:
        raise RuntimeError("No microphones detected. A working microphone is required.")
    mic_index = prompt_choice(microphones, "Available Microphones")
    test_microphone_device(mic_index)

    if not speakers:
        raise RuntimeError("No speakers detected. A speaker/output device is required.")
    speaker_index = prompt_choice(speakers, "Available Speakers")
    test_speaker_device(speaker_index)

    sd.default.device = (mic_index, speaker_index)
    return camera_index, mic_index, speaker_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DoorHello monitor")
    parser.add_argument(
        "--recheck-interval-ms",
        type=int,
        default=0,
        help="Milliseconds between entry detection passes (default: every frame)",
    )
    return parser.parse_args()


def load_asr_model() -> Model:
    candidates = ["vosk-model-small-en-us-0.15", "vosk-model-en-us-0.22"]
    for c in candidates:
        if os.path.isdir(c):
            print(f"[green]Using Vosk model: {c}[/green]")
            return Model(c)
    raise RuntimeError(
        "Vosk model not found. Download and unzip into the project directory:\n"
        "  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
        "  unzip vosk-model-small-en-us-0.15.zip\n"
    )


def asr_listener(event_q: queue.Queue, stop_ev: threading.Event, mic_device: int | None = None):
    model = load_asr_model()
    rec = KaldiRecognizer(model, ASR_SAMPLE_RATE)
    rec.SetWords(False)

    last_pause_emit = 0.0

    def audio_cb(indata, frames, timeinfo, status):
        nonlocal last_pause_emit
        if status:
            pass
        if stop_ev.is_set():
            raise sd.CallbackStop()
        chunk = bytes(indata)
        if rec.AcceptWaveform(chunk):
            j = json.loads(rec.Result())
            phrase = (j.get("text") or "").strip()
        else:
            j = json.loads(rec.PartialResult())
            phrase = (j.get("partial") or "").strip()
        if phrase:
            normalized = phrase.lower()
            now_ts = time.time()
            if any(trigger in normalized for trigger in THANK_YOU_PHRASES):
                if now_ts - last_pause_emit > 1.0:
                    event_q.put(("pause", now_ts))
                    last_pause_emit = now_ts
            if any(trigger in normalized for trigger in RESUME_PHRASES):
                event_q.put(("resume", now_ts))
                
            tokens = {tok.strip(".,!?").lower() for tok in phrase.split()}
            if tokens & HELLO_WORDS:
                event_q.put(("hello", now_ts))

    with sd.RawInputStream(
        samplerate=ASR_SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_cb,
        device=mic_device,
    ):
        while not stop_ev.is_set():
            sd.sleep(100)


def ensure_registry_loaded():
    global known_names, known_encodings
    with registry_lock:
        if known_names:
            return
        if REGISTRY_PATH.exists():
            try:
                data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print("[red]Failed to read known people registry; starting fresh.[/red]")
                data = []
        else:
            data = []
        loaded_names: list[str] = []
        loaded_encodings: list[np.ndarray] = []
        for entry in data:
            name = entry.get("name")
            encoding = entry.get("encoding")
            if not name or not encoding:
                continue
            loaded_names.append(name)
            loaded_encodings.append(np.array(encoding, dtype="float32"))
        known_names = loaded_names
        known_encodings = loaded_encodings
        if known_names:
            print(f"[green]Loaded {len(known_names)} known visitor(s).[/green]")


def persist_registry_locked():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"name": name, "encoding": encoding.tolist()}
        for name, encoding in zip(known_names, known_encodings)
    ]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_presence(name: str):
    VISITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} - {name}\n")


def share_koan(name: str):
    koan = random.choice(KOANS)
    message = f"A koan for {name}: {koan}"
    print(f"[italic blue]{message}[/italic blue]")
    speak_text(message)


def add_known_person(name: str, encoding: np.ndarray):
    encoding = np.asarray(encoding, dtype="float32")
    with registry_lock:
        known_names.append(name)
        known_encodings.append(encoding)
        persist_registry_locked()
    print(f"[bold green]Hello {name}! You have been recorded.[/bold green]")
    speak_text(f"Ahoy {name}! I will remember you.")
    log_presence(name)
    share_koan(name)


def greet_known_person(name: str):
    print(f"[bold cyan]Hello {name}! You have been recorded.[/bold cyan]")
    speak_text(f"Ahoy {name}!")
    log_presence(name)
    share_koan(name)


def analyze_faces(frame: np.ndarray):
    """
    Return list of dicts with {location, encoding, name}.
    Uses contiguous RGB to satisfy dlib/pybind.
    """
    try:
        # Ensure contiguous RGB (avoid negative-stride view from [:,:,::-1])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect face locations (HOG is CPU-only, cnn requires dlib CNN model)
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        if not locations:
            return []
        # Compute encodings; explicitly pass num_jitters for stability
        encodings = face_recognition.face_encodings(rgb_frame, locations, num_jitters=1)
    except TypeError as e:
        # Most frequent failure is non-contiguous arrays; this path should be dead now
        print(f"[red]face_encodings TypeError: {e}[/red]")
        return []
    except Exception as e:
        print(f"[red]Face analysis error: {e}[/red]")
        return []

    analyses = []
    with registry_lock:
        stored_encodings = list(known_encodings)
        stored_names = list(known_names)
    for location, encoding in zip(locations, encodings):
        match_name = None
        if stored_encodings:
            distances = face_recognition.face_distance(stored_encodings, encoding)
            best_idx = int(np.argmin(distances))
            if distances[best_idx] <= FACE_MATCH_THRESHOLD:
                match_name = stored_names[best_idx]
        analyses.append({"location": location, "encoding": encoding, "name": match_name})
    return analyses


def enrollment_worker(stop_ev: threading.Event):
    while not stop_ev.is_set():
        try:
            encoding = enrollment_requests.get(timeout=0.5)
        except queue.Empty:
            continue
        if encoding is None:
            enrollment_requests.task_done()
            break
        speak_text("I do not recognize you. Please tell me your name so I can remember you.")
        print("[yellow]Unknown visitor detected. Please enter the name they wish to be known by (blank to skip):[/yellow]")
        try:
            name = input("Visitor name> ").strip()
        except EOFError:
            name = ""
        if name:
            add_known_person(name, encoding)
        else:
            speak_text("I didn't catch a name. I'll ask again next time.")
            print("[red]No name provided. Visitor was not recorded.[/red]")
        enrollment_requests.task_done()


def handle_entry(frame: np.ndarray):
    speak_text("Ahoy!")
    faces = analyze_faces(frame)
    if not faces:
        print("[red]Entry detected but no clear face found.[/red]")
        return
    for face in faces:
        name = face["name"]
        encoding = face["encoding"]
        if name:
            greet_known_person(name)
        else:
            print("[bold yellow]Hello there! Please let us know who you are.[/bold yellow]")
            speak_text("I don't know you yet. Please share your name with me.")
            enrollment_requests.put(encoding)


def main():
    global CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX

    args = parse_args()
    recheck_interval_ms = max(0, args.recheck_interval_ms)
    recheck_interval_sec = recheck_interval_ms / 1000.0

    CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX = configure_io_devices()
    init_tts_engine()

    prompt_message = "Greetings, would you like to begin monitoring?"
    speak_text(prompt_message, blocking=True)
    while True:
        response = input(f"{prompt_message} [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            break
        if response in {"n", "no", ""}:
            print("[cyan]Monitoring cancelled by user.[/cyan]")
            speak_text("Exiting monitoring mode. Goodbye.", blocking=True)
            return
        print("[yellow]Please answer 'yes' or 'no'.[/yellow]")

    # Event channels
    asr_events = queue.Queue()
    stop_ev = threading.Event()
    t_asr = threading.Thread(
        target=asr_listener,
        args=(asr_events, stop_ev, MIC_DEVICE_INDEX),
        daemon=True,
    )
    t_asr.start()

    ensure_registry_loaded()
    t_enroll = threading.Thread(target=enrollment_worker, args=(stop_ev,), daemon=True)
    t_enroll.start()

    # YOLOv8n
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    print("[cyan]Camera opened. Press 'q' to quit.[/cyan]")

    # Presence state
    presence_streak = 0
    occupied = False
    next_recheck_time = 0.0

    # Event deques
    entries = deque(maxlen=32)
    hellos = deque(maxlen=32)

    paused_until: float | None = None
    pause_active = False

    def pair_and_report():
        now = time.time()
        while entries and now - entries[0] > TIME_WINDOW:
            entries.popleft()
        while hellos and now - hellos[0] > TIME_WINDOW:
            hellos.popleft()
        for te in list(entries):
            for th in list(hellos):
                if abs(th - te) <= TIME_WINDOW:
                    print(f"[bold green]✅ DETECTED:[/bold green] person came in & said hello (Δt={abs(th-te):.1f}s)")
                    try:
                        entries.remove(te)
                    except ValueError:
                        pass
                    try:
                        hellos.remove(th)
                    except ValueError:
                        pass
                    return

    try:
        while True:
            # Drain ASR events
            try:
                while True:
                    kind, ts = asr_events.get_nowait()
                    if kind == "hello":
                        hellos.append(ts)
                        print(f"[yellow]Heard hello @ {time.strftime('%H:%M:%S')}[/yellow]")
                    elif kind == "pause":
                        now_mono = time.monotonic()
                        resume_at = now_mono + THANK_YOU_PAUSE_SECONDS
                        if paused_until is None or now_mono >= paused_until:
                            paused_until = resume_at
                            pause_active = True
                            print(
                                f"[blue]Pausing new entry detection for {THANK_YOU_PAUSE_SECONDS:.0f} seconds.[/blue]"
                            )
                            speak_text(
                                f"Pausing new entry detection for {int(THANK_YOU_PAUSE_SECONDS)} seconds."
                            )
                        else:
                            paused_until = resume_at
                            print("[blue]Extending detection pause.")
                            speak_text("Keeping detection paused a little longer.")
                    elif kind == "resume":
                        pause_active = False
                        paused_until = None
                        speak_text("Resuming new entry detection.")
            except queue.Empty:
                pass

            ret, frame = cap.read()
            if not ret:
                print("[red]Frame grab failed[/red]")
                break

            now_mono = time.monotonic()
            if pause_active and paused_until is not None and now_mono >= paused_until:
                pause_active = False
                paused_until = None
                print("[green]Resuming new entry detection.[/green]")
                speak_text("Resuming new entry detection.")

            should_update = not pause_active and (
                recheck_interval_sec == 0.0 or now_mono >= next_recheck_time
            )
            if should_update and recheck_interval_sec > 0.0:
                next_recheck_time = now_mono + recheck_interval_sec
            elif not should_update and recheck_interval_sec > 0.0 and next_recheck_time == 0.0:
                next_recheck_time = now_mono + recheck_interval_sec

            person_present = False
            if should_update:
                results = model.predict(source=frame, verbose=False, classes=[0], conf=CONF_THRES)
                det = results[0]
                person_present = (len(det.boxes) > 0)

                # Draw boxes (debug)
                for b in det.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person: {len(det.boxes)}"
                cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Entry logic: fire once on threshold crossing
            if pause_active:
                presence_streak = 0
                occupied = False
            elif should_update:
                if person_present:
                    presence_streak = min(presence_streak + 1, 1_000_000)
                    if not occupied and presence_streak >= MIN_PERSIST_NEW:
                        occupied = True
                        ts = time.time()
                        entries.append(ts)
                        print(f"[magenta]Entry detected @ {time.strftime('%H:%M:%S')}[/magenta]")
                        try:
                            handle_entry(frame.copy())
                        except Exception as e:
                            print(f"[red]handle_entry error: {e}[/red]")
                else:
                    presence_streak = 0
                    occupied = False

            # Pairing + TTS
            pair_and_report()
            drain_tts_queue()

            # UI
            if pause_active and paused_until is not None:
                remaining = max(0.0, paused_until - now_mono)
                overlay = f"Detection paused {remaining:0.1f}s"
                cv2.putText(
                    frame,
                    overlay,
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            cv2.imshow("DoorHello (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        stop_ev.set()
        enrollment_requests.put(None)
        try:
            t_enroll.join(timeout=2.0)
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            sd.stop()  # be nice to PortAudio
        except Exception:
            pass
        shutdown_tts()


if __name__ == "__main__":
    main()
