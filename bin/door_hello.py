#!/usr/bin/env python3
"""
DoorHello — camera+mic monitor with on-device person detect (YOLOv8n),
keyword ASR ("hello" via Vosk), face recognition, and speech (pyttsx3).

Fixes vs your original:
  • TTS now runs on the MAIN THREAD (macOS-safe). No background TTS worker.
  • Entry trigger fixed: fires once when presence streak crosses threshold.
  • Vosk JSON is parsed properly; matches whole-word tokens.

Notes
  • Dependencies: ultralytics, opencv-python, numpy, rich, sounddevice, vosk,
    pyttsx3, face_recognition (and its native deps).
  • Vosk model: unzip one of:
        vosk-model-small-en-us-0.15
        vosk-model-en-us-0.22
    into the working directory.
  • pyttsx3 uses the OS **default** output device (System Settings → Sound).
    The 'speaker test' tone uses sounddevice’s chosen device; those may differ.
  • Optional macOS fallback: set USE_OSX_SAY=1 to use /usr/bin/say.

macOS: grant Camera & Microphone permissions on first run.
"""

from __future__ import annotations

import json
import os
import sys
import time
import queue
import random
import threading
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

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
# Device defaults (will be configured at runtime)
CAMERA_INDEX = 0          # built-in cam usually 0
MIC_DEVICE_INDEX: int | None = None
SPEAKER_DEVICE_INDEX: int | None = None
CONF_THRES = 0.35         # person conf threshold
FRAME_SKIP = 2            # run detector every N frames (reduce CPU)
TIME_WINDOW = 7.0         # seconds to pair entry + "hello"
MIN_PERSIST_NEW = 3       # detection cycles required before counting as "entry"
ASR_SAMPLE_RATE = 16000   # Vosk model default
HELLO_WORDS = {"hello", "hi", "hey"}  # accept any of these
FACE_MATCH_THRESHOLD = 0.45

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "logs" / "known_people.json"
VISITOR_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "visitors.log"
CAMERA_HINTS_PATH = Path(__file__).resolve().parent.parent / "logs" / "camera_hints.json"

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


def _safe_read_text(path: Path) -> str:
    """Return stripped text from *path* or an empty string when unavailable."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_sysfs_key_values(path: Path) -> dict[str, str]:
    """Parse key=value lines from a sysfs file into a dictionary."""
    text = _safe_read_text(path)
    data: dict[str, str] = {}
    if not text:
        return data
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            data[key] = value
    return data


def _normalize_detail(value: Any) -> str | None:
    """Normalize various value types into user-presentable detail text."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    return text or None


def load_camera_hints() -> dict[str, Any]:
    """Load optional user-provided camera metadata from env or JSON file."""
    hints_env = os.getenv("LOOKY_CAMERA_HINTS")
    if hints_env:
        try:
            parsed = json.loads(hints_env)
            if isinstance(parsed, dict):
                return parsed
            print("[yellow]Ignoring LOOKY_CAMERA_HINTS: expected JSON object.[/yellow]")
        except json.JSONDecodeError as exc:
            print(f"[yellow]Ignoring LOOKY_CAMERA_HINTS due to JSON error: {exc}[/yellow]")

    if CAMERA_HINTS_PATH.exists():
        try:
            with CAMERA_HINTS_PATH.open("r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            if isinstance(parsed, dict):
                return parsed
            print(
                f"[yellow]Ignoring {CAMERA_HINTS_PATH}: expected top-level JSON object.[/yellow]"
            )
        except Exception as exc:  # pragma: no cover - depends on host FS
            print(f"[yellow]Unable to read camera hints: {exc}[/yellow]")
    return {}


def _append_hint_detail(
    details: list[str], seen: set[str], key: str | None, value: Any
) -> None:
    normalized = _normalize_detail(value)
    if not normalized:
        return
    if key and isinstance(key, str):
        label = key.strip().lower()
        if label in {"details", "detail", "notes", "note", "extra"}:
            text = normalized
        else:
            text = f"{key}: {normalized}"
    else:
        text = normalized
    if text not in seen:
        details.append(text)
        seen.add(text)


def build_camera_label(
    index: int,
    hints: dict[str, Any],
    backend: str,
    width: int,
    height: int,
    fps: float,
) -> str:
    """Construct a descriptive label for a camera index."""

    dev_node = Path(f"/dev/video{index}")
    sys_base = Path(f"/sys/class/video4linux/video{index}")
    default_name = _safe_read_text(sys_base / "name") or None

    details: list[str] = []
    seen: set[str] = set()

    hint_entry = hints.get(str(index)) or hints.get(str(dev_node))
    hint_name: str | None = None

    if isinstance(hint_entry, dict):
        raw_name = hint_entry.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            hint_name = raw_name.strip()
        for key, value in hint_entry.items():
            if key == "name":
                continue
            if isinstance(value, list):
                for item in value:
                    _append_hint_detail(details, seen, key, item)
            else:
                _append_hint_detail(details, seen, key, value)
    elif hint_entry is not None:
        _append_hint_detail(details, seen, None, hint_entry)

    if dev_node.exists():
        _append_hint_detail(details, seen, "path", str(dev_node))
        try:
            real_path = os.path.realpath(dev_node)
        except OSError:
            real_path = None
        if real_path and real_path != str(dev_node):
            _append_hint_detail(details, seen, "resolved", real_path)

    uevent = _read_sysfs_key_values(sys_base / "device" / "uevent")
    if uevent:
        product = uevent.get("PRODUCT")
        if product:
            parts = product.split("/")
            if len(parts) == 3:
                vendor, product_id, revision = parts
                _append_hint_detail(
                    details,
                    seen,
                    None,
                    f"product {vendor}:{product_id} (rev {revision})",
                )
            else:
                _append_hint_detail(details, seen, None, f"product {product}")
        driver = uevent.get("DRIVER")
        if driver:
            _append_hint_detail(details, seen, None, f"driver {driver}")
        serial = uevent.get("SERIAL")
        if serial:
            _append_hint_detail(details, seen, "serial", serial)
        slot = uevent.get("PCI_SLOT_NAME")
        if slot:
            _append_hint_detail(details, seen, "slot", slot)

    if width > 0 and height > 0:
        _append_hint_detail(details, seen, None, f"{width}x{height}")
    if fps > 0:
        _append_hint_detail(details, seen, None, f"{fps:.1f} fps")
    if backend:
        _append_hint_detail(details, seen, "backend", backend)

    name = hint_name or default_name or f"Camera {index}"
    if hint_name and default_name and hint_name.lower() != default_name.lower():
        _append_hint_detail(details, seen, "hardware", default_name)

    if not details:
        return name
    return f"{name} — {', '.join(details)}"


# ---------------- TTS ----------------
def init_tts_engine() -> pyttsx3.Engine:
    """Initialize pyttsx3 engine lazily."""
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


def _speak_mac_say(text: str, blocking: bool = True) -> None:
    """Optional macOS fallback via /usr/bin/say when USE_OSX_SAY=1."""
    if not text:
        return
    cmd = ["/usr/bin/say", text]
    try:
        if blocking:
            subprocess.run(cmd, check=False)
        else:
            subprocess.Popen(cmd)
    except Exception:
        pass  # swallow; we'll still have printed logs


def speak_text(message: str, *, blocking: bool = False) -> None:
    """Queue speech or speak immediately on main thread if blocking=True."""
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
            # Last-chance fallback for macOS if pyttsx3 trips
            if sys.platform == "darwin":
                _speak_mac_say(message, blocking=True)
    else:
        tts_queue.append(message)


def drain_tts_queue() -> None:
    """Run any queued speech on the main thread (macOS-safe)."""
    if not tts_queue:
        return
    if sys.platform == "darwin" and os.getenv("USE_OSX_SAY") == "1":
        # Drain via /usr/bin/say
        while tts_queue:
            _speak_mac_say(tts_queue.popleft(), blocking=True)
        return
    engine = init_tts_engine()
    try:
        while tts_queue:
            engine.say(tts_queue.popleft())
        engine.runAndWait()
    except Exception:
        # If pyttsx3 fails mid-run, try macOS say for remaining messages
        if sys.platform == "darwin":
            while tts_queue:
                _speak_mac_say(tts_queue.popleft(), blocking=True)
# -------------- end TTS --------------


def detect_cameras(max_index: int = 10) -> list[dict[str, str]]:
    """Return descriptive camera options for usable video capture devices."""

    options: list[dict[str, str]] = []
    hints = load_camera_hints()

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is None:
            continue
        if not cap.isOpened():
            cap.release()
            continue

        backend_name = ""
        width = 0
        height = 0
        fps = 0.0
        usable = False

        try:
            try:
                ret, _ = cap.read()
            except Exception:
                ret = False
            if ret:
                usable = True
                get_backend = getattr(cap, "getBackendName", None)
                if callable(get_backend):
                    try:
                        backend_name = str(get_backend())
                    except Exception:
                        backend_name = ""
                try:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                except Exception:
                    width = height = 0
                    fps = 0.0
        finally:
            cap.release()

        if not usable:
            continue

        label = build_camera_label(idx, hints, backend_name, width, height, fps)
        options.append({"index": str(idx), "label": label})

    return options


def describe_audio_devices() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Gather input/output-capable audio devices with metadata for display."""
    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - depends on host setup
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
    """Ask the user to pick a device and return the chosen device index."""
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
    """Detect available devices, prompt user, run quick validation."""
    camera_opts = detect_cameras()
    if not camera_opts:
        raise RuntimeError("No usable cameras detected. Ensure at least one camera is connected.")
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


def load_asr_model() -> Model:
    """
    Use Vosk English model. You must download a model once and unzip it in CWD.
    """
    candidates = [
        "vosk-model-small-en-us-0.15",
        "vosk-model-en-us-0.22",
    ]
    for c in candidates:
        if os.path.isdir(c):
            print(f"[green]Using Vosk model: {c}[/green]")
            return Model(c)
    raise RuntimeError(
        "Vosk model not found. Download and unzip one into the project directory:\n"
        "  wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip\n"
        "  unzip vosk-model-small-en-us-0.15.zip\n"
    )


def asr_listener(event_q: queue.Queue, stop_ev: threading.Event, mic_device: int | None = None):
    """
    Stream mic audio -> Vosk -> push 'hello' events with timestamps.
    """
    model = load_asr_model()
    rec = KaldiRecognizer(model, ASR_SAMPLE_RATE)
    rec.SetWords(False)

    def audio_cb(indata, frames, timeinfo, status):
        if status:
            # Ignore occasional over/underflow messages
            pass
        if stop_ev.is_set():
            raise sd.CallbackStop()
        # RawInputStream gives bytes-like buffer
        chunk = bytes(indata)
        if rec.AcceptWaveform(chunk):
            j = json.loads(rec.Result())
            phrase = (j.get("text") or "").strip()
        else:
            j = json.loads(rec.PartialResult())
            phrase = (j.get("partial") or "").strip()
        if phrase:
            tokens = {tok.strip(".,!?").lower() for tok in phrase.split()}
            if tokens & HELLO_WORDS:
                event_q.put(("hello", time.time()))

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
    """Load known people from disk once at startup."""
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
    """Persist the known people registry to disk. Caller must hold registry_lock."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"name": name, "encoding": encoding.tolist()}
        for name, encoding in zip(known_names, known_encodings)
    ]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def log_presence(name: str):
    """Append a presence entry to the visitor log."""
    VISITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} - {name}\n")


def share_koan(name: str):
    koan = random.choice(KOANS)
    print(f"[italic blue]A koan for {name}: {koan}[/italic blue]")


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
    rgb_frame = frame[:, :, ::-1]
    locations = face_recognition.face_locations(rgb_frame)
    if not locations:
        return []
    encodings = face_recognition.face_encodings(rgb_frame, locations)
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
        analyses.append({
            "location": location,
            "encoding": encoding,
            "name": match_name,
        })
    return analyses


def enrollment_worker(stop_ev: threading.Event):
    """Background enrollment dialog (reads from stdin)."""
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

    # Prepare event channels
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

    # Load YOLOv8n (auto-download on first use)
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    print("[cyan]Camera opened. Press 'q' to quit.[/cyan]")

    # Presence state: fire once when we cross MIN_PERSIST_NEW
    presence_streak = 0
    occupied = False
    frame_idx = 0

    # Deques for event times
    entries = deque(maxlen=32)
    hellos = deque(maxlen=32)

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
                    print(f"[bold green]✅ DETECTED:[/bold green] person came in & said hello "
                          f"(Δt={abs(th-te):.1f}s)")
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
            # Drain ASR events quickly
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
            updated_this_frame = (frame_idx % FRAME_SKIP == 0)

            # Run detector every FRAME_SKIP frames
            person_present = False
            if updated_this_frame:
                results = model.predict(source=frame, verbose=False, classes=[0], conf=CONF_THRES)
                det = results[0]
                person_present = (len(det.boxes) > 0)

                # Draw boxes (debug)
                for b in det.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person: {len(det.boxes)}"
                cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Entry logic: fire once when we cross the persistence threshold
            if updated_this_frame:
                if person_present:
                    presence_streak = min(presence_streak + 1, 1_000_000)
                    if not occupied and presence_streak >= MIN_PERSIST_NEW:
                        occupied = True
                        ts = time.time()
                        entries.append(ts)
                        print(f"[magenta]Entry detected @ {time.strftime('%H:%M:%S')}[/magenta]")
                        handle_entry(frame.copy())
                else:
                    presence_streak = 0
                    occupied = False

            # Pairing
            pair_and_report()

            # Speak any queued TTS from the main thread
            drain_tts_queue()

            # Show window
            cv2.imshow("DoorHello (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        stop_ev.set()
        enrollment_requests.put(None)
        t_enroll.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        # best-effort drain any final messages
        drain_tts_queue()


if __name__ == "__main__":
    main()
