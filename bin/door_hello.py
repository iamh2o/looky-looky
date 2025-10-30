#!/usr/bin/env python3
"""
DoorHello — camera+mic monitor with on-device person detect (YOLOv8n),
keyword ASR (Vosk), face recognition, and robust TTS backends.

Adds voice commands (always listening while scanning):
  rosy, pause      -> pause entry scanning (keep listening for commands)
  rosy, resume     -> resume entry scanning
  rosy, exit       -> cleanly exit
  rosy, hello      -> reply with fixed phrase (also true for a lone 'hello')
  rosy, who have you seen -> summarize seen names since this run

'hello' used ALONE is a command; 'hello' inside a longer phrase still
counts for the entry+hello pairing logic.
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
import re
import threading
import subprocess
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from rich import print
from ultralytics import YOLO

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3

# ---------- TTS Globals / Backends ----------
tts_engine: pyttsx3.Engine | None = None
tts_queue: deque[str] = deque()
_tts_backend: str = "pyttsx3"          # "pyttsx3" | "nsss" | "say"
_tts_voice_requested: str | None = None
_mac_synth = None                      # lazy NSSpeechSynthesizer wrapper instance
tts_active = threading.Event()        # True while TTS is speaking
ASR_TAIL_MS = 600                     # ignore mic this long *after* TTS ends

def _mac_list_voices() -> list[tuple[str, str]]:
    """Return [(voice_name, voice_id), ...] for macOS."""
    try:
        import AppKit
        voices = []
        for vid in AppKit.NSSpeechSynthesizer.availableVoices():
            attrs = AppKit.NSSpeechSynthesizer.attributesForVoice_(vid) or {}
            name = attrs.get("VoiceName", vid)
            voices.append((str(name), str(vid)))
        return voices
    except Exception:
        return []

class _MacNSSS:
    """macOS NSSpeechSynthesizer with robust voice selection + runloop pump."""
    def __init__(self, voice_req: str | None = None):
        import AppKit, Foundation
        self.AppKit = AppKit
        self.Foundation = Foundation
        self.synth = AppKit.NSSpeechSynthesizer.alloc().init()
        if not self.synth:
            raise RuntimeError("Failed to init NSSpeechSynthesizer")
        self._select_voice(voice_req)

    def _select_voice(self, voice_req: str | None):
        AppKit = self.AppKit
        if not voice_req:
            self._log_selected("default")
            return
        want = voice_req.strip().lower()
        chosen_id = None
        chosen_name = None
        for vid in AppKit.NSSpeechSynthesizer.availableVoices():
            attrs = AppKit.NSSpeechSynthesizer.attributesForVoice_(vid) or {}
            nm = str(attrs.get("VoiceName", "")) or str(vid)
            if (nm.lower() == want or want in nm.lower()
                or str(vid).lower() == want or want in str(vid).lower()):
                chosen_id = vid
                chosen_name = nm
                break
        if chosen_id:
            ok = self.synth.setVoice_(chosen_id)
            self._log_selected(chosen_name if ok else "fallback-after-setVoice")
        else:
            self._log_selected("not-found-fallback")

    def _log_selected(self, note: str):
        AppKit = self.AppKit
        vid = self.synth.voice()
        attrs = AppKit.NSSpeechSynthesizer.attributesForVoice_(vid) or {}
        nm = attrs.get("VoiceName", vid)
        print(f"[cyan]NSSS voice:[/cyan] {nm}  [cyan]id:[/cyan] {vid}  [cyan]note:[/cyan] {note}")

    def speak(self, text: str, blocking: bool = True):
        if not text:
            return
        self.synth.startSpeakingString_(text)
        if not blocking:
            return
        rl = self.Foundation.NSRunLoop.currentRunLoop()
        while self.synth.isSpeaking():
            rl.runUntilDate_(self.Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05))

    def stop(self):
        try:
            self.synth.stopSpeaking()
        except Exception:
            pass

def _speak_mac_say(text: str, blocking: bool = True, voice: str | None = None):
    if not text:
        return
    cmd = ["/usr/bin/say"]
    if voice:
        cmd += ["-v", voice]
    cmd += [text]
    if blocking:
        subprocess.run(cmd, check=False)
    else:
        subprocess.Popen(cmd)

def init_tts_engine() -> pyttsx3.Engine:
    """Initialize pyttsx3 and honor --voice if possible."""
    global tts_engine
    if tts_engine is None:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            if _tts_voice_requested:
                def _pick(vname):
                    for v in engine.getProperty("voices") or []:
                        if v.name.lower() == vname.lower() or vname.lower() in v.name.lower():
                            return v.id
                    return None
                vid = _pick(_tts_voice_requested)
                if vid:
                    engine.setProperty("voice", vid)
        except Exception:
            pass
        tts_engine = engine
    return tts_engine

def speak_text(message: str, *, blocking: bool = True) -> None:
    """Speak via selected backend. Blocking by default for reliability."""
    if not message:
        return
    tts_active.set()
    try:
        if _tts_backend == "say":
            _speak_mac_say(message, blocking=blocking, voice=_tts_voice_requested)
            return
        if _tts_backend == "nsss":
            global _mac_synth
            if _mac_synth is None:
                _mac_synth = _MacNSSS(_tts_voice_requested)
            _mac_synth.speak(message, blocking=blocking)
            return
        # pyttsx3
        if blocking:
            engine = init_tts_engine()
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception:
                if platform.system() == "Darwin":
                    _speak_mac_say(message, blocking=True, voice=_tts_voice_requested)
        else:
            tts_queue.append(message)
    finally:
        # End TX: keep mic gated briefly to avoid echo tail
        until = time.monotonic() + (ASR_TAIL_MS / 1000.0)
        def _clear_after_tail():
            # spin until tail expires; then clear
            while time.monotonic() < until:
                time.sleep(0.01)
            tts_active.clear()
        # Clear asynchronously so we return quickly
        threading.Thread(target=_clear_after_tail, daemon=True).start()

def drain_tts_queue() -> None:
    if not tts_queue:
        return
    tts_active.set()
    if _tts_backend == "say":
        while tts_queue:
            _speak_mac_say(tts_queue.popleft(), blocking=True, voice=_tts_voice_requested)
    elif _tts_backend == "nsss":
        global _mac_synth
        if _mac_synth is None:
            _mac_synth = _MacNSSS(_tts_voice_requested)
        while tts_queue:
            _mac_synth.speak(tts_queue.popleft(), blocking=True)
    else:
        engine = init_tts_engine()
        try:
            while tts_queue:
                engine.say(tts_queue.popleft())
            engine.runAndWait()
        except Exception:
            if platform.system() == "Darwin":
                while tts_queue:
                    _speak_mac_say(tts_queue.popleft(), blocking=True, voice=_tts_voice_requested)
    # tail clear
    until = time.monotonic() + (ASR_TAIL_MS / 1000.0)
    def _clear_after_tail():
        while time.monotonic() < until:
            time.sleep(0.01)
        tts_active.clear()
    threading.Thread(target=_clear_after_tail, daemon=True).start()



def shutdown_tts() -> None:
    try:
        drain_tts_queue()
    except Exception:
        pass
    try:
        if _tts_backend == "nsss" and _mac_synth:
            _mac_synth.stop()
    except Exception:
        pass
    try:
        if _tts_backend == "pyttsx3" and tts_engine is not None:
            tts_engine.stop()
    except Exception:
        pass

# ---------- Face recognition dep ----------
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
HELLO_WORDS = {"hello", "hi", "hey"}          # for pairer when 'hello' not used as a command
FACE_MATCH_THRESHOLD = 0.45

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

# In-memory registry and session stats
registry_lock = threading.Lock()
known_names: list[str] = []
known_encodings: list[np.ndarray] = []
enrollment_requests: "queue.Queue[np.ndarray | None]" = queue.Queue()
session_seen: list[tuple[float, str]] = []     # (ts, name) since program start
PROGRAM_STARTED_AT = time.time()

# ---------- Device discovery ----------
def _mac_list_video_device_names() -> dict[int, str]:
    """Return {opencv_index: device_name} via AVFoundation (best-effort)."""
    names = {}
    try:
        if platform.system() != "Darwin":
            return names
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
        if not devices:
            return names
        for i, dev in enumerate(list(devices)):
            label = dev.localizedName() or "Camera"
            names[i] = str(label)
    except Exception:
        pass
    return names

def describe_camera(index: int) -> dict[str, str | int | None]:
    name: str | None = None
    ip: str | None = None
    mac_names = _mac_list_video_device_names()
    if index in mac_names:
        name = mac_names[index]
    sysfs_name = Path(f"/sys/class/video4linux/video{index}/name")
    if sysfs_name.exists():
        try:
            sysfs_value = sysfs_name.read_text(encoding="utf-8").strip()
            if sysfs_value:
                name = sysfs_value
        except OSError:
            pass
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

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DoorHello monitor")
    parser.add_argument("--recheck-interval-ms", type=int, default=0,
                        help="Milliseconds between entry detection passes (default: every frame)")
    parser.add_argument("--tts", choices=["auto", "say", "nsss", "pyttsx3"], default="auto",
                        help="TTS backend (mac: prefer 'nsss' or 'say').")
    parser.add_argument("--voice", type=str, default="Amélie",
                        help="Voice name (e.g., 'Samantha', 'Alex', 'Amélie').")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available TTS voices and exit.")
    return parser.parse_args()

# ---------- ASR ----------
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

# --- Command parsing helpers ---
# --- Command parsing helpers (robust) ---

_ROSY_ALIASES = {"rosy", "rosie", "rosi", "rozzy", "rosybot", "rosiebot"}
_CMD_HELLO_WORDS  = {"hello", "hi", "hey"}
_CMD_PAUSE_WORDS  = {"pause", "hold", "stop"}
_CMD_RESUME_WORDS = {"resume", "continue", "unpause"}
_CMD_EXIT_WORDS   = {"exit", "quit", "bye", "goodbye"}
# "who have you seen" variants are messy; match keywords
def _normalize(phrase: str) -> str:
    # lowercase, remove punctuation to spaces, collapse whitespace
    p = phrase.lower()
    p = re.sub(r"[^\w\s]", " ", p)     # drop commas etc.
    p = re.sub(r"\s+", " ", p).strip()
    return p

def _parse_command(phrase: str) -> str | None:
    """
    Returns one of: cmd_pause, cmd_resume, cmd_exit, cmd_hello, cmd_seen, or None.
    Command priority: explicit 'rosy ...' hotword; else allow bare 'hello'.
    """
    p = _normalize(phrase)
    if not p:
        return None
    tokens = p.split()
    if not tokens:
        return None

    # Bare 'hello' stays a command (your requirement)
    if len(tokens) == 1 and tokens[0] in _CMD_HELLO_WORDS:
        return "cmd_hello"

    # Hotword-prefixed commands: 'rosy ...' (allow aliases)
    if tokens[0] in _ROSY_ALIASES:
        rest = tokens[1:]

        # hello
        if any(w in _CMD_HELLO_WORDS for w in rest):
            return "cmd_hello"

        # pause/resume/exit
        if any(w in _CMD_PAUSE_WORDS for w in rest):
            return "cmd_pause"
        if any(w in _CMD_RESUME_WORDS for w in rest):
            return "cmd_resume"
        if any(w in _CMD_EXIT_WORDS for w in rest):
            return "cmd_exit"

        # who have you seen — look for who + seen (+ optional have/you)
        s = " ".join(rest)
        if re.search(r"\bwho\b", s) and re.search(r"\bseen\b", s):
            return "cmd_seen"

    return None

DEBUG_ASR = True  # flip to True to print everything Vosk hears

def asr_listener(event_q: queue.Queue, stop_ev: threading.Event, mic_device: int | None = None):
    rec = KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE)
    rec.SetWords(False)

    def audio_cb(indata, frames, timeinfo, status):
        if tts_active.is_set():
            return
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
        if not phrase:
            return

        if DEBUG_ASR:
            print(f"[dim]ASR heard:[/dim] {phrase}")

        # Commands have priority and should NOT feed the hello-pairer.
        cmd = _parse_command(phrase)
        if cmd:
            event_q.put((cmd, time.time()))
            return

        # Pairer hello (only if 'hello' appears within longer utterances)
        norm = _normalize(phrase)
        tokens = set(norm.split())
        if tokens & _CMD_HELLO_WORDS:
            event_q.put(("pair_hello", time.time()))


    with sd.RawInputStream(
        samplerate=ASR_SAMPLE_RATE, blocksize=8000, dtype="int16", channels=1,
        callback=audio_cb, device=mic_device,
    ):
        while not stop_ev.is_set():
            sd.sleep(100)

# ---------- Registry / greetings ----------
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
    payload = [{"name": n, "encoding": e.tolist()} for n, e in zip(known_names, known_encodings)]
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def log_presence(name: str):
    VISITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{timestamp} - {name}\n")
    # also track this session
    session_seen.append((time.time(), name))

def share_koan(name: str):
    koan = random.choice(KOANS)
    message = f"A koan for {name}: {koan}"
    print(f"[italic blue]{message}[/italic blue]")
    speak_text(message, blocking=True)

def add_known_person(name: str, encoding: np.ndarray):
    encoding = np.asarray(encoding, dtype="float32")
    with registry_lock:
        known_names.append(name)
        known_encodings.append(encoding)
        persist_registry_locked()
    print(f"[bold green]Hello {name}! You have been recorded.[/bold green]")
    speak_text(f"Ahoy {name}! I will remember you.", blocking=True)
    log_presence(name)
    share_koan(name)

def greet_known_person(name: str):
    print(f"[bold cyan]Hello {name}! You have been recorded.[/bold cyan]")
    speak_text(f"Ahoy {name}!", blocking=True)
    log_presence(name)
    share_koan(name)

def analyze_faces(frame: np.ndarray):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame, model="hog")
        if not locations:
            return []
        encodings = face_recognition.face_encodings(rgb_frame, locations, num_jitters=1)
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
        speak_text("I do not recognize you. Please tell me your name so I can remember you.", blocking=True)
        print("[yellow]Unknown visitor detected. Please enter the name they wish to be known by (blank to skip):[/yellow]")
        try:
            name = input("Visitor name> ").strip()
        except EOFError:
            name = ""
        if name:
            add_known_person(name, encoding)
        else:
            speak_text("I didn't catch a name. I'll ask again next time.", blocking=True)
            print("[red]No name provided. Visitor was not recorded.[/red]")
        enrollment_requests.task_done()

def handle_entry(frame: np.ndarray):
    speak_text("Ahoy!", blocking=True)
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
            speak_text("I don't know you yet. Please share your name with me.", blocking=True)
            enrollment_requests.put(encoding)

# ---------- Main ----------
def main():
    global CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX
    global _tts_backend, _tts_voice_requested, _mac_synth

    args = parse_args()
    _tts_voice_requested = args.voice
    # Backend choice
    if args.tts == "auto":
        _tts_backend = "nsss" if platform.system() == "Darwin" else "pyttsx3"
    else:
        _tts_backend = args.tts

    # Voice listing
    if args.list_voices:
        if _tts_backend in ("nsss", "say") and platform.system() == "Darwin":
            voices = _mac_list_voices()
            print("[bold cyan]macOS voices[/bold cyan]")
            for nm, vid in voices:
                print(f"  {nm}  —  {vid}")
        elif _tts_backend == "pyttsx3":
            eng = init_tts_engine()
            print("[bold cyan]pyttsx3 voices[/bold cyan]")
            for v in eng.getProperty("voices") or []:
                print(f"  {v.name}  —  {v.id}")
        else:
            print("No voice list available on this backend/OS.")
        return

    print(f"[cyan]TTS backend:[/cyan] {_tts_backend}  [cyan]voice:[/cyan] {(_tts_voice_requested or 'default')}")

    # IO device selection
    CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX = configure_io_devices()
    if _tts_backend == "pyttsx3":
        init_tts_engine()

    # Audible self-test
    speak_text("TTS online.", blocking=True)

    # Prompt
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

    # Recheck throttle
    recheck_interval_ms = max(0, args.recheck_interval_ms)
    recheck_interval_sec = recheck_interval_ms / 1000.0

    # Threads
    asr_events = queue.Queue()
    stop_ev = threading.Event()
    t_asr = threading.Thread(target=asr_listener, args=(asr_events, stop_ev, MIC_DEVICE_INDEX), daemon=True)
    t_asr.start()

    ensure_registry_loaded()
    t_enroll = threading.Thread(target=enrollment_worker, args=(stop_ev,), daemon=True)
    t_enroll.start()

    # YOLO
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    print("[cyan]Camera opened. Press 'q' to quit.[/cyan]")

    presence_streak = 0
    occupied = False
    next_recheck_time = 0.0

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
                    try: entries.remove(te)
                    except ValueError: pass
                    try: hellos.remove(th)
                    except ValueError: pass
                    return

    def speak_seen_summary():
        """Summarize names seen since this run; speak it."""
        if not session_seen:
            speak_text("I haven't seen anyone yet.", blocking=True)
            return
        names = [n for _, n in session_seen]
        counts = Counter(names)
        parts = [f"{name} {cnt} time{'s' if cnt!=1 else ''}" for name, cnt in counts.items()]
        # Keep the readout sane
        if len(parts) > 8:
            parts = parts[:8] + [f"and {len(counts)-8} more"]
        speak_text("Since I started, I've seen " + ", ".join(parts) + ".", blocking=True)

    try:
        while True:
            # ASR events
            try:
                while True:
                    kind, ts = asr_events.get_nowait()
                    if kind == "pair_hello":
                        hellos.append(ts)
                        print(f"[yellow]Pairer hello @ {time.strftime('%H:%M:%S')}[/yellow]")
                    elif kind == "cmd_pause":
                        pause_active = True
                        paused_until = None
                        print("[blue]Voice command: pause[/blue]")
                        speak_text("Pausing new entry detection.", blocking=True)
                    elif kind == "cmd_resume":
                        pause_active = False
                        paused_until = None
                        print("[blue]Voice command: resume[/blue]")
                        speak_text("Resuming new entry detection.", blocking=True)
                    elif kind == "cmd_exit":
                        print("[blue]Voice command: exit[/blue]")
                        speak_text("Exiting now. Goodbye.", blocking=True)
                        raise KeyboardInterrupt
                    elif kind == "cmd_hello":
                        print("[blue]Voice command: hello[/blue]")
                        speak_text("hello, what the fuck do you want?", blocking=True)
                        # keep scanning as normal
                    elif kind == "cmd_seen":
                        print("[blue]Voice command: who have you seen[/blue]")
                        speak_seen_summary()
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                break

            ret, frame = cap.read()
            if not ret:
                print("[red]Frame grab failed[/red]")
                break

            now_mono = time.monotonic()

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
                for b in det.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person: {len(det.boxes)}"
                cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Entry logic
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

            pair_and_report()
            drain_tts_queue()

            # HUD
            if pause_active:
                overlay = "Detection paused"
                cv2.putText(frame, overlay, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("DoorHello (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        stop_ev.set()
        enrollment_requests.put(None)
        try: t_enroll.join(timeout=2.0)
        except Exception: pass
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: sd.stop()
        except Exception: pass
        shutdown_tts()

if __name__ == "__main__":
    main()
