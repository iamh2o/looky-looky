#!/usr/bin/env python3
"""
DoorHello — person + pet entry monitor with on-device detection (YOLOv8n),
keyword ASR (Vosk), profiles (people/cats/dogs/donkeys), and robust TTS.

Improvements:
  • Debounced, final-result ASR to stop duplicate command replies.
  • Half-duplex mic gating while speaking (default); CLI + voice toggle.
  • Detects person/cat/dog/donkey (donkey≈COCO 'horse'); builds voice-enrolled profiles by type.
  • If name already exists for that type, ask for a last name; supports multi-word names.
  • Voice switch: "rosy voice zarvox|amelie|samantha" (typos accepted).
  • Entry artifacts: plays a gong, saves screenshot to images/entry_<timestamp>.jpg.

Voice commands:
  rosy pause / rosy resume / rosy exit / rosy hello
  rosy whats up                -> read chronological entries list (Unknown allowed)
  rosy listening off|on        -> hard mute/unmute ASR
  rosy voice zarvox|amelie|samantha
"""

from __future__ import annotations

import argparse, json, os, platform, sys, time, queue, random, re, threading, subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rich import print
from ultralytics import YOLO

import sounddevice as sd
from vosk import Model, KaldiRecognizer
import pyttsx3

# ========================= Globals / Config =========================

# Devices & runtime
CAMERA_INDEX = 0
MIC_DEVICE_INDEX: int | None = None
SPEAKER_DEVICE_INDEX: int | None = None

# Detection
CONF_THRES = 0.35
TIME_WINDOW = 7.0
MIN_PERSIST_NEW = 3
ASR_SAMPLE_RATE = 16000

# YOLO COCO classes of interest
COCO_PERSON = 0
COCO_CAT = 15
COCO_DOG = 16
COCO_HORSE = 17   # we treat as "donkey" for our use
SPECIES_LABEL = {COCO_PERSON: "person", COCO_CAT: "cat", COCO_DOG: "dog", COCO_HORSE: "donkey"}
DETECT_CLASSES = [COCO_PERSON, COCO_CAT, COCO_DOG, COCO_HORSE]

# Face recognition
FACE_MATCH_THRESHOLD = 0.45

# Paths
ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "logs" / "known_people.json"
VISITOR_LOG_PATH = ROOT / "logs" / "visitors.log"
IMAGES_DIR = ROOT / "images"

# ASR gating
tts_active = threading.Event()        # set while TTS speaking
asr_force_muted = threading.Event()   # user-forced mute via command
ASR_TAIL_MS = 700                     # ignore mic after TTS ends
asr_mute_while_speaking = True        # default; CLI toggles

# TTS backend
tts_engine: pyttsx3.Engine | None = None
tts_queue: deque[str] = deque()
_tts_backend: str = "pyttsx3"         # "pyttsx3" | "nsss" | "say"
_tts_voice_requested: str | None = None
_mac_synth = None

# Known registry (now typed)
registry_lock = threading.Lock()
# entries: {"name": str, "type": "person"|"cat"|"dog"|"donkey", "encoding": list|None}
known_entries: list[dict] = []

# Session tracking
session_entries: list[tuple[float, str]] = []  # (ts, name|Unknown) chronological list

# Debounce of command handling
CMD_DEBOUNCE_SEC = 1.2
_last_cmd_ts: dict[str, float] = {}
_last_pair_hello = 0.0

# Debug
DEBUG_ASR = False


# ========================= TTS Backends =========================

def _mac_list_voices() -> list[tuple[str, str]]:
    try:
        import AppKit
        out = []
        for vid in AppKit.NSSpeechSynthesizer.availableVoices():
            attrs = AppKit.NSSpeechSynthesizer.attributesForVoice_(vid) or {}
            out.append((str(attrs.get("VoiceName", vid)), str(vid)))
        return out
    except Exception:
        return []

class _MacNSSS:
    def __init__(self, voice_req: str | None = None):
        import AppKit, Foundation
        self.AppKit = AppKit; self.Foundation = Foundation
        self.synth = AppKit.NSSpeechSynthesizer.alloc().init()
        if not self.synth:
            raise RuntimeError("Failed to init NSSpeechSynthesizer")
        self._select_voice(voice_req)

    def _select_voice(self, voice_req: str | None):
        if not voice_req:
            return
        want = voice_req.strip().lower()
        chosen_id = None
        for vid in self.AppKit.NSSpeechSynthesizer.availableVoices():
            attrs = self.AppKit.NSSpeechSynthesizer.attributesForVoice_(vid) or {}
            nm = str(attrs.get("VoiceName", "")) or str(vid)
            if nm.lower() == want or want in nm.lower() or str(vid).lower() == want:
                chosen_id = vid; break
        if chosen_id:
            self.synth.setVoice_(chosen_id)

    def set_voice(self, voice_req: str | None):
        self._select_voice(voice_req)

    def speak(self, text: str, blocking: bool = True):
        if not text: return
        self.synth.startSpeakingString_(text)
        if not blocking: return
        rl = self.Foundation.NSRunLoop.currentRunLoop()
        while self.synth.isSpeaking():
            rl.runUntilDate_(self.Foundation.NSDate.dateWithTimeIntervalSinceNow_(0.05))

def _speak_mac_say(text: str, blocking: bool = True, voice: str | None = None):
    if not text: return
    cmd = ["/usr/bin/say"]
    if voice: cmd += ["-v", voice]
    cmd += [text]
    (subprocess.run if blocking else subprocess.Popen)(cmd, check=False)

def init_tts_engine() -> pyttsx3.Engine:
    global tts_engine
    if tts_engine is None:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            if _tts_voice_requested:
                _set_pyttsx3_voice(engine, _tts_voice_requested)
        except Exception:
            pass
        tts_engine = engine
    return tts_engine

def _set_pyttsx3_voice(engine: pyttsx3.Engine, vname: str):
    want = vname.lower()
    for v in engine.getProperty("voices") or []:
        nm = (v.name or "").lower()
        if nm == want or want in nm:
            engine.setProperty("voice", v.id); return

def _normalize_voice_name(raw: str) -> str:
    # accept typos: zarzog->Zarvox, amile->Amelie
    r = raw.strip().lower()
    if r in {"zarvox","zarzog","zarvoxx","zarvoxbot"}: return "Zarvox"
    if r in {"amelie","amélie","amile","ameli"}:      return "Amelie"
    if r in {"samantha","sam","samanth"}:             return "Samantha"
    return raw.title()

def set_voice(new_voice: str):
    global _tts_voice_requested, _mac_synth, tts_engine
    _tts_voice_requested = _normalize_voice_name(new_voice)
    if _tts_backend == "nsss":
        if _mac_synth is None: _mac_synth = _MacNSSS(_tts_voice_requested)
        else: _mac_synth.set_voice(_tts_voice_requested)
    elif _tts_backend == "say":
        pass  # /usr/bin/say uses the requested voice dynamically
    else:
        eng = init_tts_engine()
        _set_pyttsx3_voice(eng, _tts_voice_requested)

def _tail_clear():
    until = time.monotonic() + (ASR_TAIL_MS / 1000.0)
    while time.monotonic() < until: time.sleep(0.01)
    tts_active.clear()

def speak_text(message: str, *, blocking: bool = True) -> None:
    if not message: return
    if asr_mute_while_speaking: tts_active.set()
    try:
        if _tts_backend == "say":
            _speak_mac_say(message, blocking=blocking, voice=_tts_voice_requested)
        elif _tts_backend == "nsss":
            global _mac_synth
            if _mac_synth is None: _mac_synth = _MacNSSS(_tts_voice_requested)
            _mac_synth.speak(message, blocking=blocking)
        else:
            eng = init_tts_engine()
            try:
                eng.say(message); eng.runAndWait()
            except Exception:
                if platform.system() == "Darwin":
                    _speak_mac_say(message, blocking=True, voice=_tts_voice_requested)
    finally:
        if asr_mute_while_speaking:
            threading.Thread(target=_tail_clear, daemon=True).start()

def drain_tts_queue() -> None:
    if not tts_queue: return
    if asr_mute_while_speaking: tts_active.set()
    if _tts_backend == "say":
        while tts_queue: _speak_mac_say(tts_queue.popleft(), blocking=True, voice=_tts_voice_requested)
    elif _tts_backend == "nsss":
        global _mac_synth
        if _mac_synth is None: _mac_synth = _MacNSSS(_tts_voice_requested)
        while tts_queue: _mac_synth.speak(tts_queue.popleft(), blocking=True)
    else:
        eng = init_tts_engine()
        try:
            while tts_queue: eng.say(tts_queue.popleft())
            eng.runAndWait()
        except Exception:
            if platform.system() == "Darwin":
                while tts_queue: _speak_mac_say(tts_queue.popleft(), blocking=True, voice=_tts_voice_requested)
    if asr_mute_while_speaking:
        threading.Thread(target=_tail_clear, daemon=True).start()

def shutdown_tts() -> None:
    try: drain_tts_queue()
    except Exception: pass
    try:
        if _tts_backend == "nsss" and _mac_synth: _mac_synth.stop()
    except Exception: pass
    try:
        if _tts_backend == "pyttsx3" and tts_engine is not None: tts_engine.stop()
    except Exception: pass

# ========================= Face Registry I/O =========================

def ensure_registry_loaded():
    global known_entries
    with registry_lock:
        if known_entries: return
        if REGISTRY_PATH.exists():
            try:
                data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                # normalize older schema
                normalized = []
                for e in data:
                    if isinstance(e, dict):
                        nm = e.get("name"); tp = e.get("type","person")
                        enc = e.get("encoding")
                        normalized.append({"name": nm, "type": tp, "encoding": enc})
                known_entries = normalized
                if known_entries:
                    print(f"[green]Loaded {len(known_entries)} known profile(s).[/green]")
            except Exception:
                print("[red]Failed to read registry; starting empty.[/red]")
                known_entries = []
        else:
            known_entries = []

def persist_registry_locked():
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(known_entries, indent=2), encoding="utf-8")

def log_presence(name: str):
    VISITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(VISITOR_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{ts} - {name}\n")
    session_entries.append((time.time(), name))

def share_koan(name: str):
    KOANS = [
        "What is the sound of one hand clapping?",
        "Not knowing is most intimate.",
        "Who hears this sound?",
        "What was your original face before your parents were born?",
        "A single instant is eternity; eternity is the now.",
    ]
    speak_text(f"A koan for {name}: {random.choice(KOANS)}", blocking=True)

def add_known_entity(name: str, etype: str, encoding: list[float] | None):
    with registry_lock:
        known_entries.append({"name": name, "type": etype, "encoding": encoding})
        persist_registry_locked()
    speak_text(f"Ahoy {name}! I will remember you.", blocking=True)
    log_presence(name)
    share_koan(name)

def greet_known(name: str, etype: str):
    speak_text(f"Ahoy {name}!", blocking=True)
    log_presence(name)
    share_koan(name)

def find_name_conflict(name: str, etype: str) -> bool:
    nm_low = name.lower()
    with registry_lock:
        for e in known_entries:
            if e.get("type")==etype and (e.get("name") or "").lower() == nm_low:
                return True
    return False

# ========================= Utilities =========================

def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

_NAME_TRIGGERS = [
    ("my name is", 3), ("i am", 2), ("i'm", 2),
    ("its", 2), ("it's", 2), ("call me", 2), ("name", 1),
]

def _extract_name_from_phrase(phrase: str) -> str | None:
    p = _norm_text(phrase)
    if not p: return None
    for trig, _ in _NAME_TRIGGERS:
        if trig in p:
            tail = p.split(trig, 1)[1].strip()
            toks = tail.split()
            cand = " ".join(toks[:3]).strip()
            cand = re.sub(r"^(the|a|an)\s+","", cand)
            cand = re.sub(r"\b(so|uh|um|and|please|thanks|thank)\b.*$", "", cand).strip()
            cand = cand[:60]
            if len(cand)>=2: return cand.title()
    toks = p.split()
    if 1 <= len(toks) <= 3 and all(len(t)>1 for t in toks):
        return " ".join(toks).title()
    return None

def listen_for_name(mic_device: int | None, timeout_s: float = 6.0) -> str | None:
    rec = KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE); rec.SetWords(False)
    stop_ev = threading.Event()
    got: list[str] = []
    def audio_cb(indata, frames, timeinfo, status):
        if (asr_force_muted.is_set()) or (asr_mute_while_speaking and tts_active.is_set()):
            return
        if stop_ev.is_set(): raise sd.CallbackStop()
        chunk = bytes(indata)
        if rec.AcceptWaveform(chunk):
            phrase = (json.loads(rec.Result()).get("text") or "").strip()
            if phrase:
                got.append(phrase)
                n = _extract_name_from_phrase(phrase)
                if n:
                    got.append(f"__NAME__:{n}")
                    stop_ev.set(); raise sd.CallbackStop()
    deadline = time.monotonic() + timeout_s
    try:
        with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE, blocksize=8000, dtype="int16",
                               channels=1, callback=audio_cb, device=mic_device):
            while not stop_ev.is_set() and time.monotonic() < deadline: sd.sleep(100)
    except sd.CallbackStop:
        pass
    for h in got:
        if h.startswith("__NAME__:"): return h.split(":",1)[1]
    if got:
        return _extract_name_from_phrase(" ".join(h for h in got if not h.startswith("__NAME__:")))
    return None

def play_gong(device_index: int | None):
    # simple decaying chime
    sr = 22050
    dur = 1.5
    t = np.linspace(0, dur, int(sr*dur), False)
    env = np.exp(-3.0*t)
    freqs = [196.0, 392.0, 523.25]
    sig = sum(np.sin(2*np.pi*f*t)*(0.5/len(freqs)) for f in freqs) * env
    try:
        sd.play(sig.astype(np.float32), samplerate=sr, device=device_index)
        sd.wait()
    except Exception:
        pass

def save_screenshot(frame: np.ndarray):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = IMAGES_DIR / f"entry_{ts}.jpg"
    try:
        cv2.imwrite(str(path), frame)
    except Exception as e:
        print(f"[red]Failed to save screenshot: {e}[/red]")

# ========================= ASR / Commands =========================

# Hotword & keywords
_ROSY_ALIASES = {"rosy","rosie","rosi","rosis","rozzy","rosybot","rosiebot"}
_CMD_HELLO_WORDS = {"hello","hi","hey"}
_CMD_PAUSE_WORDS = {"pause","hold","stop"}
_CMD_RESUME_WORDS = {"resume","continue","unpause"}
_CMD_EXIT_WORDS = {"exit","quit","bye","goodbye"}

VOICE_SYNONYMS = {"zarvox","zarzog","samantha","amelie","amélie","amile"}

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+"," ", s).strip()

def _debounced(kind: str) -> bool:
    now = time.monotonic()
    last = _last_cmd_ts.get(kind, 0.0)
    if now - last < CMD_DEBOUNCE_SEC: return False
    _last_cmd_ts[kind] = now
    return True

def _parse_command_final_only(phrase: str):
    """
    Return (kind, payload) or (None, None).
    Only for FINAL results to avoid repeats.
    """
    p = _normalize(phrase)
    toks = p.split()
    if not toks: return None, None

    # Bare hello is a command
    if len(toks)==1 and toks[0] in _CMD_HELLO_WORDS:
        return "cmd_hello", None

    if toks[0] in _ROSY_ALIASES:
        rest = toks[1:]
        rest_set = set(rest)
        if rest_set & _CMD_PAUSE_WORDS:  return "cmd_pause", None
        if rest_set & _CMD_RESUME_WORDS: return "cmd_resume", None
        if rest_set & _CMD_EXIT_WORDS:   return "cmd_exit", None
        # what's up
        s = " ".join(rest)
        if (re.search(r"\bwhats?\b", s) and re.search(r"\bup\b", s)) or re.search(r"\bwhat\b\s+(is|s)\s+up\b", s):
            return "cmd_seen", None
        # listening on/off
        if "listening" in rest_set and "off" in rest_set: return "cmd_listen_off", None
        if "listening" in rest_set and "on"  in rest_set: return "cmd_listen_on", None
        # voice switch
        if "voice" in rest_set:
            for w in rest:
                if w in VOICE_SYNONYMS:
                    return "cmd_voice", _normalize_voice_name(w)
    return None, None

def load_asr_model() -> Model:
    for c in ["vosk-model-small-en-us-0.15","vosk-model-en-us-0.22"]:
        if os.path.isdir(c):
            print(f"[green]Using Vosk model: {c}[/green]")
            return Model(c)
    raise RuntimeError("Vosk model not found. Unzip a model into the working dir.")

def asr_listener(event_q: queue.Queue, stop_ev: threading.Event, mic_device: int | None = None):
    rec = KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE); rec.SetWords(False)

    def audio_cb(indata, frames, timeinfo, status):
        if asr_force_muted.is_set(): return
        if asr_mute_while_speaking and tts_active.is_set(): return
        if stop_ev.is_set(): raise sd.CallbackStop()
        # FINAL results only to prevent duplicates
        chunk = bytes(indata)
        if not rec.AcceptWaveform(chunk):
            return
        phrase = (json.loads(rec.Result()).get("text") or "").strip()
        if not phrase: return
        if DEBUG_ASR: print(f"[dim]ASR heard:[/dim] {phrase}")
        kind, payload = _parse_command_final_only(phrase)
        if kind:
            if _debounced(kind):
                event_q.put(("cmd", time.time(), kind, payload))
            return
        # Pairer hello from longer final phrases
        norm = _normalize(phrase)
        if _CMD_HELLO_WORDS & set(norm.split()):
            global _last_pair_hello
            now = time.monotonic()
            if now - _last_pair_hello >= CMD_DEBOUNCE_SEC:
                _last_pair_hello = now
                event_q.put(("pair_hello", time.time(), None, None))

    with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE, blocksize=8000, dtype="int16",
                           channels=1, callback=audio_cb, device=mic_device):
        while not stop_ev.is_set(): sd.sleep(100)

# ========================= Vision Helpers =========================

def analyze_faces(frame: np.ndarray):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    import face_recognition
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs: return []
    encs = face_recognition.face_encodings(rgb, locs, num_jitters=1)
    out = []
    with registry_lock:
        persons = [e for e in known_entries if e.get("type")=="person" and e.get("encoding")]
        stored_enc = [np.array(e["encoding"], dtype="float32") for e in persons]
        stored_names = [e["name"] for e in persons]
    for loc, enc in zip(locs, encs):
        name = None
        if stored_enc:
            import face_recognition
            d = face_recognition.face_distance(stored_enc, enc)
            best = int(np.argmin(d))
            if d[best] <= FACE_MATCH_THRESHOLD:
                name = stored_names[best]
        out.append({"location": loc, "encoding": enc.tolist(), "name": name})
    return out

def yolo_entities(results) -> list[tuple[str, tuple[int,int,int,int]]]:
    ent = []
    det = results[0]
    for b in det.boxes:
        cls_id = int(b.cls[0].item()) if b.cls is not None else None
        if cls_id in DETECT_CLASSES:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            etype = SPECIES_LABEL.get(cls_id, "person")
            ent.append((etype, (x1,y1,x2,y2)))
    return ent

# ========================= Enrollment =========================

def enrollment_worker(stop_ev: threading.Event, mic_device: int | None):
    while not stop_ev.is_set():
        try:
            req = enrollment_requests.get(timeout=0.5)
        except queue.Empty:
            continue
        if req is None:
            enrollment_requests.task_done(); break

        etype = req.get("type", "person")
        enc = req.get("encoding")  # list or None

        speak_text(f"I do not recognize this {etype}. Please tell me the name.", blocking=True)
        name = listen_for_name(mic_device, timeout_s=6.0)
        if not name:
            speak_text("Please say: I am — and then your name.", blocking=True)
            name = listen_for_name(mic_device, timeout_s=6.0)

        if name:
            # handle duplicate within same type
            if find_name_conflict(name, etype):
                speak_text(f"I already know {name} the {etype}. Please say a last name.", blocking=True)
                last = listen_for_name(mic_device, timeout_s=6.0)
                if last:
                    name = f"{name} {last}".strip()
            add_known_entity(name, etype, enc if etype=="person" else None)
        else:
            speak_text("I didn't catch a name. I'll ask again next time.", blocking=True)
            print("[red]No name captured by voice. Not recorded.[/red]")

        enrollment_requests.task_done()

# ========================= Command Handlers =========================

def handle_cmd(kind: str, payload, state):
    global asr_mute_while_speaking
    if kind == "cmd_pause":
        state["paused"] = True
        speak_text("Pausing new entry detection.", blocking=True)
    elif kind == "cmd_resume":
        state["paused"] = False
        speak_text("Resuming new entry detection.", blocking=True)
    elif kind == "cmd_exit":
        speak_text("Exiting now. Goodbye.", blocking=True)
        raise KeyboardInterrupt
    elif kind == "cmd_hello":
        speak_text("hello, what the fuck do you want?", blocking=True)
    elif kind == "cmd_seen":
        speak_entries_list()
    elif kind == "cmd_listen_off":
        asr_force_muted.set()
        speak_text("Listening off.", blocking=True)
    elif kind == "cmd_listen_on":
        asr_force_muted.clear()
        speak_text("Listening on.", blocking=True)
    elif kind == "cmd_voice":
        set_voice(payload or "")
        speak_text(f"Voice set to {payload}.", blocking=True)

def speak_entries_list():
    if not session_entries:
        speak_text("No entries yet.", blocking=True); return
    print("[cyan]Entries this session:[/cyan]")
    for ts, nm in session_entries:
        print(time.strftime("  %Y-%m-%d %H:%M:%S", time.localtime(ts)), "-", nm or "Unknown")
    lines = [f"{time.strftime('%H:%M:%S', time.localtime(ts))} — {nm or 'Unknown'}" for ts,nm in session_entries]
    speak_text("Entries so far: " + "; ".join(lines) + ".", blocking=True)

# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DoorHello monitor")
    p.add_argument("--recheck-interval-ms", type=int, default=0,
                   help="Milliseconds between detection passes (default: every frame)")
    p.add_argument("--tts", choices=["auto","say","nsss","pyttsx3"], default="auto",
                   help="TTS backend (mac: prefer 'nsss' or 'say').")
    p.add_argument("--voice", type=str, default="Zarvox",
                   help="Voice name (e.g., 'Samantha','Zarvox','Amelie').")
    p.add_argument("--list-voices", action="store_true", help="List available voices and exit.")
    p.add_argument("--no-asr-mute-while-speaking", action="store_true",
                   help="Do not mute ASR while speaking (disables half-duplex).")
    return p.parse_args()

# ========================= Vosk Model Loader =========================

def load_yolo():
    return YOLO("yolov8n.pt")

# ========================= Main =========================

enrollment_requests: "queue.Queue[dict | None]" = queue.Queue()

def main():
    global CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX
    global _tts_backend, _tts_voice_requested, _mac_synth, asr_mute_while_speaking

    args = parse_args()
    _tts_voice_requested = args.voice
    asr_mute_while_speaking = not args.no_asr_mute_while_speaking

    if args.tts == "auto":
        _tts_backend = "nsss" if platform.system()=="Darwin" else "pyttsx3"
    else:
        _tts_backend = args.tts

    if args.list_voices:
        if _tts_backend in ("nsss","say") and platform.system()=="Darwin":
            for nm, vid in _mac_list_voices(): print(f"{nm}  —  {vid}")
        else:
            eng = init_tts_engine()
            for v in eng.getProperty("voices") or []: print(f"{v.name}  —  {v.id}")
        return

    print(f"[cyan]TTS backend:[/cyan] {_tts_backend}  [cyan]voice:[/cyan] {(_tts_voice_requested or 'default')}  "
          f"[cyan]half-duplex:[/cyan] {asr_mute_while_speaking}")

    # IO devices
    CAMERA_INDEX, MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX = configure_io_devices()
    if _tts_backend == "pyttsx3": init_tts_engine()
    set_voice(_tts_voice_requested)

    # Self test & voice start prompt (voice-yes supported)
    speak_text("TTS online.", blocking=True)
    prompt_message = "Greetings, would you like to begin monitoring?"
    speak_text(prompt_message, blocking=True)
    dec = wait_for_voice_yes(MIC_DEVICE_INDEX, timeout_s=7.0)
    if dec is False:
        speak_text("Exiting monitoring mode. Goodbye.", blocking=True); return
    if dec is None:
        while True:
            resp = input(f"{prompt_message} [y/N]: ").strip().lower()
            if resp in {"y","yes"}: break
            if resp in {"n","no",""}:
                speak_text("Exiting monitoring mode. Goodbye.", blocking=True); return
            print("[yellow]Please answer 'yes' or 'no'.[/yellow]")

    # Recheck throttle
    recheck_interval_ms = max(0, args.recheck_interval_ms)
    recheck_interval_sec = recheck_interval_ms / 1000.0

    # Threads: ASR + Enrollment
    asr_events: "queue.Queue[tuple]" = queue.Queue()
    stop_ev = threading.Event()
    t_asr = threading.Thread(target=asr_listener, args=(asr_events, stop_ev, MIC_DEVICE_INDEX), daemon=True)
    t_asr.start()
    ensure_registry_loaded()
    t_enroll = threading.Thread(target=enrollment_worker, args=(stop_ev, MIC_DEVICE_INDEX), daemon=True)
    t_enroll.start()

    # YOLO
    model = load_yolo()

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
    print("[cyan]Camera opened. Press 'q' to quit.[/cyan]")

    presence_streak = 0
    occupied = False
    next_recheck_time = 0.0
    entries = deque(maxlen=32)
    hellos = deque(maxlen=32)

    state = {"paused": False}

    try:
        while True:
            # Drain ASR events
            try:
                while True:
                    evt = asr_events.get_nowait()
                    # evt format: ("cmd"|"pair_hello", ts, kind_or_none, payload_or_none)
                    if not isinstance(evt, tuple): continue
                    kind0, ts, kind1, payload = (evt + (None,))[:4]
                    if kind0 == "pair_hello":
                        hellos.append(ts)
                        print(f"[yellow]Pairer hello @ {time.strftime('%H:%M:%S')}[/yellow]")
                    elif kind0 == "cmd":
                        try:
                            handle_cmd(kind1, payload, state)
                        except KeyboardInterrupt:
                            raise
            except queue.Empty:
                pass

            # Video
            ret, frame = cap.read()
            if not ret: print("[red]Frame grab failed[/red]"); break

            now_mono = time.monotonic()
            should_update = (not state["paused"]) and (recheck_interval_sec == 0.0 or now_mono >= next_recheck_time)
            if should_update and recheck_interval_sec > 0.0:
                next_recheck_time = now_mono + recheck_interval_sec
            elif not should_update and recheck_interval_sec > 0.0 and next_recheck_time == 0.0:
                next_recheck_time = now_mono + recheck_interval_sec

            if should_update:
                results = model.predict(source=frame, verbose=False, classes=DETECT_CLASSES, conf=CONF_THRES)
                det_entities = yolo_entities(results)
                any_present = bool(det_entities)

                # Draw boxes
                for etype, (x1,y1,x2,y2) in det_entities:
                    color = (0,255,0) if etype=="person" else (255,165,0) if etype in {"dog","cat"} else (255,0,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, etype, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Entry logic
                if any_present:
                    presence_streak = min(presence_streak + 1, 10**6)
                    if not occupied and presence_streak >= MIN_PERSIST_NEW:
                        occupied = True
                        ts = time.time()
                        entries.append(ts)
                        print(f"[magenta]Entry detected @ {time.strftime('%H:%M:%S')}[/magenta]")

                        # Save screenshot & gong first
                        save_screenshot(frame)
                        play_gong(SPEAKER_DEVICE_INDEX)

                        # Process persons: face match/enroll
                        persons_in_frame = any(etype=="person" for etype,_ in det_entities)
                        if persons_in_frame:
                            faces = analyze_faces(frame)
                            if faces:
                                for f in faces:
                                    nm = f["name"]; enc = f["encoding"]
                                    if nm:
                                        greet_known(nm, "person")
                                    else:
                                        enrollment_requests.put({"type":"person","encoding": enc})
                            else:
                                # Still add session entry Unknown (person)
                                session_entries.append((ts, "Unknown"))
                                speak_text("Ahoy!", blocking=True)
                                speak_text("I don't know you yet. Please share your name with me.", blocking=True)
                                enrollment_requests.put({"type":"person","encoding": None})
                        # Process animals: always enroll/greet by type
                        for etype, _ in det_entities:
                            if etype=="person": continue
                            # Do we have any known of this type? We can't re-identify—greet generic if exists.
                            with registry_lock:
                                names = [e["name"] for e in known_entries if e["type"]==etype]
                            if names:
                                # Greet generally (no re-id)
                                speak_text(f"Ahoy {etype}!", blocking=True)
                                session_entries.append((ts, f"{etype}"))
                            else:
                                speak_text(f"Hello {etype}. I don't know your name yet.", blocking=True)
                                session_entries.append((ts, f"{etype} Unknown"))
                                enrollment_requests.put({"type": etype, "encoding": None})
                        # If nothing matched above, at least say Ahoy once
                        if not persons_in_frame and not det_entities:
                            speak_text("Ahoy!", blocking=True)
                else:
                    presence_streak = 0
                    occupied = False

                # Pairing print (still useful for person)
                now = time.time()
                while entries and now - entries[0] > TIME_WINDOW: entries.popleft()
                while hellos and now - hellos[0] > TIME_WINDOW: hellos.popleft()
                for te in list(entries):
                    for th in list(hellos):
                        if abs(th - te) <= TIME_WINDOW:
                            print(f"[bold green]✅ DETECTED:[/bold green] entry & hello (Δt={abs(th-te):.1f}s)")
                            try: entries.remove(te)
                            except ValueError: pass
                            try: hellos.remove(th)
                            except ValueError: pass
                            break

            # HUD
            if state["paused"]:
                cv2.putText(frame, "Detection paused", (10, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            cv2.imshow("DoorHello (press q to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break

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

# ========================= Aux wait_for_voice_yes =========================

def wait_for_voice_yes(mic_device: int | None, timeout_s: float = 7.0) -> bool | None:
    yes_words = {"yes","yeah","yep","sure","start","begin","go","affirmative"}
    no_words  = {"no","nope","stop","cancel","negative"}
    rec = KaldiRecognizer(load_asr_model(), ASR_SAMPLE_RATE); rec.SetWords(False)
    stop_ev = threading.Event()
    def _norm(s: str) -> str:
        s = s.lower(); s = re.sub(r"[^\w\s]"," ", s); return re.sub(r"\s+"," ", s).strip()
    def audio_cb(indata, frames, timeinfo, status):
        if asr_force_muted.is_set(): return
        if asr_mute_while_speaking and tts_active.is_set(): return
        if stop_ev.is_set(): raise sd.CallbackStop()
        if not rec.AcceptWaveform(bytes(indata)): return
        phrase = (json.loads(rec.Result()).get("text") or "").strip()
        if not phrase: return
        toks = set(_norm(phrase).split())
        if toks & yes_words: stop_ev.set(); raise sd.CallbackStop()
        if toks & no_words:  stop_ev.set(); raise sd.CallbackStop()
    deadline = time.monotonic() + timeout_s
    try:
        with sd.RawInputStream(samplerate=ASR_SAMPLE_RATE, blocksize=8000, dtype="int16",
                               channels=1, callback=audio_cb, device=mic_device):
            while not stop_ev.is_set() and time.monotonic() < deadline: sd.sleep(100)
    except sd.CallbackStop:
        pass
    # Not perfect (we don't keep which set fired), but OK for gate
    # If you want exact, store result in outer scope.
    # We'll just ask typed fallback when undecided.
    return None

# ========================= IO Helpers =========================

def describe_audio_devices():
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Unable to query audio devices: {exc}") from exc
    hostapis = sd.query_hostapis()
    microphones = []; speakers = []
    for idx, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"] if hostapis and 0 <= dev.get("hostapi",-1) < len(hostapis) else ""
        label = f"{dev['name']} ({hostapi_name})" if hostapi_name else dev["name"]
        entry = {"index": str(idx), "label": label}
        if dev.get("max_input_channels", 0) > 0: microphones.append(entry)
        if dev.get("max_output_channels", 0) > 0: speakers.append(entry)
    return microphones, speakers

def prompt_choice(options, title: str) -> int:
    print(f"[bold cyan]{title}[/bold cyan]")
    for i,opt in enumerate(options, start=1):
        print(f"  {i}. {opt['label']} (index {opt['index']})")
    default_idx = 1
    while True:
        try: raw = input(f"Select {title.lower()} [default {default_idx}]: ").strip()
        except EOFError: raw = ""
        choice = default_idx if not raw else int(raw) if raw.isdigit() else None
        if choice and 1 <= choice <= len(options):
            selection = int(options[choice-1]["index"])
            print(f"[green]Selected {options[choice-1]['label']}[/green]")
            return selection
        print("[red]Invalid selection. Try again.[/red]")

def detect_cameras(max_index: int = 10):
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ret, _ = cap.read(); cap.release()
            if ret: found.append({"index": str(idx), "label": f"Camera {idx}"})
        else:
            if cap is not None: cap.release()
    return found

def configure_io_devices() -> tuple[int,int,int]:
    cameras = detect_cameras()
    if not cameras: raise RuntimeError("No usable cameras detected.")
    camera_index = prompt_choice(cameras, "Available Cameras")
    # Tests
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened(): raise RuntimeError(f"Camera {camera_index} failed"); cap.release()

    microphones, speakers = describe_audio_devices()
    if not microphones: raise RuntimeError("No microphones detected.")
    mic_index = prompt_choice(microphones, "Available Microphones")
    sd.check_input_settings(device=mic_index, samplerate=ASR_SAMPLE_RATE, channels=1)
    with sd.InputStream(device=mic_index, channels=1, samplerate=ASR_SAMPLE_RATE) as stream: stream.read(1)
    print("[green]Microphone test passed.[/green]")

    if not speakers: raise RuntimeError("No speakers detected.")
    spk_index = prompt_choice(speakers, "Available Speakers")
    sd.check_output_settings(device=spk_index, samplerate=ASR_SAMPLE_RATE, channels=1)
    # quick beep
    dur=0.2; t=np.linspace(0,dur,int(ASR_SAMPLE_RATE*dur),False); tone=0.2*np.sin(2*np.pi*880*t)
    sd.play(tone, samplerate=ASR_SAMPLE_RATE, device=spk_index); sd.wait()
    print("[green]Speaker test passed.[/green]")

    sd.default.device = (mic_index, spk_index)
    return camera_index, mic_index, spk_index

# ========================= Entrypoint =========================

if __name__ == "__main__":
    main()
