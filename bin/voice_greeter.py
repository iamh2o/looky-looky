#!/usr/bin/env python3
"""
voice_greeter.py — local speaker ID greeter

What it does
------------
1) Speaks: "ahoy! How may I help you?"
2) Records a short voice sample from mic.
3) Compares speaker embedding to enrolled profiles in ./voice_profiles.
4) If matched above threshold: greets by name.
5) If unknown: offers to create a new voice profile (records once, saves).

Dependencies
------------
pip install resemblyzer sounddevice soundfile pyttsx3 numpy

Notes
-----
- Profiles are stored as: ./voice_profiles/<safe_name>.npy (embedding)
- A simple ./voice_profiles/profiles.json tracks display names.
- Threshold is conservative (0.82). Adjust if you get false rejects/accepts.
- All offline. No cloud calls.

Tested on macOS; should work on Linux with PortAudio available.
"""

import os
import json
import time
import queue
import signal
import threading
import numpy as np

import sounddevice as sd
import soundfile as sf
import pyttsx3

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

# ---------- Config ----------
PROFILES_DIR = Path("voice_profiles")
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_DB = PROFILES_DIR / "profiles.json"

SAMPLE_RATE = 16000       # target SR for embeddings
CAPTURE_SECONDS = 5       # seconds to capture per attempt
THRESHOLD = 0.82          # cosine similarity threshold for "same speaker"
PROMPT_VOICE = None       # e.g., "Samantha" on macOS, or None for default
# ----------------------------

# ---------- TTS ----------
def speak(text: str):
    eng = pyttsx3.init()
    if PROMPT_VOICE is not None:
        # Attempt to set a specific voice by name substring
        for v in eng.getProperty("voices"):
            if PROMPT_VOICE.lower() in (v.name or "").lower():
                eng.setProperty("voice", v.id)
                break
    eng.say(text)
    eng.runAndWait()
    eng.stop()
# -----------------------

# ---------- Storage ----------
def load_db():
    if PROFILES_DB.exists():
        with open(PROFILES_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"profiles": []}

def save_db(db):
    with open(PROFILES_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def safe_filename(name: str) -> str:
    # Keep letters, numbers, space, dash, underscore. Replace others with dash.
    return "".join(ch if ch.isalnum() or ch in " -_." else "-" for ch in name).strip().replace(" ", "_")
# -----------------------------

# ---------- Audio capture ----------
def record_blocking(seconds=CAPTURE_SECONDS, sample_rate=SAMPLE_RATE) -> np.ndarray:
    """
    Capture mono audio for `seconds` and return float32 numpy array at `sample_rate`.
    """
    speak("Listening now.")
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    data = sd.rec(int(seconds * sample_rate), dtype="float32")
    sd.wait()
    # Optional: write temp wav for debugging
    # sf.write("debug_input.wav", data, sample_rate)
    return data.flatten()
# -----------------------------------

# ---------- Embeddings ----------
_encoder = None

def get_encoder() -> VoiceEncoder:
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder

def embed_audio(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    # Resemblyzer's preprocess does resampling + VAD; pass raw samples & sr
    pwav = preprocess_wav(wav, source_sr=sr)
    if len(pwav) < 0.5 * sr:
        # Too little speech after VAD; fall back to raw (less ideal)
        pwav = wav
    enc = get_encoder()
    emb = enc.embed_utterance(pwav)
    return emb.astype(np.float32)
# ----------------------------------

# ---------- Matching ----------
def list_profiles():
    db = load_db()
    out = []
    for p in db.get("profiles", []):
        pth = PROFILES_DIR / p["file"]
        if pth.exists():
            out.append({
                "name": p["name"],
                "file": str(pth),
                "embedding": np.load(pth)
            })
    return out

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def best_match(emb: np.ndarray, profiles: list) -> tuple[str, float] | tuple[None, float]:
    if not profiles:
        return None, 0.0
    best_name, best_score = None, -1.0
    for p in profiles:
        s = cosine_similarity(emb, p["embedding"])
        if s > best_score:
            best_score = s
            best_name = p["name"]
    return best_name, best_score
# ----------------------------------

# ---------- Enrollment ----------
def enroll_new_profile(emb: np.ndarray):
    speak("I don't recognize this voice. Would you like to create a voice profile now? Type y or n in the terminal.")
    choice = input("Create new voice profile? [y/N]: ").strip().lower()
    if choice != "y":
        speak("Okay. Skipping enrollment.")
        return

    # Name capture via keyboard for accuracy & multi-word support
    while True:
        name = input("Enter the person's name (multi-word OK): ").strip()
        if not name:
            print("Name cannot be empty.")
            continue

        db = load_db()
        existing_names = {p["name"] for p in db.get("profiles", [])}
        if name in existing_names:
            print(f"'{name}' already exists. Please add a last name or a unique label.")
            continue

        fname = safe_filename(name) or f"profile_{int(time.time())}"
        dest = PROFILES_DIR / f"{fname}.npy"
        np.save(dest, emb)

        db.setdefault("profiles", []).append({
            "name": name,
            "file": f"{fname}.npy"
        })
        save_db(db)
        speak(f"Profile created for {name}. Hello, {name}.")
        print(f"[enrolled] {name} -> {dest}")
        break
# -----------------------------------

# ---------- Main flow ----------
def main():
    # Opening line
    speak("ahoy! How may I help you?")

    # Record and embed
    try:
        wav = record_blocking()
    except Exception as e:
        print(f"[error] Audio capture failed: {e}")
        speak("Sorry. I could not access the microphone.")
        return

    try:
        emb = embed_audio(wav, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"[error] Embedding failed: {e}")
        speak("Sorry. I ran into a problem analyzing your voice.")
        return

    # Match against profiles
    profiles = list_profiles()
    name, score = best_match(emb, profiles)

    if name is not None and score >= THRESHOLD:
        print(f"[match] {name} (cosine={score:.3f})")
        speak(f"Welcome back, {name}. What can I do for you?")
    else:
        if name:
            print(f"[near-miss] Best was {name} (cosine={score:.3f}) below threshold {THRESHOLD}.")
        else:
            print("[no profiles or no good match].")
        enroll_new_profile(emb)

if __name__ == "__main__":
    # Be nice on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: exit(0))
    main()
