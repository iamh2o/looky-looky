# looky-looky

**MAC ONLY**

AI door monitor to recognize when someone comes into the frame. The AI should greet them if they are recognized, and if not recognized, ask themn for the name they wish to be known as in the future and record a profile for them to greet them by name in the future.  Each recognized person should be told their presence has been logged and read a koan.


to setup the venv, run bin/setup.sh

to run the app
source .venv/bin/activate || (source bin/setup.sh && source .venv/bin/activate)
python bin/door_hello.py

```text
DoorHello — camera+mic monitor with on-device person detect (YOLOv8n),
keyword ASR (Vosk), face recognition, and robust TTS backends.

Adds voice commands (always listening while scanning):
  rosy, pause      -> pause entry scanning (keep listening for commands)
  rosy, resume     -> resume entry scanning
  rosy, exit       -> cleanly exit
  rosy, hello      -> reply with fixed phrase (also true for a lone 'hello')
  rosy, whats up -> summarize seen names since this run

'hello' used ALONE is a command; 'hello' inside a longer phrase still
counts for the entry+hello pairing logic.
```
