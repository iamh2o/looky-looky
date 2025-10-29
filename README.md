# looky-looky

AI door monitor to recognize when someone comes into the frame. The AI should greet them if they are recognized, and if not recognized, ask themn for the name they wish to be known as in the future.  Each recognized person should be told their presence has been logged.


to setup the venv, run bin/setup.sh

to run the app
source .venv/bin/activate || (source bin/setup.sh && source .venv/bin/activate)
python bin/door_hello.py

