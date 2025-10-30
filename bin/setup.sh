
/usr/bin/xcode-select --install || true

# 1) Create project + venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Dependencies
pip install ultralytics opencv-python numpy sounddevice vosk rich face-recognition resemblyzer

# 3) (Optional) pin exact versions for reproducibility
pip freeze > requirements.txt

curl -L -o vosk-small.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-small.zip
rm vosk-small.zip
