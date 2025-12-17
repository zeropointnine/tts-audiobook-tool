"""
Re: Linux:

faster-whisper must use ctranslate2 4.6.1, which uses cudnn 9.1.0

    faster-whisper==1.2.1
    ctranslate2==4.6.1

the latest version of torch which also uses cudnn 9.1.0 is torch 260, or torch 260 with cuda <= 124

    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

the order in which ctranslate2 and torch are loaded does not matter

"cudnn version matrix"

    ctranslate2 4.6.1     cudnn 9.1.0
    latest torch          cudnn 9.10
    torch 260+cu126       cudnn 9.5 -- BAD, confirmed 
    torch 260+cu124       cudnn 9.1 -- GOOD
    torch 260             cudnn 9.1 -- GOOD
"""


# TODO: i notice pip list shows "nvidia-cudnn-cu12 9.1.0.70" -- can i manually install that with a later cuda version?

print("importing torch")
import torch
print("torch cudnn version:", torch.backends.cudnn.version())


print("running faster-whisper")
from faster_whisper import WhisperModel
import librosa
path = "something.flac"
data, sr = librosa.load(path) # would resample this in real use
model = WhisperModel("large-v3", device="cuda")
o = model.transcribe(audio=data, word_timestamps=True, language=None)
print("did not crash. transcription result:", o)

