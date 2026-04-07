"""
Use this script to verify faster-whisper is working correctly in the current environment


Re: Linux:

    faster-whisper 1.2.1 plays nice with ctranslate2 4.7.1 in tandem with torch 280

    (it used to conflict badly with various versions of cudnn associated with certain versions of torch etc)

    done 


fyi the order in which ctranslate2 and torch are loaded does not matter

"cudnn version matrix"

    ctranslate2 4.6.1     cudnn 9.1.0
    
    torch 280             cudnn 91002 9.10.2
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
PATH = "something.flac"
data, sr = librosa.load(PATH) # would resample this in real use
model = WhisperModel("large-v3", device="cuda")
o = model.transcribe(audio=data, word_timestamps=True, language=None)
print("did not crash. transcription result:", o)

