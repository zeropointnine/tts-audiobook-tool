# Description

This is a generative-AI audiobook creation tool that supports a growing list of text-to-speech models which utilize zero shot voice cloning:
- IndexTTS2
- VibeVoice 1.5B
- Higgs Audio V2
- Fish OpenAudio S1-mini
- Chatterbox TTS
- Oute TTS

The app features a number of quality control measures designed to mitigate the inherently variable nature of generative text-to-speech models:

- Rational segmentation of long text at paragraph/sentence/phrase boundaries, as needed
- Detection and correction of many inference errors and hallucinations using speech-to-text comparison to the source text
- Semantically-aware modulation of *caesuras* between concatenated sound segments (think "prosody")
- Industry standard loudness normalization (EBU R 128)

Plain-vanilla text interface.

### Web-based player:

The app embeds text and timing information into the metadata of the FLAC and M4A files it generates, allowing for the included web app to display the audiobook's text in sync with the generated audio (similar to Kindle+Audible or the Google Play Books app). The web app can be launched directly from the html source (no need for a web server), or from the mapped [github.io page](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/).

**Some example outputs**, all using the same source text and same 15-second voice clone sample:

- [IndexTTS2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-indextts2.abr.m4a)
- [IndexTTS2 (with added emotional guidance voice sample)](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-indextts2-plus-emo.abr.m4a)
- [VibeVoice 1.5B](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-vibevoice-1.5b.abr.m4a)
- [Higgs Audio V2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-higgs.abr.m4a)
- [Higgs Audio V2 (a different voice this time, and at high temperature)](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-higgs-different-voice.abr.m4a)
- [Fish OpenAudio S1-mini](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-s1-mini.abr.m4a)
- [Chatterbox](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-chatterbox.abr.m4a)
- [Oute](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-oute.abr.m4a)

### Bonus feature: Enhance existing audiobooks

Using speech-to-text, the app is able to embed its custom metadata into pre-existing (ie, professionally produced) audiobook files so that they can be opened and used with the custom player/reader.

Select `Options` > `Enhance existing audiobook`, and select your source audiobook file (typically M4A or M4B) and corresponding book text. This feature is experimental.

### How to create an audiobook (quick summary):

1. Assign a working project directory.
2. Select a short reference audio clip for the voice clone.
3. Select the source text. 3b. Optionally define file split points.
4. Start inferencing, and ... be prepared to wait.
5. Concatenate the generated audio segments to create the final FLAC or M4A file/s.
6. Optionally use the aforementioned web player/reader to play and read your audiobook.


# Installation

First, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

Clone the repository and cd into it:

    git clone https://github.com/zeropointnine/tts-audiobook-tool
    cd tts-audiobook-tool

A separate virtual environment must be created for each model you want to use. Perform the operations as described in one or more of the sections below. Model-specific options will be enabled automatically in the app based on which virtual environment has been enabled.

Finally, run the app by entering:

    python -m tts_audiobook_tool

## Install for IndexTTS2

Initialize a **Python v3.11** virtual environment named `venv-indextts2`. For example:

    path\to\python3.11\python.exe -m venv venv-indextts2

Activate the virtual environment:

    venv-indextts2\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-indextts2.txt

### Additional steps for CUDA:

Uninstall the vanilla version of torch:

    pip uninstall torch torchaudio

Install torch 2.8 for CUDA v12.8:

    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

> **ℹ️ Note:**
> To run the app using IndexTTS2 comfortably within 12GB of VRAM, consider setting FP16 to True

## Install for VibeVoice

Initialize a **Python v3.11** virtual environment named `venv-vibevoice`. For example:

    path\to\python3.11\python.exe -m venv venv-vibevoice

Activate the virtual environment:

    venv-vibevoice\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-vibevoice.txt

Note that because Microsoft famously removed the source code from their project github repo, we pull from an archived, third-party fork, [vibevoice-community](https://github.com/vibevoice-community/VibeVoice).

### Additional steps for CUDA:

Uninstall the vanilla version of torch:

    pip uninstall torch

Install torch 2.6 for CUDA v12.6:

    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

Finally, install Flash Attention. The procedure for doing so varies by operating system. On Windows, I'm using a wheel that has the filename `flash_attn-2.7.4+cu126torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl`.


## Install for Higgs Audio V2:

> **ℹ️ Note!**
> Higgs V2 requires CUDA, and 24 GB is almost a necessity (yes really)

Initialize a **Python v3.12** virtual environment named `venv-higgs`. For example:

    path\to\python3.12\python.exe -m venv venv-higgs

Activate the virtual environment:

    venv-higgs\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-higgs.txt

Note that the above `requirements` file draws from a personal fork of the `higgs-audio` library due to the fact that the higgs repo is missing `__init__.py` files required for module use.

Uninstall the vanilla version of torch that just got installed, and install the CUDA version of torch (latest is fine).

## Install for Fish OpenAudio-S1-mini:

Initialize a **Python v3.12** virtual environment named "venv-fish". For example:

    path\to\python3.12\python.exe -m venv venv-fish

Activate the virtual environment:

    venv-fish\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-fish.txt

And then, two extra steps:

You have to opt in to gain access to the Fish/OpenAudio model by visiting the [FishAudio Hugging Face page](https://huggingface.co/fishaudio/openaudio-s1-mini), using a logged-in Hugging Face account.

Then, [generate a Hugging Face access token](https://huggingface.co/settings/tokens) and paste the token at the command line after entering:

    hf auth login

When the app runs for the first time and tries to download the models from huggingface, it should now be authorized to do so.

On Linux and macOS, portaudio must be installed (eg, `brew install portaudio`)

### Additional steps for CUDA:

Uninstall the vanilla version of torch that just got installed, and install the [CUDA version of torch v2.7.1](https://pytorch.org/get-started/previous-versions/)

## Install for Chatterbox TTS:

Initialize a **Python v3.12** virtual environment named "venv-chatterbox". For example:

    path\to\python3.12\python.exe -m venv venv-chatterbox

Activate the virtual environment:

    venv-chatterbox\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-chatterbox.txt

### Additional steps for CUDA:

Uninstall the vanilla version of torch:

    pip uninstall torch torchaudio

Install torch 2.6 for CUDA v12.6:

    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

## Install for Oute TTS:

Initialize a **Python v3.12** virtual environment named "venv-oute". For example:

    path\to\python3.12\python.exe -m venv venv-oute

Activate the virtual environment:

    venv-oute\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-oute.txt

If using CUDA, uninstall the vanilla version of torch that just got installed, and install the [CUDA version of torch v2.6.0](https://pytorch.org/get-started/previous-versions/)

### Oute TTS model configuration

Running the app optimally with Oute TTS requires extra steps due to the way the model supports multiple backends, model sizes and quantizations. You will need to review and hand-edit the file **`config_oute.py`** accordingly.

The [OuteTTS Github project page](https://github.com/edwko/OuteTTS) documents these various options. But here are some recommendations based on my own testing...

**Nvidia cards (CUDA):**

Prefer the *ExLllama2* backend if at all possible: `backend=outetts.Backend.EXL2` (See the example Oute config in `config_oute.py`). However, this requires successfully installing into the environment three extra things:
- exllama2 library (`pip install exllamav2`)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) (note that on Windows, you may need to source a compatible wheel for this)
- Triton (eg, on Windows, `pip install triton-windows`).

Alternatively, `Backend.HF` is also hardware accelerated but considerably slower. Flash Attention 2 is optional in this case, and also does not speed up inference in practice.

**Mac with Apple silicon:**

Use `Backend.LLAMACPP`.

# Usage notes

The app saves its state between sessions, so you can interrupt the program at any time and resume later (important due to how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in  manageable chunks over time, allowing you to to use early chapter files before the full text is completed.

Note too that it's possible to utilize different voices and even different models while generating the audio segments for a given project.

### Voice cloning

When prepping reference audio for voice cloning, it's worthwhile to prepare three or so different sound samples from a given source (not just one), and then test each one out in turn on a short passage of the intended text, as the quality and characteristics of each voice clone from the same source can vary quite a bit (as well as the word error rate).

### Inference speeds, expectations

These are my anecdotal inference speeds (running Windows unless otherwise noted). The app adopts each respective model's reference inference implementation logic as much as possible.

| TTS Model               | Hardware             | Speed | Notes |
| ----------------------- | -------------------- | ----- | ----- |
| IndexTTS2               | GTX 4090             | ~150% realtime |
|                         | GTX 3080 Ti          | ~90% realtime |
| VibeVoice 1.5B          | GTX 3080 Ti          | ~120% realtime | with Flash attention 2 enabled
|                         | Macbook Pro M1       | ~40% realtime |
| Higgs V2 3B             | GTX 4090             | 200+% realtime | inference speed inversely proportional to voice sample duration, FYI
|                         | GTX 3080 Ti          | N/A | does not fit in 12 GB VRAM
| Fish OpenAudio S1-mini  | GTX 3080 Ti          | 500+% realtime | best combination of inference speed and quality output IMO
|                         | Macbook Pro M1 (MPS) | ~15% realtime
| Chatterbox              | GTX 3080 Ti          | ~130% realtime
|                         | Macbook Pro M1 (MPS) | 20-35% realtime
| Oute                    | GTX 3080 Ti          | ~90% realtime | using `outetts.Backend.EXL2`
|                         | Macbook Pro M1 (MPS) | 20-25% realtime | using `outetts.Backend.LLAMACPP`


# Update highlights

**2025-10-24**

Added option to disable transcription validation (`Options` > `Whisper model type` > `Disabled`). This is only advisable when using IndexTTS2, which generates the least number of inference errors of any of the supported models to date.

FYI, Whisper model type is now an app-wide preference setting rather than a per-project setting.

**2025-10-23**

Added option to insert a "page turn" sound effect at section breaks (default is on; must replace text to enable on pre-existing projects)

Rearranged some menu items

**2025-10-03**

Added support for **IndexTTS2**, including secondary voice reference sample guiding emotion and "emotion vectors".

Added option to specify Whisper model (large-v3 or large-v3-turbo)

**2025-09-15**

Added support for VibeVoice alternate models (ie, large/7B model)

**2025-09-12**

Added support for **VibeVoice**.

**2025-08-10**

Migrated from openai-whisper to faster-whisper (faster, less memory, equivalent accuracy).

**2025-08-06**

Added support for **Higgs Audio V2** (3B base model).

**2025-07-18**

Added support for **Fish OpenAudio S1-mini**

Logic to detect dropped phrases at end of generated audio segments (common occurrence with the Fish model)

Added utility to transcode and concatenate directory of MP3 chapter files to M4A (meant for use with the "Enhance existing audiobook" tool)

**2025-07-02**

New feature: **Real-time playback** (`Options > Real-time playback`)

This serves as a quicker and more "casual" alternative to the regular `Generate audio` UI flow, and allows for more-or-less immediate and uninterrupted audio playback (contingent on system performance, ofc). It employs the same quality control measures except for loudness normalization, and does not save its output.

**2025-06-28** (many)

Generated audio segments now have silence trimmed off the ends, and in the concatenation step, stitched-together lines have pauses of varying lengths inserted at paragraph, sentence, and phrase boundaries, resulting in much improved prosody / flow.

Loudness normalization is now being applied correctly (on the final audio file instead of per audio segment)

Better detection of undesired repeating phrases (Oute especially)

Better detection and fix for spooky Chatterbox noises at the end of prompts

Short 1-2 word sentences now get grouped with adjacent sentences to mitigate Chatterbox and Oute's issues with short prompts

Streamlined handling of audio data throughout

Encoding audiobook files in AAC/M4A format no longer requires intermediate FLAC step

Streamlined some UI

Some improvements to the web player/reader
