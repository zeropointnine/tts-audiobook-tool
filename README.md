# Description

This is a generative-AI audiobook creation tool that supports a growing list of text-to-speech models which utilize zero shot voice cloning:
- Chatterbox TTS
- Fish OpenAudio S1-mini
- Higgs Audio V2
- Oute TTS

The app features a number of quality control measures designed to mitigate the inherently variable nature of generative text-to-speech models:

- Rational segmentation of long text at paragraph/sentence/phrase boundaries, as needed
- Detection and correction of inference errors and hallucinations using speech-to-text comparison to the source text
- Semantically-aware modulation of *caesuras* between concatenated sound segments (think "prosody")
- Industry standard loudness normalization (EBU R 128)

Plain-vanilla interactive console interface.

### Web-based player:

The app embeds text and timing information into the metadata of the FLAC and M4A files it generates, allowing for the included web app to display the audiobook's text in sync with the generated audio (similar to Kindle+Audible or the Google Play Books app). The web app can be launched directly from the html source (no need for a web server), or from the mapped github.io url.

**Online examples**, all using the same source text and voice clone sample:

- [Online example using Oute](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-oute.abr.m4a)
- [Online example using Fish OpenAudio S1-mini](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-s1-mini.abr.m4a)
- [Online example using Chatterbox](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-chatterbox.abr.m4a)
- [Online example using Higgs Audio V2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-higgs.abr.m4a)
- [Online example using Higgs Audio V2 (a different voice this time, high temperature)](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-higgs-different-voice.abr.m4a)

### Bonus feature: Enhance existing audiobooks

Using speech-to-text, the app is able to embed its custom metadata into pre-existing (ie, professionally produced) audiobook files so that they can be opened and used with the custom player/reader.

Select `Options` > `Enhance existing audiobook`, and select your source audiobook file (typically M4A or M4B) and corresponding book text. This feature is experimental.

### How to create an audiobook (quick summary):

1. Assign a working project directory.
2. Select a short reference audio clip for the voice clone.
3. Select the source text. 3b. Optionally define file split points.
4. Start inferencing, and ... be prepared to do some waiting.
5. Concatenate the generated audio segments to create the final FLAC or M4A file/s.
6. Optionally use the aforementioned web player/reader to play/read your audiobook.


# Installation

First, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

Clone the repository and cd into it:

    git clone tts-audiobook-tool
    cd tts-audiobook-tool

A separate virtual environment must be created for each model you want to use. Perform the operations as described in one or more of the sections below (and refer to the respective TTS model's project github pages for further guidance as needed). Model-specific options will be enabled automatically in the app based on which virtual environment is active.

In all cases, the CUDA flavor of torch may require an extra install step depending on platform and installed CUDA version, in the usual manner. Just make sure to use the same torch version as specified in the respective `requirements-*.txt` file.

Once the desired virtual environment/s is set up and has been activated, run the app by entering:

    python -m tts_audiobook_tool


## Install for Chatterbox TTS:

Initialize a Python v3.12 virtual environment named "venv-chatterbox". For example:

    python.exe -m venv venv-chatterbox

Activate the virtual environment:

    venv-chatterbox\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-chatterbox.txt


## Install for Higgs Audio V2:

> **ℹ️ Note!**
> Higgs V2 almost requires 24GB VRAM (yes really)

Initialize a Python v3.12 virtual environment named "venv-higgs". For example:

    python.exe -m venv venv-higgs

Activate the virtual environment:

    venv-higgs\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-higgs.txt

Note that the above `requirements` file draws from a personal fork of the `higgs-audio` library due to the fact that the original repo is missing some needed `__init__.py` files at the moment.


## Install for Fish OpenAudio-S1-mini:

Initialize a Python v3.12 virtual environment named "venv-fish". For example:

    python.exe -m venv venv-fish

Activate the virtual environment:

    venv-fish\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-fish.txt

And if using CUDA on Windows, also do:

    pip install triton-windows

And then, two extra steps:

You have to opt in to gain access to the Fish/OpenAudio model by visiting the FishAudio [Hugging Face page](https://huggingface.co/fishaudio/openaudio-s1-mini), using a logged-in HF account.

Then, [generate a Hugging Face access token](https://huggingface.co/settings/tokens) and paste the token at the command line after entering:

    huggingface-cli login

## Install for Oute TTS:

Initialize a Python v3.12 virtual environment named "venv-oute". For example:

    python.exe -m venv venv-oute

Activate the virtual environment:

    venv-oute\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-oute.txt

### Oute TTS model configuration

Running the app optimally with Oute TTS requires extra steps due to the way the model supports multiple backends, model sizes and quantizations. You will need to review and hand-edit the Python file **`config_oute.py`** accordingly.

The [OuteTTS Github project page](https://github.com/edwko/OuteTTS) documents these various options. But here are some recommendations based on my own testing...

**Nvidia (CUDA) cards:**

Prefer the *ExLllama2* backend if at all possible: `backend=outetts.Backend.EXL2` (See the example Oute config in `config_oute.py`). However, this requires successfully installing into the environment three extra things:
- exllama2 library (`pip install exllamav2`)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)
- Triton (eg, on Windows, `pip install triton-windows`).

Alternatively, `Backend.HF` is also hardware accelerated but considerably slower.

I couldn't get acceleration going using `Backend.LLAMACPP` but I'm not sure if that's just be me.

**Mac with Apple silicon:**

Use `Backend.LLAMACPP`.

# Usage notes

The app saves its state between sessions, so you can interrupt the program at any time and resume later (important due to how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in  manageable chunks over time, allowing you to to use early chapter files before the full text is completed.

Note too that it's possible to utilize different voices and even different models while generating the audio segments for a given project.

### Voice cloning

When prepping reference audio for voice cloning, it's worthwhile to prepare three or more sound samples from a given source (not just one), and then testing each one out in turn on a short passage of the intended text, as the quality and characteristics of each voice clone from the same source can vary quite a bit (as well as the word error rate!).

### Inference speed, expectations, etc.

With Oute TTS, I get inference speeds of 80-85% of realtime using `Backend.EXL2` with a GeForce 3080Ti. And about 20% with an M1 MacBook Pro (using llama.cpp as the configured backend).

Chatterbox is considerably faster, about 130% of realtime for me using a GeForce 3080Ti, 25-30% of realtime on M1 Macbook Pro.

On the same setup, Fish OpenAudio S1-mini is the fastest of the three by far, about 2-3x faster than Chatterbox.


# Updates

**2025-08-10**

Migrated from openai-whisper to faster-whisper (faster, less memory).

**2025-08-06**

Added support for **Higgs Audio V2** (3B base model). Adding support some of its model-specific features (system prompt, context continuation, etc) is planned.

**2025-07-18**

Added support for **Fish OpenAudio S1-mini**

Logic to detect dropped phrases at end of generated audio segments (common occurrence with the Fish model)

Added utility to transcode and concatenate directory of MP3 chapter files to M4A (meant for use with the "Enhance existing audiobook" tool)

**2025-07-02**

New feature: **Real-time generation and playback** (`Options > Real-time generation and playback`)

This serves as a quicker and more "casual" alternative to the regular `Generate audio` UI flow, and allows for more-or-less immediate and uninterrupted audio playback (contingent on system performance, ofc). It employs the same quality control measures except for loudness normalization, and does not save its output.

**2025-06-28** (many)

Generated audio segments now have silence trimmed off the ends, and in the concatenation step, stitched-together lines have pauses of varying lengths inserted at paragraph, sentence, and phrase boundaries, resulting in much improved prosody / flow.

Loudness normalization is now being applied correctly (on the final audio file instead of per audio segment)

Better detection of undesired repeating phrases (Oute especially)

Better detection and fix for spooky Chatterbox noises at the end of prompts

Short 1-2 word sentences now get grouped with adjacent sentences to minimize Chatterbox and Oute's issues with short prompts

Streamlined handling of audio data throughout

Encoding audiobook files in AAC/M4A format no longer requires intermediate FLAC step

Streamlined some UI

Some improvements to the web player/reader
