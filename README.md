# Description

This is an audiobook creation tool, designed to support multiple text-to-speech models. Currently, Oute TTS and Chatterbox TTS are supported. Both Oute and Chatterbox support "zero shot voice cloning", which is in my opinion what makes creating audiobooks using these two models so appealing.

The app attempts to detect and correct inference errors using speech-to-text, regulates the pacing of pauses between sentences/sound segments, and applies loudness normalization to the generated audio. These are important quality control features due to the uh *variable* nature of generative text-to-speech models.

**Web-based player:**

The app embeds text and timing information into the metadata of the FLAC and M4A files it generates, allowing for the included web app to display the audiobook's text in sync with the generated audio (think Kindle+Audible). You simply open `browser_player/index.html` in the browser or navigate to the mapped [github.io page](https://zeropointnine.github.io/tts-audiobook-tool/browser_player).

[Online example 1 (Oute)](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-oute.m4a)

[Online example 2 (same text, using Chatterbox)](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-chatterbox.m4a)

**Bonus speech-to-text feature:**

The application can also add its custom metadata to pre-existing (ie, professionally produced) audiobook files (M4B, etc) so that they can be opened using the custom player/reader. Select `Options` > `Add app metadata to a pre-existing audiobook`, and point it at your source audiobook file and corresponding book text. This feature is experimental.

**How to create an audiobook (quick summary):**

1. Assign a working project directory.
2. Select a short reference audio clip for the voice clone.
3. Select the source text. 3b. Optionally define file split points.
4. Start inferencing, and ... be prepared to wait.
5. Concatenate the generated audio segments to create the final FLAC or M4A file/s.
6. Optionally use the aforementioned web player/reader to play/read your audiobook.


# Installation

Firstly, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

Clone the repository and cd into it:

    git clone tts-audiobook-tool
    cd tts-audiobook-tool

A separate virtual environment must be created for the Chatterbox TTS model and for Oute TTS. Perform the operations as described in one or both of the sections below. Model-specific options will be enabled automatically based on which virtual environment is active.

Once the desired virtual environment/s are set up and activated, run the app by entering:

    python -m tts_audiobook_tool


## Install the Chatterbox TTS flavor of the application:

This is considerably easier compared to setting up the Oute environment:

Initialize a Python virtual environment named "venv-chatterbox" (Python 3.13 or recent earlier versions should be okay). For example:

    python.exe -m venv venv-chatterbox

Activate the virtual environment:

    venv-chatterbox\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-chatterbox.txt

Installing the CUDA version of torch must be done manually as an extra step in the typical manner: Uninstall the vanilla version of torch (`pip uninstall torch torchaudio torchvision`) and then install the correct version of CUDA torch for your setup ([link](https://pytorch.org/get-started/locally/)).


## Install the Oute TTS flavor of the application:

Initialize a Python v3.12 virtual environment named "venv-oute" (nb, *not* v3.13). For example:

    C:\path\to\v3.12\python.exe -m venv venv-oute

Activate the virtual environment:

    venv-oute\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-oute.txt

### Oute TTS model configuration

The Oute TTS model requires some configuration, as it supports multiple backends and more than one model flavor, which can greatly affect inference speed and also output quality. You will want to review and hand-edit the Python file **`config_oute.py`** accordingly.

The [OuteTTS Github project page](https://github.com/edwko/OuteTTS) documents these various options. That being said, here are some recommendations, based on my own tests up to this point...

**Nvidia (CUDA) cards:**

Install Pytorch with CUDA in the normal fashion: Uninstall the vanilla version (`pytorch uninstall torch torchvision torchaudio`) and then [install](https://pytorch.org/get-started/locally/) the CUDA-enabled version.

Prefer the *ExLllama2* backend if at all possible: (`backend=outetts.Backend.EXL2`). See the example config in `config_oute.py`. However, this requires successfully installing into the environment three extra things: the exllama2 library (`pip install exllamav2`); [Flash Attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features); and Triton (eg, `pip install triton-windows==3.2.0.post19`).

Alternatively, `Backend.HF` is also hardware accelerated but maybe 50% slower.

I couldn't get acceleration going using `Backend.LLAMACPP` but I'm not sure if that's just be me.

**Mac with Apple silicon**

Use `Backend.LLAMACPP`.

# Usage notes

The app saves its state between sessions, so you can interrupt the program at any time and resume later (important due to how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in  manageable chunks over time, allowing you to to use early chapter files before the full text is completed.

### Voice cloning

When prepping reference audio for doing the voice clone, it can be worthwhile to prepare more than one sound sample from a given source, and then testing each one out in turn on a short passage of the intended text, as the quality and characteristics of each voice clone from the same source can vary quite a bit. Oute recommends a voice sample of 15 seconds or less.

### Inference speed, expectations.

With Oute TTS, I get inference speeds of 80-85% of realtime using `Backend.EXL2` with a GeForce 3080Ti. And about 20% with an M1 MacBook Pro (using llama.cpp as the configured backend).

Chatterbox is considerably faster, about 130% of realtime using a GeForce 3080Ti.


# Updates

**2025-06-28** (many)

Generated audio segments now have silence trimmed off the ends, and in the concatenation step, stitched-together lines have pauses of varying lengths inserted at paragraph, sentence, and phrase boundaries, resulting in much improved prosody / flow.

Loudness normalization is now being applied correctly (on the final audio file instead of per audio segment) (d'oh)

Better detection of undesired repeating phrases (Oute especially)

Better detection and fix for spooky Chatterbox noises at the end of prompts

Short 1-2 word sentences now get grouped with adjacent sentences to minimize Chatterbox errors with short prompts

Streamlined handling of audio data throughout

Encoding audiobook files in AAC/M4A format no longer requires intermediate FLAC step

Streamlined some UI

Some improvements to the web player/reader