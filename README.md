# Description

This is an audiobook creation tool, using the [Oute TTS 1.0 1B](https://github.com/edwko/OuteTTS) text-to-speech model. It uses a rudimentary, no-frills console interface.

What's interesting about Oute TTS is that it does zero-shot voice cloning and that it outputs audio at 44khz. Because generative TTS models can hallucinate or repeat sentences and phrases, the app tries to mitigate this by using Whisper to compare the transcribed text of the generated audio against the source text.

TLDR instructions: (1) Assign a working project directory, (2) select a 15-second reference audio file for the voice cloning, (3) paste in some (or a lot of) text, (3b) optionally define chapter points, (4) start inferencing, and ... be prepared to wait (that's the catch, lol). Finally, (5) concatenate the generated audio segments to create the final FLAC file/s.

### Bonus:

I've added a browser-based player/reader which runs without the need for a web server. Simply open `browser_player\index.html` in the web browser or navigate to the mapped [github.io page](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves.flac), and choose a FLAC file you've generated using the app. As the audio file plays, the corresponding text is displayed and highlighted. This is made possible by embedding text and timing information as metadata in the FLAC file.

# Installation

Clone repository and CD into it:

    git clone tts-audiobook-tool
    cd tts-audiobook-tool

Initialize a Python v3.12 virtual environment. For example:

    C:\path\to\v3.12\python.exe -m venv venv

Activate the virtual environment:

    venv\Scripts\activate.bat

Install library dependencies:

    pip install -r requirements.txt

If CUDA is desired, uninstall pytorch, and then reinstall the CUDA version of pytorch in its place.

Finally, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

Run by entering:

    python -m tts_audiobook_tool

... but be sure to review the section below as well.

# Oute TTS model configuration

This project relies on the oute-tts reference project, which allows for the use of different backends, different quantizations, with or without flash attention, etc. These settings can greatly affect inference speed, and to a lesser extent, output quality.

To change these settings, hand-edit the Python file **`tts_config.py`**, and refer to the  example configs therein.

Refer to the [OuteTTS interface usage page](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md) for more.

That being said, here are some setup notes, based on my own tests up to this point...

### Nvidia (CUDA) cards:

Install Pytorch with CUDA in the normal fashion: Uninstall the vanilla version (`pytorch uninstall torch torchvision torchaudio`) and then [install](https://pytorch.org/get-started/locally/) the CUDA-enabled version.

Prefer the **ExLllama2** backend if at all possible: (`backend=outetts.Backend.EXL2`). See the example config in `tts_config.py`. This requires manually installing the exllama2 library: `pip install exllamav2`, and also requires installing [flash attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

Alternatively, `Backend.HF` is also hardware accelerated but a good deal slower than ExLlama2.

I couldn't get acceleration going using `Backend.LLAMACPP` but that may just be me.

You can also install Triton, which may speed up Whisper STT inferencing, but is pretty optional (eg, `pip install triton-windows==3.2.0.post19`).

### For Mac with Apple silicon:

Use `Backend.LLAMACPP`.

# Usage notes

The app is designed to save its state between sessions, so you can interrupt the program at any time and resume later (which can be almost a necessity considering how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in chunks over time, making it easier to use (listen to) early chapter files before the full text is completed.

### Inference speed

Inference can take some time, depending on the length of the source text. I'm getting inference speeds of 80-85% of realtime using `Backend.EXL2` using a GeForce 3080Ti. And about 20% with an M1 MacBook Pro (using llama.cpp as the configured backend).

### Voice cloning

When prepping reference audio for doing the voice clone, it can be worthwhile to prepare more than one sound sample from a given source, and then testing each one out in turn on a short passage of the intended text, as the quality and characteristics of each voice clone from the same source can vary quite a bit.

# Todos

Support for AAC or Opus

Support for other TTS models? (namely, Orpheus)
