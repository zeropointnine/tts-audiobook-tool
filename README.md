# Description

This is an "audiobook maker", using the [Oute TTS 1.0 1B](https://github.com/edwko/OuteTTS) text-to-speech model. It employs as minimal an interface and feature set as I felt like I could get away with while still keeping it useful and functional. The upside is that it should be very simple to pick up and use.

What's interesting about Oute TTS is that it does zero-shot voice cloning and that it outputs at 44khz, which is CD-quality audio, and quite unusual for a TTS model at the moment.

TLDR usage instructions: (1) Assign a working project directory, (2) select a 15-second reference audio file for the voice cloning, (3) paste in some text, (4) start inferencing, and (5) ... be prepared to wait (that's the catch, lol). And that's it.

# Install

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

To change these settings, hand-edit the Python file **`model_config.py`**, and refer to the  example configs therein.

Refer to the [OuteTTS inferface usage page](https://github.com/edwko/OuteTTS/blob/main/docs/interface_usage.md) for more.

That being said, here are some setup notes, based on my own tests up to this point...

### Nvidia (CUDA) cards:

Install Pytorch with CUDA in the normal fashion: Uninstall the vanilla version (`pytorch uninstall torch torchvision torchaudio`) and then [install](https://pytorch.org/get-started/locally/) the CUDA-enabled version based on your system configuration.

Prefer the ExLllama2 backend if at all possible: (`backend=outetts.Backend.EXL2`). See the example config in `model_config.py`.

Requires manually installing the exllama2 library: `pip install exllamav2`, and also requires installing flash attention.

Alternatively, `Backend.HF`, will also be hardware accelerated, just slower.

I couldn't get acceleration going using `Backend.LLAMACPP` but that may just be me.

You can also install Triton, which may speed up Whisper STT inferencing, but is pretty optional (eg, `pip install triton-windows==3.2.0.post19`).

### For Mac with Apple silicon:

Use `Backend.LLAMACPP`.

# Usage notes

The app is designed to save its state between sessions, so you can interrupt the program at any time and resume later.

Inference can take some time, depending on the length of the source text. I'm getting inference speeds of 80+% using `Backend.EXL2` with an undervolted GeForce 3080Ti. And about 20% with an M1 MacBook Pro.

The program can detect and fix most hallucinations by selecting `[D] Detect and fix audio generation errors`.

Additionally, if you don't like the rendition of certain voice lines, you can selectively delete generated audio files from the working project directory, and select `[G] Generate audio` to re-render them.

# Todos

Command line?