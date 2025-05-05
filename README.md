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

This project relies on the oute-tts reference project, which allows for utilizing different quantizations of the Oute-1B model, different backends, etc.

To change these settings, hand-edit the Python file `model_config.py`. Please refer to the [OuteTTS project](https://github.com/edwko/OuteTTS)'s README and related documentation to find out what the possible values are.

For *Mac with Apple silicon*: Definitely start by replacing the default value, `backend=outetts.Backend.HF` with `backend=outetts.Backend.LLAMACPP`.

For CUDA systems, consider installing flash attention for an additional boost in inference speed.

# Usage notes

The app is designed to save its state between sessions, so you can interrupt the program at any time and resume later.

Inference can take some time, depending on the length of the source text. Audio rendering speed is about 30% of realtime on my development system with a Ryzen 7700x and 3080Ti, so be forewarned.

If you don't like the rendition of certain voice lines, you can selectively delete generated audio files from the working project directory, and press [A] to make it re-render.

# Todos

Command line?

