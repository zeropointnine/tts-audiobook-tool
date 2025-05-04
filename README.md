# Description

Audiobook-maker using Oute TTS 1.0 text-to-speech model. Employs a minimal interactive console interface.

TLDR usage instructions: Assign a working project directory, specify a 15-second reference audio file for the voice cloning, add text, start, and be prepared to wait. That's it.

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

This project relies on the [oute-tts](https://github.com/edwko/OuteTTS) reference project, which allows for utilizing different quantizations of the Oute-1B model, different backends, etc.

To change these settings, hand-edit the Python file `model_config.py`. Please refer to the [OuteTTS project](https://github.com/edwko/OuteTTS)'s README and related documentation to find out what the possible values are.

For optimal hardware acceleration, consider installing flash attention as well.

# Usage notes

The app is designed to save its state between sessions, so you can interrupt the program at any time.

Inference can take a while, depending on the source text. I get a rendering speed of about 30% on a system with a 7700x and 3080Ti.

If you don't like the rendition of certain voice lines, you can selectively delete generated audio files from the working project directory, and press [A] to re-render.
