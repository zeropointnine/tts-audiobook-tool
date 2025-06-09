# Description

This is an audiobook creation tool, designed to support multiple text-to-speech models. Currently, Oute TTS and Chatterbox TTS are supported. Both Oute and Chatterbox support "zero shot voice cloning", which is in my opinion what makes creating audiobooks using them so appealing.

The app attempts to detect and potentially correct audio inference hallucinations/errors, and can also apply dynamic loudness normalization to the generated audio. These are important quality control features due to the umm... *variable* nature of generative text-to-speech models.

**Player/reader feature**: The repo includes a browser-based player/reader which runs without the need for a web server. Simply open `browser_player\index.html` in the web browser or navigate to the mapped [github.io page](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-oute.mp4) ([Oute output example](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-oute.mp4) | [Chatterbox output example](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool/browser_player/waves-chatterbox.mp4)), and choose a FLAC or MP4 file you've generated using the app. As the audio file plays, the corresponding text is displayed and highlighted (somehwat akin to using the Kindle app + Audible). This is made possible by embedding text and timing information as custom metadata in the audio file.

**Bonus speech-to-text feature**: The application can also add its custom metadata to pre-existing (ie, professionally produced) audiobook files (M4B, etc) so that they can be played using the custom player/reader. Select `Options` > `Create tts-audiobook-tool metadata for a pre-existing audiobook`, and point it to your source audiobook file and corresponding book text.

**How to create an audiobook (quick summary):**
1. Assign a working project directory.
2. Select a short reference audio clip for the voice clone.
3. Select the source text.
3b. Optionally define file split points.
4. Start inferencing, and ... be prepared to wait.
5. Concatenate the generated audio segments to create the final FLAC or MP4 file/s.
6. Optionally use the aforementioned web player/reader to play/read your audiobook.


# Installation

Firstly, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

Clone the repository and cd into it:

    git clone tts-audiobook-tool
    cd tts-audiobook-tool

A separate virtual environment must be created for the Oute TTS model and for Chatterbox TTS. Perform the operations as described in one or both of the sections below. Model-specific options will be enabled automatically based on which virtual environment is active.

Once the desired virtual environment/s are set up and activated, run the app by entering:

    python -m tts_audiobook_tool


## Install the Oute TTS flavor of the application:

Initialize a Python v3.12 virtual environment named "venv-oute". For example:

    C:\path\to\v3.12\python.exe -m venv venv-oute

Activate the virtual environment:

    venv-oute\Scripts\activate.bat

Install library dependencies:

    pip install -r requirements-oute.txt

The CUDA-enabled version of pytorch must be installed in the typical manner: Uninstall the vanilla version of torch (`pip uninstall torch torchaudio torchvision`) and then install the correct version of CUDA torch for your setup ([link](https://pytorch.org/get-started/locally/))

### Oute TTS model configuration

The Oute TTS model requires some configuration, as these settings can greatly affect inference speed and also output quality. You'll want to review and hand-edit the Python file **`config_oute.py`**.

The [OuteTTS Github project page](https://github.com/edwko/OuteTTS) is the best place to find out more. That being said, here are some recommendations, based on my own tests up to this point...

**Nvidia (CUDA) cards:**

Install Pytorch with CUDA in the normal fashion: Uninstall the vanilla version (`pytorch uninstall torch torchvision torchaudio`) and then [install](https://pytorch.org/get-started/locally/) the CUDA-enabled version.

Prefer the *ExLllama2* backend if at all possible: (`backend=outetts.Backend.EXL2`). See the example config in `config_oute.py`. However, this requires manually installing the exllama2 library: `pip install exllamav2`, and also requires installing [flash attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

Alternatively, `Backend.HF` is also hardware accelerated but slower than ExLlama2.

I couldn't get acceleration going using `Backend.LLAMACPP` but that may just be me.

You may also need to install Triton(eg, `pip install triton-windows==3.2.0.post19`).

**Mac/Apple silicon**

Use `Backend.LLAMACPP`.

## Install the Chatterbox TTS flavor of the application:

Setting up the app to use Chatterbox TTS is less involved.

Initialize a Python virtual environment named "venv-chatterbox" (Using Python 3.13 should work). For example:

    C:\path\to\v3.12\python.exe -m venv venv-chatterbox

Activate the virtual environment:

    venv-chatterbox\Scripts\activate.bat

Install library dependencies:

    pip install -r requirements-chatterbox.txt

As above, the CUDA-enabled version of pytorch must be installed in the typical manner: Uninstall the vanilla version of torch (`pip uninstall torch torchaudio torchvision`) and then install the correct version of CUDA torch for your setup ([link](https://pytorch.org/get-started/locally/))


# Usage notes

The app is designed to save its state between sessions, so you can interrupt the program at any time and resume later (important due to how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in chunks over time, making it easier to use (listen to) early chapter files before the full text is completed.

### Voice cloning

When prepping reference audio for doing the voice clone, it can be worthwhile to prepare more than one sound sample from a given source, and then testing each one out in turn on a short passage of the intended text, as the quality and characteristics of each voice clone from the same source can vary quite a bit. Oute recommends 15 seconds or less.

### Inference speed

Inference can take some time, depending on the length of the source text. Using Oute TTS, I'm getting inference speeds of 80-85% of realtime using `Backend.EXL2` using a GeForce 3080Ti. And about 20% with an M1 MacBook Pro (using llama.cpp as the configured backend).


### Known issues

The player/reader still probably needs work re: small screens

Chatterbox inference may start to fail after about 1000 text segments have been generated, requiring an app restart. See logged [issue](https://github.com/resemble-ai/chatterbox/issues/126).
