# tts-audiobook-tool

Generative-AI audiobook creation tool focused on high-quality output which supports a growing list of text-to-speech models:
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [IndexTTS2](https://github.com/index-tts/index-tts)
- [VibeVoice](https://github.com/microsoft/VibeVoice)
- [Chatterbox (Multilingual, Turbo)](https://github.com/resemble-ai/chatterbox)
- [Fish Speech (S2-Pro, S1-mini)](https://github.com/fishaudio/fish-speech)
- [Higgs Audio V2](https://github.com/boson-ai/higgs-audio)
- [GLM-TTS](https://github.com/zai-org/GLM-TTS)
- [MiraTTS](https://github.com/ysharma3501/MiraTTS)
- [Oute TTS](https://github.com/edwko/OuteTTS)

The app employs several techniques to manage and improve on the inherently nondeterministic output of text-to-speech models. Eg:

- Intelligent segmentation of long-form text at paragraph/sentence/phrase boundaries
- Automatic detection and correction of inference errors using speech-to-text verification, with retry logic that keeps the most accurate take
- Silence trimming and semantically-aware pause modulation at segment boundaries to improve prosody
- Industry-standard loudness normalization

The app uses a plain-text interface in the console.


### How to create an audiobook (quick summary)

1. Assign a working project directory.
2. Select a short reference audio clip for the voice clone, and adjust related model parameters if desired.
3. Select the source text, and optionally define file split points.
4. Generate
5. Concatenate the generated audio segments to create the finished audiobook file/s.
6. Use the optional web player to play and read your audiobook.

<video src="https://github.com/user-attachments/assets/5516ce38-ef61-4126-91f5-71861e63eba5" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-full" style="max-height:640px;" autoplay="autoplay" loop="loop">
</video>

### Browser player

The included web player displays text highlighted in sync with the generated audio — similar to the Kindle Whispersync or Google Play Books read-along experience. This works because the app embeds the text and word-level timing data directly into the metadata of the FLAC and M4B files it produces.

The player is a static HTML page — no web server or install required. Open `browser_player/index.html` directly in any browser, or use the [hosted version](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/).

**Sample outputs**

All examples use the same source text and the same 15-second voice clone sample using each model's default settings, unless noted:

- [Qwen3-TTS-1.7B-Base](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-qwen3-12hz-1.7b-base.abr.m4a)
- [IndexTTS2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-indextts2.abr.m4a)
- [IndexTTS2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-indextts2-plus-emo.abr.m4a) (with added emotional guidance voice sample)
- [VibeVoice 1.5B](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-vibevoice-1.5b.abr.m4a)
- [VibeVoice 1.5B](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-vibevoice-1.5b-lora-klett.abr.m4a) (LoRA example) ([LoRA link](https://huggingface.co/vibevoice-community/klett))
- [VibeVoice 7B](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-vibevoice-7b.abr.m4a) 
- [Chatterbox](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-chatterbox.abr.m4a)
- [Fish S2-Pro](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-s2-pro.abr.m4a)
- [Fish S1-mini](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-s1-mini.abr.m4a)
- [Higgs Audio V2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-higgs.abr.m4a)
- [Higgs Audio V2](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-higgs-different-voice.abr.m4a) (using a different voice, at high temperature)
- [GLM-TTS](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-glm.abr.m4a)
- [MiraTTS](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-mira.abr.m4a)
- [Oute](https://zeropointnine.github.io/tts-audiobook-tool/browser_player/?url=https://zeropointnine.github.io/tts-audiobook-tool-sample-output/waves-oute.abr.m4a)

### Enhance existing audiobooks

This feature extends the browser player to work with professionally produced audiobooks — not just ones generated with this tool. It uses speech-to-text to align the existing audio with the corresponding book text, then embeds the same timing metadata the app normally produces during generation.

Select `Options` > `Enhance existing audiobook`, and choose your source audiobook file (M4A or M4B) and the corresponding book text. This feature is experimental.

### tts-server-tool

The project also includes an optional stand-alone REST service that runs the app's full configured TTS pipeline — your chosen model, voice clone, and project settings — and plays generated audio through the default sound device or serves it as an HTTP audio stream. This is aimed at developers who want to integrate the app's TTS capabilities into their own tools and workflows — for example, triggering spoken audio from a script, a home automation system, a chatbot, or any other application that can issue HTTP requests. See [server readme](tts_audiobook_tool/server/server-README.md) for setup and API details.

# Installation

### Step 1

First, [ffmpeg](https://ffmpeg.org/download.html) must be in your system path.

### Step 2

Clone the repository and cd into it:

    git clone https://github.com/zeropointnine/tts-audiobook-tool
    cd tts-audiobook-tool

### Step 3 

A separate virtual environment must be created for each model you want to use. Perform the operations as described in one or more of the sections below, and then return here. 

### Step 4 (Windows)

To enable torch CUDA acceleration on Windows, run the following commands (The project uses the same version of torch for each TTS model's virtual environments unless otherwise noted - v2.8.0/cu128). This extra step is not required when using Linux.

    pip uninstall -y torch torchaudio
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

### Step 5

Run the app by entering:

    python -m tts_audiobook_tool

Note that any settings and features that are specific to a given TTS model will be enabled automatically based on which virtual environment has been enabled.


## Virtual environment for Qwen3-TTS 

Initialize a **Python v3.12** virtual environment named `venv-qwen3tts`. For example:

    path\to\python3.11\python.exe -m venv venv-qwen3tts

Activate the virtual environment:

    venv-qwen3tts\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-qwen3tts.txt

Install [Flash Attention](#installing-flash-attention) if using CUDA (optional but recommended).

## Virtual environment for IndexTTS2

Initialize a **Python v3.11** virtual environment named `venv-indextts2`. For example:

    path\to\python3.11\python.exe -m venv venv-indextts2

Activate the virtual environment:

    venv-indextts2\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-indextts2.txt


## Virtual environment for VibeVoice

Initialize a **Python v3.11** virtual environment named `venv-vibevoice`. For example:

    path\to\python3.11\python.exe -m venv venv-vibevoice

Activate the virtual environment:

    venv-vibevoice\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-vibevoice.txt

Install [Flash Attention](#installing-flash-attention) if using CUDA (optional but recommended).

Note that because Microsoft famously removed the source code from their github repository, we pull from an archived, third-party fork, [vibevoice-community](https://github.com/vibevoice-community/VibeVoice).

## Virtual environment for Chatterbox:

Initialize a **Python v3.11** virtual environment named "venv-chatterbox". For example:

    path\to\python3.11\python.exe -m venv venv-chatterbox

Activate the virtual environment:

    venv-chatterbox\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-chatterbox.txt

Windows CUDA support (unlike the other models, we must use torch 2.6/cu124)

    pip uninstall -y torch torchaudio
    pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

## Virtual environment for Fish S2-Pro:

> **ℹ️ Note!**
> For CUDA acceleration, 24 GB VRAM is required (yes really)

Initialize a **Python v3.12** virtual environment named "venv-fish-s2". For example:

    path\to\python3.12\python.exe -m venv venv-fish-s2

Activate the virtual environment:

    venv-fish-s2\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-fish-s2.txt

Authenticate the model on HuggingFace:

1. Accept the license terms on the [HuggingFace Fish model page](https://huggingface.co/fishaudio/s2-pro).

2. Authenticate locally using your [access token](https://huggingface.co/settings/tokens) by running `hf auth login`
 

## Virtual environment for Fish S1-mini:

Initialize a **Python v3.12** virtual environment named "venv-fish-s1". For example:

    path\to\python3.12\python.exe -m venv venv-fish-s1

Activate the virtual environment:

    venv-fish-s1\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-fish-s1.txt

Authenticate the model on HuggingFace:

1. Accept the license terms on the [HuggingFace Fish model page](https://huggingface.co/fishaudio/s1-mini).

2. Authenticate locally using your [access token](https://huggingface.co/settings/tokens) by running `hf auth login`
 

## Virtual environment for Higgs Audio V2:

> **ℹ️ Note!**
> For CUDA acceleration, 24 GB VRAM is recommended

On Linux and macOS, portaudio must be installed (eg, on Mac, `brew install portaudio`)

Initialize a **Python v3.12** virtual environment named `venv-higgs`. For example:

    path\to\python3.12\python.exe -m venv venv-higgs

Activate the virtual environment:

    venv-higgs\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-higgs.txt

Note that the above `requirements` file draws from a personal fork of the `higgs-audio` library due to the fact that the higgs repo is missing `__init__.py` files required for module use.


## Virtual environment for GLM-TTS

> **ℹ️ Requires CUDA** due to hardcoded cuda-related torch operations throughout the glm inference code. 

### Linux

Initialize a **Python v3.11** virtual environment named `venv-glm`. For example:

    pyenv local 3.11
    python -m venv venv-glm

Activate the virtual environment:

    source venv-glm/bin/activate

Install dependencies:

    pip install -r requirements-glm.txt

### Windows

Requires setting up a "hybrid conda environment" due to a transitive dependency which is difficult to install on Windows using only pip.

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) if not already.

Initialize the conda environment. For example:

    conda create --prefix "c:\path\to\conda-glm" python=3.11 -y

Activate the conda environment:

    conda activate "c:\path\to\conda-glm"

Install this dependency using conda:

    conda install pynini==2.1.6 -c conda-forge

Install the rest of the project dependencies using pip like normal:

    pip install -r requirements-glm.txt

Note that we pull from [a fork of glm-tts](https://github.com/zeropointnine/glm-tts-packaged) that has been refactored for use as an installable package.


## Virtual environment for MiraTTS

> **ℹ️ Requires CUDA** 

Initialize a **Python v3.12** virtual environment named `venv-mira`. For example:

    path\to\python3.12\python.exe -m venv venv-mira

Activate the virtual environment:

    venv-mira\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-mira.txt


## Virtual environment for Oute TTS:

Initialize a **Python v3.12** virtual environment named "venv-oute". For example:

    path\to\python3.12\python.exe -m venv venv-oute

Activate the virtual environment:

    venv-oute\Scripts\activate.bat

Install dependencies:

    pip install -r requirements-oute.txt


### Oute TTS model configuration

Running the app optimally with Oute TTS requires extra steps due to the way the model supports multiple backends, model sizes and quantizations. You will need to review and hand-edit the source file **`config_oute.py`** accordingly.

The [OuteTTS Github project page](https://github.com/edwko/OuteTTS) documents these various options. But here are some recommendations based on my own testing...

**Nvidia cards (CUDA):**

Prefer the *ExLllama2* backend if at all possible: `backend=outetts.Backend.EXL2` (See the example Oute config in `config_oute.py`). However, this requires successfully installing into the environment three extra things:
- exllama2 library (`pip install exllamav2`)
- [Flash Attention](#installing-flash-attention)

Alternatively, `Backend.HF` is also hardware accelerated but considerably slower. Flash Attention is optional in this case.

**Mac with Apple silicon:**

Use `Backend.LLAMACPP`.

### Installing Flash Attention

If instructed to do so, perform this step after successfully running `pip install requirements-[model].txt` without errors.

On Linux, enter:

    pip install flash-attn==2.8.3 --no-build-isolation

On Windows, download a pre-compiled wheel from a trustworthy source (for example, [mjun0812](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.4.10)). The filename should look something like this, depending on Python version: 

- flash_attn-2.7.4+cu128torch2.8-cp311-cp311-win_amd64.whl
- flash_attn-2.7.4+cu128torch2.8-cp312-cp312-win_amd64.whl


# Usage notes

The app saves its state between sessions, so you can interrupt the program at any time and resume later (important due to how long generating a full-length novel can take).

Additionally, setting chapter cut points can be useful to generate and export a long work in manageable chunks over time, allowing you to to use early chapter files before the full text is completed.

Note too that it's possible to utilize different voices and even different models over the course of generating the audio segments for a given project.

### Voice cloning

When prepping reference audio for voice cloning, it's worthwhile to prepare three or so different sound samples from a given source (not just one), and then test each one out in turn on a short passage of the intended text, as the quality, characteristics, and word error rate resulting from each sample from the same source can vary quite a bit.

### VRAM considerations

The app ideally wants to use ~2-4 GB extra VRAM for the Whisper model, which needs to runs concurrently with the currently active TTS model to validate its output. If you are getting out of memory errors, try one of the following:

- Choose the Whisper turbo model (saves about 1 GB of VRAM) (`Options` > `Whisper model` > `large-v3-turbo`)
- Force Whisper to use system memory instead of running on the GPU (runs much slower, ofc) (`Options` > `Whisper device` > `CPU`)
- Disable Whisper altogether (last resort) (`Options` > `Whisper model` > `Disabled`)


### Model-specific features supported by the app 

Voice cloning is a first-class feature, supported for all models. 

**Qwen3-TTS**

- CustomVoice and VoiceDesign model variants
- Batch processing
- Temperature, seed

**IndexTTS2**

- Emotion voice sample
- Emotion vector 
- Emotion alpha
- FP16 / FP8
- Temperature, top_p, top_k, seed

**VibeVoice**

- Alternate VibeVoice models (eg, VibeVoice-7B or custom finetunes)
- LoRAs
- Batch processing
- Music detection/rejection
- CFG, steps, seed

**Chatterbox**

- Chatterbox Multilingual and Chatterbox-Turbo model variants
- Exaggeration
- CFG
- Temperature, top_p, top_k, repetition_penalty, seed

**Fish S2-Pro**

- Emotion tags
- Torch compile 
- Temperature, top_p, top_k, seed

**Fish S1-mini**

- Torch compile 
- Temperature, top_p, repetition penalty, seed

**Higgs Audio V2**

- Temperature, top_p, top_k, seed

**GLM-TTS**

- Samplerate (24khz or 32 khz)
- Seed

**MiraTTS**

- Batch processing
- Temperature, top_p, top_k, repetition_penalty, seed

**Oute TTS**

- Temperature


### Inference speeds, expectations

Listed below are my anecdotal TTS inference speeds. The app adopts each respective model's reference inference implementation logic as much as possible. Note how CUDA inference speeds on Linux are usually significantly faster than on Windows.

| TTS Model               | Setup                | Speed           | Notes |
| ----------------------- | -------------------- | --------------- | ----- |
| Qwen3-TTS 1.6B          | GTX 3080 Ti, Linux   | 300% realtime   | batch size=5
|                         | GTX 3080 Ti, Linux   | 100% realtime   | batch size=1
| IndexTTS2               | GTX 4090, Windows    | ~150% realtime  | 
|                         | GTX 3080 Ti, Windows | ~90% realtime   |
|                         | Macbook Pro M1 (MPS) | ~20% realtime   |
| VibeVoice-Large 7B      | GTX 4090, Windows    | 600%+ realtime  | batch size=10, default steps
| VibeVoice 1.5B          | GTX 3080 Ti, Linux   | 1000% realtime  | batch size=10, default steps
|                         | GTX 3080 Ti, Linux   | 200%+ realtime  | batch size=1, default steps
|                         | GTX 3080 Ti, Windows | ~120% realtime  | batch size=1, default steps
|                         | Macbook Pro M1       | ~40% realtime   |
| Chatterbox Turbo        | GTX 3080 Ti, Linux   | 500%+ realtime  | 
| Chatterbox Multilingual | GTX 4090, Windows    | ~190% realtime  | 
|                         | GTX 3080 Ti, Windows | ~130% realtime  | 
|                         | Macbook Pro M1 (MPS) | 20-35% realtime |
| Fish S2-Pro             | GTX 4090, Windows    | 150% realtime   | 
| Fish S1-mini            | GTX 3080 Ti, Windows | 500%+ realtime  | 
|                         | Macbook Pro M1 (MPS) | ~15% realtime   | 
| Higgs V2 3B             | GTX 4090, Windows    | ~200% realtime  | inference speed varies with voice sample duration
| GLM-TTS                 | GTX 3080 Ti, Linux   | 200%+ realtime  | 
| MiraTTS                 | GTX 3080 Ti, Linux   | 3000% realtime (yes really) | batch size=10
|                         | GTX 3080 Ti, Linux   | 800% realtime   | batch size=1
| Oute                    | GTX 3080 Ti, Windows | ~90% realtime   | using `outetts.Backend.EXL2`


# Update highlights

**2026-04-13**

- Added side feature: **tts-server-tool**. View [server readme](server/server-README.md). 

**2026-04-11**

- Added configurable **top-p**, **top-k**, and **repetition penalty** hyperparameters to models where applicable given their architecture.

**2026-04-07**

- Added support for **Fish S2-Pro**

- Updated torch, ctranslate2, and flash-attention dependencies for all virtual environments. 

- Fish S1-mini and S2-Pro - Added torch compile toggle

**2025-01-26**

- Added support for **Qwen3-TTS** (Base, CustomVoice, and VoiceDesign models)

**2025-01-23**

- Added new text segmentation method **"multiple sentences"** (`Text` > `Text segmentation strategy`). Probably the ideal way to chunk text in combination with `max_words_per_segment = 80` when the TTS model can handle long prompts with high accuracy and without speeding up.

**2025-01-21**

- Added support for **VibeVoice LoRAs** [(recommended trainer)](https://github.com/voicepowered-ai/VibeVoice-finetuning)

**2025-01-18, circa**

- **Batching optimization** improves net TTS inference speed by up to 25% (VibeVoice and Mira).

- **VibeVoice batch support** (drastically speeds up inference). 

- **VibeVoice seed support**

- **Music detection** - Validator will now reject audio with music hallucinations (VibeVoice)

- Added submenu: `Concat` > `Open audiobook file in the player app`

**2025-01-12**

- Improved per-phrase text segmentation (ie, `Concat` > `Subdivide into phrases`). For pre-existing projects, requires reimporting the source text.

- Prevent some TTS models from mangling the output of prompts starting with **all-caps phrases** (relevant for MiraTTS in particular, plus some others).

**2025-01-10**

- "Chapter dividers" can now be used for creating **M4B chapters** and player bookmarks (`Concatenate` > `Chapter dividers` > `Mode`)

- Note, app now saves AAC files using `.m4b` suffix instead of `.m4a`.

**2025-01-05**

- Added option to **add chapter dividers using regular expressions** (thanks @AntitrustEnthusiast).

**2025-01-04**

- Added support for **Chatterbox-Turbo**.

- Improved logic for trimming audio generations with spurious words and noises at the beginning or end (important for Chatterbox).

**2025-12-31**

- Added support for **MiraTTS**, including batch mode. The model is exceptionally fast when running in batch mode.

- Added option: `Generate` > `Transcript validation strictness`

- Option to control seed value now available for: Chatterbox, Fish, GLM

**2025-12-23**

- Added option: **`Project`** > **`Word substitutions`**. This allows you to replace words in the prompt that the TTS model may mispronounce. Requires some experimentation, as you might imagine. 

- Relatedly, added: `Project` > `Word substitutions` > `Inspect project text for uncommon words`.

- Also added option: `Generate` > `Transcript validation max retries`

**2025-12-21**

- Simplified and much-improved **speech-to-text validation logic**, which is now able to calculate the number of word errors for an audio generation (rather than simply adjudging "pass" vs "fail" using a patchwork of bespoke functions).

- When a prompt is re-generated due to excessive word errors, the program is able to cherrypick the generation with the least number of errors.

**2025-12-16**

- Added support for **GLM-TTS**.

- Added option: **`Realtime audio generation` > `Save output`**.

**2025-12-13**

- Added option **`Concatenate` > `Subdivide into phrases`**. This allows the player app to highlight text on a "per-phrase" basis by leveraging the STT word-level timing data (Note, on pre-existing projects, all audio would first need be re-generated).

**2025-12-09**

- Added option to **clone project** (load an existing project > make new project > "Do you want to carry over the current project's settings?" > "Yes")

- Added option to force use of **CPU torch device** even when CUDA or MPS is available.

**2025-12-07**

- Added **max text segment word length** option, allowing for up to 80 words per text segment prompt. Can be useful with VibeVoice and Fish (other models are best left at the default of 40 words).

- **Resolved Linux CUDA crashes** related to faster-whisper library by changing Linux torch requirements. If you were experiencing this problem, get latest and update your virtual environment (`pip install -r requirements-[modelname].txt`).

**2025-12-06**

- **Updated Chatterbox** requirement to v0.1.4, which is multilanguage capable, and **added language code** project setting. Thanks to @JuMGameN, co-author for this update.

- For existing users, the Chatterbox venv must be recreated using Python v3.11.

**2025-11-11**

- Added option to keep the **Whisper** model on the CPU.

**2025-11-05**

- Player/reader: Added **bookmarks** feature, ability to pin the audio player widget in desktop mode, plus.

**2025-10-24**

- Added option to **disable transcription validation** (`Options` > `Whisper model type` > `Disabled`). Doing so is only advisable while VRAM is at a premium or when using reliably accurate and stable TTS models (eg, IndexTTS2; definitely not VoiceVoice 1.5B or Chatterbox).

**2025-10-23**

- Added option to insert a "page turn" **sound effect at section breaks** (2+ blank lines in the text). I use this or text files converted from epub files using Calibre, which tends to insert two or more blank lines at chapter breaks, etc.

**2025-10-03**

- Added support for **IndexTTS2**, including secondary voice reference sample guiding emotion and "emotion vectors".

- Added option to **specify Whisper model** (large-v3 or large-v3-turbo)

**2025-09-15**

- Added support for **VibeVoice alternative models** (ie, large/7B model)

**2025-09-12**

- Added support for **VibeVoice**.

**2025-08-10**

- Migrated from openai-whisper to **faster-whisper** (faster, less memory, equivalent accuracy).

**2025-08-06**

- Added support for **Higgs Audio V2** (3B base model).

**2025-07-18**

- Added support for **Fish OpenAudio S1-mini**

- Logic to detect dropped phrases at end of generated audio segments (common occurrence with the Fish model)

- Added utility to transcode and concatenate directory of MP3 chapter files to M4A (meant for use with the "Enhance existing audiobook" tool)

**2025-07-02**

- New feature: **Real-time playback** (`Options > Real-time playback`)

    This serves as a quicker and more "casual" alternative to the regular `Generate audio` UI flow, and allows for more-or-less immediate and uninterrupted audio playback (contingent on system performance, ofc). It employs the same quality control measures except for loudness normalization, and does not save its output.

**2025-06-28** (many)

- Generated audio segments now have silence trimmed off the ends, and in the concatenation step, stitched-together lines have pauses of varying lengths inserted at paragraph, sentence, and phrase boundaries, resulting in much improved prosody / flow.

- Loudness normalization is now being applied correctly (on the final audio file instead of per audio segment)

- Better detection of undesired repeating phrases (Oute especially)

- Better detection and fix for spooky Chatterbox noises at the end of prompts

- Short 1-2 word sentences now get grouped with adjacent sentences to mitigate Chatterbox and Oute's issues with short prompts

- Streamlined handling of audio data throughout

- Encoding audiobook files in AAC/M4A format no longer requires intermediate FLAC step

- Streamlined some UI

- Some improvements to the web player/reader
