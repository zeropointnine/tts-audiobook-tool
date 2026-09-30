# Running SGL-Omni alongside tts-audiobook-tool

`tts-audiobook-tool` supports server-based TTS inference using SGL-Omni. In a typical setup, SGL-Omni runs alongside tts-audiobook-tool on the same computer, as a separate process; it can also run on another machine.

Install instructions for SGL-Omni can be found [here](https://sgl-project.github.io/sglang-omni/get_started/installation.html). Note that SGL-Omni is typically installed using Docker (especially on Windows).

You may also need to perform additional per-model install steps as described in their "cookbook" pages, as found here:

- [**AuK / AuK-Flash**](https://sgl-project.github.io/sglang-omni/cookbook/auk.html) (24GB VRAM recommended)
- [**Fish S2 Pro**](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html) (24GB VRAM recommended)
- [**Fun-CosyVoice3**](https://sgl-project.github.io/sglang-omni/cookbook/fun_cosyvoice3.html)
- [**Higgs Audio V3**](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html) (24GB VRAM recommended)
- [**MOSS-TTS v1.5**](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html) (>24GB VRAM required)
- [**Qwen3TTS-Base**](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html)
- [**ZONOS2**](https://sgl-project.github.io/sglang-omni/cookbook/zonos2.html) (16+GB VRAM recommended)

Below are typical launch commands for starting SGL-Omni server with each of the models that tts-audiobook-tool currently supports. Model downloads should occur automatically on first run.

- `sgl-omni serve --model-path tencent/AuK --port 8000`
- `sgl-omni serve --model-path tencent/AuK-Flash --port 8000`
- `sgl-omni serve --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000`
- `sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000`
- `sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000`
- `sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-v1.5 --port 8000`
- `sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-v1.5 --config examples/configs/moss_tts_24gb.yaml --port 8000` (for 24GB VRAM)
- `sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config examples/configs/qwen3_tts_1_7b.yaml --port 8000`
- `sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --port 8000`
- `sgl-omni serve --model-path Zyphra/zonos2 --port 8000`
