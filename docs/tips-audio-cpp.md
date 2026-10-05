# Running audio.cpp alongside tts-audiobook-tool

`tts-audiobook-tool` supports server-based TTS inference using audio.cpp. In a typical setup, audio.cpp runs alongside tts-audiobook-tool on the same computer, as a separate process; it can also run on another machine.

For regular use, especially if you want to try several TTS models, audio.cpp is a convenient choice: one server installation can serve multiple models, without a separate Python inference environment for each one, and even without restarting the server if configured correctly.

## Getting started

Install and model-download instructions can be found in the [audio.cpp documentation](https://github.com/0xShug0/audio.cpp#readme). [Prebuilt binaries](https://github.com/0xShug0/audio.cpp/releases) are available, and the [server documentation](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md) covers launch options and configuration. Follow those instructions for your operating system, GPU, and chosen models; this page focuses on using the server with `tts-audiobook-tool`.

1. Install audio.cpp and download one or more models. Check the [supported-model table](<../README.md#description>) for the models currently supported through audio.cpp by `tts-audiobook-tool`; not every audio.cpp model is supported here.
2. Start `audiocpp_server` using audio.cpp's instructions. Its web UI is useful for downloading models and testing that they work before connecting the audiobook tool.
3. Set up and launch `tts-audiobook-tool` using the [virtual environment for remote TTS servers](<../README.md#virtual-environment-for-remote-tts-servers-sgl-omni--audiocpp>).
4. Under `Options` > `Remote TTS server URL`, enter the audio.cpp server's base URL, typically `http://127.0.0.1:8080`. Do not append `/v1` or an endpoint path. Keep the server running while using the app.
5. Create or open a project, then use `Project` > `Switch TTS model` to choose among the available models (when only one is available, the item is labeled `TTS model`). If you change what the server offers, use `Options` > `Refresh remote TTS server` to update the list.

When both programs run on the same computer, keeping the server bound to `127.0.0.1` is sufficient. If the server runs on another machine, use that machine's address and follow audio.cpp's guidance for network access; avoid exposing it directly to the public internet.

## Using multiple models: configure server.json

**If you plan to use more than one TTS model, it is worth setting up a server config.** Declare each model in a `server.json` file and launch audio.cpp with `--config server.json`. This lets `tts-audiobook-tool` discover the configured models and request the selected one directly, without having to manually change models in the audio.cpp web UI or restart the server each time.

For example, a two-model config using OmniVoice and Echo-TTS could look like this:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "lazy_load": true,
  "max_loaded_models": 1,
  "models": [
    {
      "id": "omnivoice",
      "family": "omnivoice",
      "path": "/path/to/models/OmniVoice",
      "task": "tts",
      "mode": "offline"
    },
    {
      "id": "echo-tts",
      "family": "echo_tts",
      "path": "/path/to/models/echo-tts",
      "task": "clon",
      "mode": "offline"
    }
  ]
}
```

Adapt the example to your downloaded models and hardware. See audio.cpp's [server configuration guide](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md#config) for the config format and available options.

Launch the server with:

```bash
audiocpp_server --config server.json
```

After adding or changing config entries, restart the server to apply them.

## MOSS-TTS Delay and Local

The model selector uses the same names as SGL-Omni: **MOSS-TTS Delay** and **MOSS-TTS Local**. These correspond to different audio.cpp families:

| App name | audio.cpp family | Output | Temperature / top-p / top-k defaults |
|---|---|---|---|
| MOSS-TTS Delay | `moss_tts_v15` | 24 kHz mono | `1.5 / 0.6 / 50` |
| MOSS-TTS Local | `moss_tts_local` | 48 kHz | `1.7 / 0.8 / 25` |

Delay is audio.cpp's experimental community implementation of the 8B v1.5 model, with English and Chinese advertised. “Local” identifies the Local Transformer architecture, not whether the server runs on the same computer.

For example, use these entries in your server config's `models` array:

```json
[
  {
    "id": "my-moss-delay",
    "family": "moss_tts_v15",
    "path": "/path/to/models/MOSS-TTS-v1.5-GGUF",
    "task": "tts",
    "mode": "offline"
  },
  {
    "id": "my-moss-local",
    "family": "moss_tts_local",
    "path": "/path/to/models/MOSS-TTS-Local-v1.5-GGUF",
    "task": "tts",
    "mode": "offline"
  }
]
```

The IDs can be arbitrary: discovery matches the declared family, task, and mode. Both models also accept `"task": "clon"`. **Configure one entry per variant**, not separate `tts` and `clon` entries for the same variant; otherwise the app cannot bind an unambiguous server entry.

- Both models can generate without a voice sample, or clone from an optional sample even under task `tts`. Without a sample, the generated voice may vary between segments.
- Neither implementation consumes a reference transcript. Unlike the existing local/SGL-Omni MOSS voice-import workflow, these audio.cpp variants do not request transcription or send reference text.
- Editable controls are temperature, top-p, top-k, and seed. The project's language is automatically mapped to a full MOSS language name when recognized.
- Community Delay parses seeds as signed 32-bit integers: its catalog `max_random_seed = 2147483647` keeps random seeds in range, like Echo. Fixed seeds are not clamped by this policy; use `0`–`2147483647` to avoid the server's `stoi argument out of range` error. Local's seed range is unchanged.
- Other model parameters stay at audio.cpp/server defaults and are not app controls: this includes repetition penalty and Local's default ceiling of 4096 audio frames per chunk. The app leaves both server-side text chunkers enabled with their native defaults.
- Requests are offline and sequential, without streaming, batching, concurrent requests, or rolling continuation. Local retains the app's music detection and trailing token-noise trimming behavior.
- Local caches prepared reference codes in its server session, with one slot by default; the server-side `moss_tts_local.reference_cache_slots` session option controls this. Community Delay encodes the reference again for each request. The app does not manage either cache.

Voice samples and generation settings belong to each backend variant independently; switching from local or SGL-Omni does not copy them into audio.cpp.

## Keep one TTS model loaded at a time

**For most single-GPU setups, start with `"max_loaded_models": 1`.** You can set this directly at the top level of the JSON config, as shown above (or, alternatively, as a command-line option).

The limit applies only to models inside audio.cpp. `tts-audiobook-tool` may also need GPU memory for Whisper validation and other processing, so leave some headroom even with just one TTS model loaded. See the [VRAM considerations](<../README.md#vram-considerations>) for ways to reduce the app's memory use.
