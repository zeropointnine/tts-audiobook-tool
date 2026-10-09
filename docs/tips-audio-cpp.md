# Running audio.cpp alongside tts-audiobook-tool

`tts-audiobook-tool` supports server-based TTS inference using audio.cpp. You could elect to run audio.cpp runs alongside tts-audiobook-tool on the same computer or on a separate machine.

For regular use, especially if you want to try several TTS models, audio.cpp is a convenient choice: one server installation can serve multiple models, without a separate Python inference environment for each one, and even without restarting or reconfiguring the server when properly configured.

## Getting started

Install and model-download instructions can be found in the [audio.cpp documentation](https://github.com/0xShug0/audio.cpp#readme). [Prebuilt binaries](https://github.com/0xShug0/audio.cpp/releases) are available for some system configurations. The [server documentation](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md) covers launch options and configuration. Follow those instructions for your operating system, GPU, and chosen models.

1. Install audio.cpp and download one or more models. Check the [supported-model table](<../README.md#description>) for the models currently supported through audio.cpp by `tts-audiobook-tool`; not every audio.cpp model is supported here.
2. Start `audiocpp_server` using audio.cpp's instructions. Its web UI is useful for downloading models and testing that they work before connecting the audiobook tool.
3. Set up and launch `tts-audiobook-tool` using the [virtual environment for remote TTS servers](<../README.md#virtual-environment-for-remote-tts-servers-sgl-omni--audiocpp>).
4. Under `Options` > `Remote TTS server URL`, enter the audio.cpp server's base URL, typically `http://127.0.0.1:8080`. Do not append `/v1` or an endpoint path. Keep the server running while using the app.
5. Create or open a project. If audio.cpp is configured to serve multiple models, use `Project` > `Switch TTS model` to choose among them.

## Using multiple models: configure server.json

**If you plan to use more than one TTS model, it is worth setting up a server config.** Declare each model in a `server.json` file and launch audio.cpp with `--config server.json`. This lets `tts-audiobook-tool` discover the configured models and request the selected one directly, without having to manually change models in the audio.cpp web UI or restart the server each time.

For example, a two-model config using OmniVoice and Echo-TTS might look like this:

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

Adapt the example as needed. See audio.cpp's [server configuration guide](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md#config) for the config format and available options.

Launch the server with:

```bash
audiocpp_server --config server.json
```

After adding or changing config entries, restart the server to apply them.


## Keep only one TTS model loaded at a time

**For most single-GPU setups, use `"max_loaded_models": 1`.** You can set this directly at the top level of the JSON config, as shown above or, alternatively, as a command-line option.

The limit applies only to models inside audio.cpp. `tts-audiobook-tool` itself may also need GPU memory for Whisper validation and other supporting models, so leave some headroom even with just one TTS model loaded. See the [VRAM considerations](<../README.md#vram-considerations>) for ways to reduce the app's memory use.
