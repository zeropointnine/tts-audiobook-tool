# tts-server-tool

The project now includes a lightweight REST server that utilizes the tts-audiobook-tool TTS engine and outputs text-to-speech audio through either the default sound device or an HTTP audio stream.


## How to run

CD into the project directory

    cd path\to\tts-audiobook-tool

Activate one of your pre-existing tts-audiobook-tool virtual environments. Eg:

    venv-qwen3tts\Scripts\activate.bat

Run the server (the `host` and `port` arguments are optional, and default to 127.0.0.1 and 5001)

    python -m tts_audiobook_tool --server --host 127.0.0.1 --port 5001

As a reminder, you can accept connections from other machines on your local network by using `--host 0.0.0.0`

The server uses the TTS settings from your currently active tts-audiobook-tool project (and in particular, those found in the `Voice clone and model settings` submenu).


## Usage notes

Once running, you can test the API in the browser by visiting 

    http://localhost:5001/

By default, the server will output TTS audio through its default sound output device. 

Additional built-in pages:

- `GET /` serves the API demo page (`server/demos/api.html`)
- `GET /demos/api.html` serves the API demo directly
- `GET /demos/streaming-client.html` serves the browser audio-stream client demo
- `GET /demos/combination.html` serves the combined API + browser-streaming demo


# API

### POST /prompt

Enqueues a text prompt for TTS inference and playback.

**Request body (JSON):**

| Field                   | Type    | Default | Description |
|-------------------------|---------|---------|-------------|
| `prompt`                | string  | —       | The text to synthesize. |
| `should_segment`        | boolean | `true` | If `true`, the prompt is split into multiple segments using the loaded project settings' text segmentation settings, and each segment is enqueued separately.<br><br>Setting this to true allows for inputting (potentially very) large blocks of text, with audio output which should be functionally identical to rendering the same text using the main tts-audiobook-tool "Realtime mode". |
| `eager_first_segment`  | boolean | `false` | Only relevant when `should_segment` is `true`. If `true` and the audio buffer is empty or near empty (< 1.0s), the first phrase of the next queued segment may be generated immediately as its own playback unit so audio can start sooner, with the remainder queued after. This applies only to non-streaming queued items. |

**Response (JSON):**

| Field                  | Type     | Description |
|------------------------|----------|-------------|
| `input`                | string   | The input prompt from the request body. |
| `prompts`              | string[] | The prompt(s) that were enqueued. |
| `queue_length` | number   | Number of prompts now waiting in the queue. |

On error (e.g. empty prompt):

| Field | Type | Description |
|---|---|---|
| `error` | string | Error message. |

### GET /status

Returns the current state of the server.

**Response (JSON):**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"initializing"` while the TTS model is loading, `"ready"` when available. |
| `tts_model` | string | Human-readable name of the active TTS engine, from `Tts.get_type().value.ui["proper_name"]`. |
| `inferencing` | string | The prompt currently being processed by the TTS model, or `""` if idle. |
| `playing` | string | The text whose audio is currently being emitted by `AudioStream`, or `""` if silent. |
| `audio_buffer` | number | Seconds of audio remaining in the playback buffer. |
| `num_queued` | number | Number of prompts waiting in the queue. |
| `stream_clients` | number | Number of clients currently connected to the audio HTTP stream. |
| `local_audio` | boolean | Whether local audio playback (through the default sound device) is enabled. |
| `tts_streaming` | boolean | Whether model-side TTS streaming is currently enabled for newly queued prompts. |
| `tts_streaming_supported` | boolean | Whether the current TTS engine supports model-side streaming. |

`inferencing` and `playing` are intentionally text-based status fields, not stable segment IDs. This keeps the API accurate even when eager splitting dynamically changes the playback unit at runtime.

`tts_streaming` refers to model-side chunked generation during inference. It is separate from the HTTP audio stream endpoint at `/stream`, which mirrors whatever audio the server is currently playing.

At server startup, `tts_streaming` is initialized from the active model capability (`Tts.get_info().can_stream`). In other words, streaming-capable models start with `tts_streaming: true`, while non-streaming models start with `tts_streaming: false`.

### POST /clear

Clears the prompt queue and audio playback buffer immediately.

**Request body:** none

**Response:** `{}`

### POST /local-audio

Enables or disables local audio playback through the default sound device. This setting is stateful and persists until changed or the server is restarted.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Whether to enable local audio playback. |

**Response (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `local_audio` | boolean | The updated local audio enabled state. |

### POST /tts-streaming

Enables or disables model-side TTS streaming for newly queued prompts. This setting is stateful and persists until changed or the server is restarted.

If the current TTS engine does not support streaming, enabling this setting will not fail the request; instead the response will report `tts_streaming: false` and include a warning message.

Disabling streaming is always accepted and results in `tts_streaming: false`.

**Request body (JSON):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Whether to enable model-side TTS streaming for newly queued prompts. |

**Response (JSON):**

| Field | Type | Description |
|-------|------|-------------|
| `tts_streaming` | boolean | The updated effective TTS streaming state. |
| `tts_streaming_supported` | boolean | Whether the current TTS engine supports model-side streaming. |
| `warning` | string | Optional soft warning when streaming was requested but is unsupported by the current TTS engine. |

### Interaction with `POST /prompt`

The current effective `tts_streaming` value is captured when a prompt is enqueued, so changing the setting only affects prompts queued after the change.

`eager_first_segment` only applies to non-streaming segmented items. When model-side TTS streaming is enabled for a queued item, eager splitting is skipped because streaming already reduces first-audio latency.

### GET /stream

Streams the server's played audio over HTTP chunked transfer.

This endpoint is separate from model-side TTS streaming:

- `/tts-streaming` controls whether queued prompts use model-side streaming inference when supported
- `/stream` mirrors audio as it is played by the server, regardless of whether that audio originated from streaming or non-streaming TTS inference

For the full wire format and browser client details, see `README-streaming-client.md`.