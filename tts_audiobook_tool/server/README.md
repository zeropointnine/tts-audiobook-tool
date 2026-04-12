# tts-server-tool

Lightweight REST server that utilizes the tts-audiobook-tool TTS engine to output text-to-speech audio through the default sound device.


## Usage

CD into the project directory

    cd path\to\tts-audiobook-tool

Activate your pre-existing tts-audiobook-tool virtual environment. Eg:

    venv-qwen3tts\Scripts\activate.bat

Run the server (the `host` and `port` arguments are optional, and default to 127.0.0.1 and 5001)

    python -m tts_audiobook_tool --server --host 127.0.0.1 --port 5001

As a reminder, to accept connections from other machines on your network, you can bind to all interfaces using `--host 0.0.0.0`

The server will apply the TTS model settings of your current project from the tts-audiobook-tool app (`Voice clone and model settings` submenu).

Once running, you can test the API in the browser by visiting 

    http://localhost:5001/tester.html

## API

### POST /prompt

Enqueues a text prompt for TTS inference and playback.

**Request body (JSON):**

| Field                   | Type    | Default | Description |
|-------------------------|---------|---------|-------------|
| `prompt`                | string  | —       | The text to synthesize. |
| `should_segment`        | boolean | `false` | If `true`, the prompt is split into multiple segments using the loaded project settings' text segmentation settings, and each segment is enqueued separately.<br><br>Setting this to true allows for inputting (potentially very) large blocks of text, with audio output which should be functionally identical to rendering the same text using the main tts-audiobook-tool "Realtime mode". |
| `eager_first_segment`  | boolean | `false` | Only relevant when `should_segment` is `true`. If `true` and the audio buffer is empty or near empty (< 1.0s), the first phrase of the first segment is enqueued immediately as its own item so playback can start sooner, with the remainder of that segment enqueued after. |

**Response (JSON):**

| Field                  | Type     | Description |
|------------------------|----------|-------------|
| `input`                | string   | The input prompt from the request body. |
| `prompts`              | string[] | The prompt(s) that were enqueued. |
| `current_queue_length` | number   | Number of prompts now waiting in the queue. |

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
| `inferencing` | string | The prompt currently being processed by the TTS model, or `""` if idle. |
| `playing` | string | The prompt whose audio is currently playing, or `""` if silent. |
| `audio_buffer` | number | Seconds of audio remaining in the playback buffer. |
| `num_queued` | number | Number of prompts waiting in the queue. |

### POST /clear

Clears the prompt queue and audio playback buffer immediately.

**Request body:** none

**Response:** `{}`