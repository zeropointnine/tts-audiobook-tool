# TTS Server Design

Design document for `tts_audiobook_tool/server/server.py`.

---

## Overview

`Server` is a single-process HTTP server that accepts text prompts, runs TTS inference on a background worker thread, and plays the resulting audio both locally (via sounddevice) and to any connected HTTP streaming clients. It is started by the CLI and listens by default on `http://127.0.0.1:5001`.

---

## Startup Sequence

```
Server.__init__()
  ├── Load Prefs → resolve active project directory
  ├── Load Project (model params, language, segmentation strategy, etc.)
  ├── Tts.set_model_params_using_project()
  ├── Construct AudioStream        → starts sounddevice OutputStream immediately
  ├── Construct AudioStreamHttp    → starts encode-worker thread
  ├── Wire playback listener:
  │     AudioStream.set_playback_listener(AudioStreamHttp.on_audio_played)
  ├── Start _worker thread         → blocks on queue + waits for TTS ready
  └── (returns to caller)

Server.run()
  ├── Spawn _init_tts thread       → calls Tts.get_instance() (slow model load)
  │     sets _tts_ready event when done
  └── ThreadingHTTPServer.serve_forever()
```

Model loading runs concurrently with the HTTP server so the server can respond to `/status` (returning `"initializing"`) while the model loads.

---

## Threading Model

| Thread | Role |
|---|---|
| Main thread | Runs `ThreadingHTTPServer.serve_forever()` |
| Per-request threads | Spawned by `ThreadingHTTPServer` for each HTTP connection |
| `_worker` thread | Pulls `PromptItem`s from `_queue` and runs TTS inference |
| `_init_tts` thread | Loads the TTS model; signals `_tts_ready` when done |
| `AudioStream` callback | sounddevice audio callback — fires on the audio hardware thread |
| `AudioStreamHttp._encode_worker` | Accumulates PCM blocks and encodes WAV frames for HTTP clients |

The worker thread blocks on `_tts_ready.wait()` before processing its first item, so queued prompts accumulate safely while the model loads.

---

## Queue and Priority

```python
self._queue: queue.PriorityQueue[tuple[int, int, PromptItem]]
```

Each entry is `(priority, counter, PromptItem)`. Lower priority value = processed first.

| Priority | When used |
|---|---|
| `1` | Normal enqueue via `/prompt` |
| `0` | Eager-split remainder — placed at the front |

`counter` is a monotonically increasing integer (from `itertools.count`) used as a tiebreaker so items at the same priority level are processed FIFO.

---

## Eager Splitting

When `can_eager=True` is set on a prompt and the phrase group contains more than one phrase, the worker can split off just the first phrase for immediate inference rather than waiting for the full group.

**Trigger condition** (checked at dequeue time):
```python
self._audio_stream.get_seconds_left() < EAGER_THRESHOLD  # 1.0 s
and prompt_item.can_eager
and len(phrase_group.phrases) > 1
```

**Mechanics:**
1. Extract `phrase_group.phrases[0]` — run inference on it now.
2. Wrap `phrase_group.phrases[1:]` in a new `PhraseGroup` and re-enqueue at priority `0` (front of queue).

This keeps the audio buffer filled during long pauses between queued items — the remainder races back to the front and continues filling the buffer.

`EAGER_THRESHOLD = 1.0` (seconds) is defined at module level.

---

## Worker Pipeline

For each dequeued `PromptItem`:

```
1. Determine prompt_text   (eager split or full group text)
2. Tts.generate_using_project()  → list[Sound]
3. SilenceUtil.trim_silence()    → remove leading/trailing silence
4. SoundUtil.normalize()         → normalize to -3 dBFS headroom
5. SoundUtil.add_silence()       → pad end per phrase reason (paragraph pause, etc.)
                                   (skipped if reason == UNDEFINED)
6. AudioStream.append(sound)     → local playback + drives HTTP feed
```

Errors at step 2 (TTS failure) or step 3 (empty/silence output) log a message and `continue` — the item is discarded and the worker moves on. `_prompt_currently_inferencing` is cleared in a `finally` block so status is always consistent.

---

## HTTP API

### `POST /prompt`

Queue text for TTS inference.

**Request body (JSON):**

| Field | Type | Description |
|---|---|---|
| `prompt` | `string` | Text to synthesize |
| `should_segment` | `bool` | If true, split into phrase groups via `PhraseGrouper` before queuing |
| `eager_first_segment` | `bool` | If true and `should_segment` is true, enable eager splitting |

**Response (JSON):**

| Field | Description |
|---|---|
| `input` | The original prompt string |
| `prompts` | List of phrase group texts that were queued |
| `current_queue_length` | Queue depth after insertion |

### `POST /clear`

Stop playback immediately. Clears `AudioStream._data_buffer` and drains all `AudioStreamHttp` client queues. Returns `{}`.

### `GET /status`

Returns server state. Safe to poll while model is loading.

| Field | Description |
|---|---|
| `status` | `"initializing"` or `"ready"` |
| `inferencing` | Prompt text currently being synthesized (empty string if idle) |
| `audio_buffer` | Seconds of audio remaining in the local playback buffer |
| `num_queued` | Items waiting in the priority queue |
| `stream_clients` | Number of connected HTTP streaming clients |

### `GET /stream`

Long-lived chunked-transfer endpoint. Each connected client receives a stream of length-prefixed WAV frames. See [http-audio-streaming.md](http-audio-streaming.md) for the full wire format and browser client design.

### `GET /streaming-client-demo.html-demo.html`

Serves `server/streaming-client-demo.html-demo.html` — browser demo page for the `/stream` endpoint.

### `GET /` or `GET /api-demo.html`

Serves `server/api-demo.html` — interactive browser UI for sending prompts and monitoring status.

---

## Segmentation

When `should_segment=True`, `PhraseGrouper.text_to_groups()` splits the input into `PhraseGroup` objects, each holding one or more `Phrase` instances. The project's `max_words` and `segmentation_strategy` control splitting behaviour. Each `PhraseGroup` carries a `last_reason` (`Reason` enum) that determines how much silence is padded after that segment (e.g. paragraph boundary vs. sentence boundary vs. undefined).

Without segmentation, the raw prompt is wrapped in a single `PhraseGroup([Phrase(text=prompt, reason=Reason.UNDEFINED)])`. No trailing silence is added (`UNDEFINED` → skipped in step 5 above).

---

## Audio Pipeline Summary

```
POST /prompt
    └─► PhraseGrouper (optional)
          └─► _queue (PriorityQueue)
                └─► _worker thread
                      ├─► Tts.generate_using_project()
                      ├─► SilenceUtil.trim_silence()
                      ├─► SoundUtil.normalize()
                      ├─► SoundUtil.add_silence()  (if reason != UNDEFINED)
                      └─► AudioStream.append()
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          sounddevice output        on_audio_played()
          (local speakers)               │
                                  AudioStreamHttp
                                  _encode_worker
                                         │
                                  WAV frames pushed
                                  to all GET /stream
                                  client queues
```

---

## Key Files

| File | Role |
|---|---|
| `server/server.py` | HTTP routing, queue, worker, eager splitting |
| `server/audio_stream.py` | Local sounddevice playback; fires playback listener |
| `server/audio_stream_http.py` | WAV encoding, HTTP client management, clear |
| `server/streaming-client-demo.html-demo.html` | Browser streaming demo |
| `server/api-demo.html` | Browser prompt/status tester UI |
