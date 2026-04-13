# HTTP Audio Streaming

Real-time audio streaming from the TTS server to browser clients over HTTP chunked transfer.

---

## Overview

HTTP clients are fed directly from the `AudioStream` sounddevice playback callback. As the callback consumes PCM blocks from the local buffer and sends them to the audio hardware, those same blocks are forwarded to all connected HTTP clients. This means:

- HTTP clients are at most ~0.5 s behind local playback (one encode window).
- `clear()` empties the local `AudioStream` buffer, which immediately silences the callback, which starves the HTTP feed within one encode window (~0.5 s). `AudioStreamHttp.clear()` also drains the in-flight PCM queue and all client delivery queues for a near-instant stop.
- Late-joining clients start from "now" — they hear whatever is currently playing and everything generated after they connect.

---

## Wire Format

Each audio chunk is transmitted as a **length-prefixed WAV frame**:

```
[ 4 bytes: uint32 big-endian length ][ N bytes: WAV (IEEE 754 float32 PCM) ]
```

- Encoding: `soundfile` writes float32 WAV to a `BytesIO` buffer — no extra dependencies
- Sample rate: 48 000 Hz (all audio is resampled by `AudioStream` before entering the playback buffer)
- Channel: mono
- Each frame covers ~0.5 s of audio (`_ENCODE_BLOCK_SAMPLES = 24 000` samples)
- Framing repeats for the duration of the connection; no terminator is sent

A frame with `length = 0` is a keepalive heartbeat — the client ignores it.

The HTTP response uses `Transfer-Encoding: chunked`, so the browser's `fetch()` ReadableStream receives the decoded payload bytes directly (HTTP chunked framing is transparent).

---

## Server

### `AudioStreamHttp` (`server/audio_stream_http.py`)

Manages connected clients and distributes encoded audio. Fed by `AudioStream` via a playback listener — not by the worker thread directly.

| Method | Description |
|---|---|
| `on_audio_played(data, sr)` | Called from `AudioStream._callback` (audio thread). Enqueues the PCM block for the encode worker. Fast path — only does a `SimpleQueue.put_nowait`. |
| `connect() → queue.Queue` | Register a new client; returns a queue that receives encoded frames |
| `disconnect(q)` | Deregister a client queue |
| `clear()` | Drain the PCM queue, accumulator, and all client delivery queues |
| `client_count() → int` | Number of currently connected clients |

#### Encode worker thread

A single background thread (`_encode_worker`) drains `_pcm_queue`, accumulates samples into `_accumulator`, and flushes when either:

- `_accumulator` reaches `_ENCODE_BLOCK_SAMPLES` (24 000 samples, ~0.5 s), or
- `_pcm_queue.get()` times out after 0.2 s — audio went silent, flush the tail.

On flush, the accumulator is encoded as a single WAV frame and pushed to all client queues. Client queues are bounded (`maxsize=20`); chunks are dropped silently for slow clients.

#### Data flow

```
AudioStream._callback  →  on_audio_played()  →  _pcm_queue
                                                      ↓
                                              _encode_worker
                                                      ↓
                                           encode WAV (~0.5 s)
                                                      ↓
                                        push to all client queues
```

### `AudioStream` (`server/audio_stream.py`)

`AudioStream.set_playback_listener(listener)` registers a callable `(data: np.ndarray, sr: int) → None`. The `_callback` calls it outside the buffer lock after filling `outdata`, passing `outdata[:write_pos, 0].copy()` — only the samples that were actually played (non-silence).

### `GET /stream` endpoint (`server/server.py`)

Long-lived HTTP endpoint. One thread (from `ThreadingHTTPServer`'s pool) is occupied per connected client.

```
GET /stream HTTP/1.1

HTTP/1.1 200 OK
Content-Type: application/octet-stream
Transfer-Encoding: chunked
Cache-Control: no-cache
X-Accel-Buffering: no
```

Response body: a continuous sequence of length-prefixed WAV frames as described above.

Each frame is written in HTTP chunked format:
```
<hex length>\r\n<frame bytes>\r\n
```

The handler loops on `queue.get(timeout=5)`. On `queue.Empty` it sends a 4-byte heartbeat frame (`\x00\x00\x00\x00`) and flushes — this probes the socket so a dead connection is detected within 5 s and `disconnect()` is called. On `BrokenPipeError` / `ConnectionResetError` it breaks.

### `GET /streaming-client-demo.html`

Serves `server/streaming-client-demo.html`, the browser demo page.

### Worker integration

In `Server._worker`, after each `Sound` is processed:

```python
self._audio_stream.append(sound)   # local sounddevice playback + drives HTTP feed
```

The HTTP broadcast is implicit: `append()` places PCM in `AudioStream._data_buffer`; the sounddevice callback consumes it and calls `on_audio_played()` on each block.

### Wiring

In `Server.__init__`, after both objects are constructed:

```python
self._audio_stream.set_playback_listener(self._audio_http_stream.on_audio_played)
```

### `clear()` propagation

```python
def clear(self):
    self._audio_stream.clear()        # empties _data_buffer → callback goes silent
    self._audio_http_stream.clear()   # drains PCM queue, accumulator, client queues
```

### `/status` response

Includes `"stream_clients"` — the number of currently connected streaming clients.

---

## Browser Client

### Accumulator pattern

`fetch()` delivers data in arbitrary TCP-segment-sized chunks, not aligned to application message boundaries. An accumulator `Uint8Array` collects incoming bytes; complete messages are drained in a loop:

```javascript
while (accumulator.length >= 4) {
    const view   = new DataView(accumulator.buffer, accumulator.byteOffset, accumulator.byteLength);
    const msgLen = view.getUint32(0, false); // big-endian
    if (accumulator.length < 4 + msgLen) break;
    const wavSlice = accumulator.slice(4, 4 + msgLen);
    accumulator    = accumulator.slice(4 + msgLen);
    if (msgLen === 0) continue; // heartbeat frame — skip
    scheduleChunk(wavSlice);
}
```

### Gapless playback

`decodeAudioData()` is async. A promise chain (`scheduleChain`) serializes decodes so that `nextStartTime` is updated in strict arrival order:

```javascript
let nextStartTime = 0;
let scheduleChain = Promise.resolve();

function scheduleChunk(wavSlice) {
    // Copy to owned ArrayBuffer — decodeAudioData detaches its input
    const owned = wavSlice.buffer.slice(
        wavSlice.byteOffset,
        wavSlice.byteOffset + wavSlice.byteLength
    );
    scheduleChain = scheduleChain
        .then(() => audioCtx.decodeAudioData(owned))
        .then(audioBuffer => {
            const src = audioCtx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(audioCtx.destination);
            // At least 50ms ahead, or seamlessly after the previous chunk
            const t = Math.max(audioCtx.currentTime + 0.05, nextStartTime);
            src.start(t);
            nextStartTime = t + audioBuffer.duration;
        })
        .catch(() => { /* swallow to keep chain alive */ });
}
```

`Math.max(audioCtx.currentTime + 0.05, nextStartTime)`:
- When the next chunk arrives before the current one finishes playing, `nextStartTime` is in the future — gapless.
- When the server is silent and `nextStartTime` has drifted into the past, reschedules from `currentTime + 0.05` — small gap, no error.

### `AudioContext` lifecycle

`AudioContext` must be created or resumed inside a user gesture (browser requirement). The Connect button handler creates it on first click and calls `resume()` on subsequent reconnects.

On Disconnect, `audioCtx.close()` is called and `audioCtx` is set to `null`. This stops all scheduled audio immediately. On the next Connect a fresh context is created.

`AbortController` is used to cancel the `fetch()` when the user clicks Disconnect. This causes an `AbortError` in `reader.read()`, which is caught and ignored in the `finally` cleanup.

---

## Files

| File | Role |
|---|---|
| `tts_audiobook_tool/server/audio_stream_http.py` | `AudioStreamHttp` class — encode worker, client management, clear |
| `tts_audiobook_tool/server/server.py` | HTTP server — routes, worker integration, wiring |
| `tts_audiobook_tool/server/streaming-client-demo.html` | Browser demo page |
| `tts_audiobook_tool/server/audio_stream.py` | Local sounddevice playback; fires playback listener from callback |
