# Stream Client Demo

`tts_audiobook_tool/server/streaming-client-demo.html-demo.html` is a minimal, self-contained browser implementation of the `GET /stream` protocol. It exists as a working reference for developers integrating the audio stream into their own clients.

Served at `http://localhost:5001/streaming-client-demo.html-demo.html` while the TTS server is running.

The pattern is the same one used by Shoutcast and Icecast internet radio: a client opens a single long-lived HTTP GET, the server keeps the connection open and pushes encoded audio as fast as it is produced, and the client buffers and plays continuously without ever requesting a second resource. The difference here is framing — Shoutcast pushes a raw compressed byte stream (MP3) with no explicit message boundaries, whereas `/stream` wraps each audio chunk in a 4-byte length prefix so the client always knows exactly where one WAV frame ends and the next begins.

---

## What it demonstrates

The demo covers the full client-side contract in ~130 lines of vanilla JS:

1. Opening a streaming `fetch()` connection to `/stream`
2. Reassembling the length-prefixed wire format from raw TCP bytes
3. Recognising and discarding keepalive heartbeat frames
4. Decoding each WAV frame with the Web Audio API
5. Scheduling decoded buffers for gapless, gap-tolerant playback
6. Tearing down cleanly on user disconnect

---

## Wire format reassembly

`fetch()` delivers bytes in arbitrary TCP-segment-sized chunks, not aligned to message boundaries. The client maintains an accumulator `Uint8Array` and drains complete frames in a loop after each read:

```javascript
while (accumulator.length >= 4) {
    const view   = new DataView(accumulator.buffer, accumulator.byteOffset, accumulator.byteLength);
    const msgLen = view.getUint32(0, false); // big-endian uint32
    if (accumulator.length < 4 + msgLen) break;
    const wavSlice = accumulator.slice(4, 4 + msgLen);
    accumulator    = accumulator.slice(4 + msgLen);
    if (msgLen === 0) continue; // keepalive heartbeat — skip
    scheduleChunk(wavSlice);
}
```

Each frame is `[ 4-byte uint32 length ][ N-byte WAV ]`. A `length = 0` frame is a server heartbeat — ignore it. See [http-audio-streaming.md](http-audio-streaming.md) for the full server-side wire format.

---

## Gapless playback

`decodeAudioData()` is async, so naive scheduling would fire chunks out of order or with gaps. The solution is a promise chain that serializes all decodes:

```javascript
let nextStartTime = 0;
let scheduleChain = Promise.resolve();

function scheduleChunk(wavSlice) {
    const owned = wavSlice.buffer.slice(              // owned copy — decodeAudioData detaches input
        wavSlice.byteOffset,
        wavSlice.byteOffset + wavSlice.byteLength
    );
    scheduleChain = scheduleChain
        .then(() => audioCtx.decodeAudioData(owned))
        .then(audioBuffer => {
            const src = audioCtx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(audioCtx.destination);
            const t = Math.max(audioCtx.currentTime + 0.05, nextStartTime);
            src.start(t);
            nextStartTime = t + audioBuffer.duration;
        })
        .catch(() => { /* swallow — keep chain alive for subsequent chunks */ });
}
```

`Math.max(audioCtx.currentTime + 0.05, nextStartTime)` handles both cases:

- **Normal flow** — next chunk arrives before current one finishes → `nextStartTime` is in the future → scheduled seamlessly end-to-end, zero gap.
- **Silence / reconnect** — `nextStartTime` drifted into the past → reschedule from `currentTime + 0.05` → small startup gap, no negative-time error.

The `.catch()` swallows decode errors so a single bad frame does not break the chain for all subsequent frames.

---

## `AudioContext` lifecycle

Browsers require a user gesture to create or resume an `AudioContext`. The Connect click handler manages this:

```javascript
if (!audioCtx) {
    audioCtx = new AudioContext();   // first click — create
} else {
    audioCtx.resume();               // subsequent reconnect — resume
}
```

On disconnect, `audioCtx.close()` is called and the reference set to `null`. This immediately stops all scheduled audio. The next Connect creates a fresh context.

`AbortController` cancels the `fetch()` on disconnect. This surfaces as an `AbortError` in `reader.read()`, which is caught and suppressed in the `finally` cleanup so state is always reset regardless of how the stream ends.

---

## Related

| Resource | Description |
|---|---|
| [http-audio-streaming.md](http-audio-streaming.md) | Full wire format, server encode worker, and client design |
| [server.md](server.md) | Server architecture, threading model, and full HTTP API |
| `tts_audiobook_tool/server/audio_stream_http.py` | `AudioStreamHttp` — server-side encode and client management |
