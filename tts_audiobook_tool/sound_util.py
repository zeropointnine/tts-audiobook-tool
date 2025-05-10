import sounddevice as sd
import wave
import numpy as np
import threading
import pyloudnorm as pyln
import soundfile as sf

from tts_audiobook_tool.util import printt

class SoundUtil:

    @staticmethod
    def play_wav(file_path: str):
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            dtype = 'int16' if wf.getsampwidth() == 2 else 'int32'

            def callback(outdata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                data = wf.readframes(frames)
                if len(data) == 0:
                    raise sd.CallbackStop()
                outdata[:len(data)] = np.frombuffer(data, dtype=dtype).reshape(-1, num_channels)

            with sd.OutputStream(samplerate=sample_rate, channels=num_channels, callback=callback, dtype=dtype):
                sd.sleep(int(wf.getnframes() / sample_rate * 1000))

    @staticmethod
    def play_flac_async(file_path: str):
        """
        Plays a FLAC file asynchronously using a streaming approach.
        Eats exceptions and prints them.
        """
        def _play_stream():
            try:
                with sf.SoundFile(file_path, 'r') as flac_file:
                    samplerate = flac_file.samplerate
                    channels = flac_file.channels

                    stream_finished_event = threading.Event()

                    def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                        if status:
                            print(f"Audio callback status (FLAC): {status}")

                        # Read 'frames' number of frames from the FLAC file
                        # always_2d=True ensures data is (n_frames, n_channels)
                        data_read = flac_file.read(frames, dtype='float32', always_2d=True)

                        if data_read.shape[0] == 0: # No more data
                            raise sd.CallbackStop()

                        # Fill outdata
                        if data_read.shape[0] < frames:
                            outdata[:data_read.shape[0]] = data_read
                            # Zero out the rest of the buffer
                            outdata[data_read.shape[0]:] = 0
                        else:
                            outdata[:] = data_read

                    def set_event_on_finish():
                        stream_finished_event.set()

                    with sd.OutputStream(
                        samplerate=samplerate,
                        channels=channels,
                        callback=callback,
                        dtype='float32', # Ensure this matches data_read dtype
                        finished_callback=set_event_on_finish
                    ):
                        stream_finished_event.wait() # Wait for playback to complete

            except Exception as e:
                printt(f"Couldn't play FLAC audio file via stream: {file_path} - {e}")

        thread = threading.Thread(target=_play_stream, daemon=True)
        thread.start()

    @staticmethod
    def normalize_and_overwrite(wav_file_path: str) -> str:
        """
        Normalizes the loudness of a WAV file to a target LUFS and overwrites the original file.
        Returns empty string on success or error string on fail
        Gemini 2.5 Pro
        """
        try:
            audio, sample_rate = sf.read(wav_file_path, dtype='float32')

            # Ensure mono audio for loudness measurement if stereo
            if audio.ndim > 1 and audio.shape[1] > 1:
                 # Average channels for loudness calculation, but keep original channels for output
                 audio_mono = np.mean(audio, axis=1)
            else:
                 audio_mono = audio

            # Measure loudness
            meter = pyln.Meter(sample_rate)

            if len(audio_mono) < sample_rate * 0.4:
                 # Skip, needs at least 400ms for reliable measurement
                 return ""

            loudness = meter.integrated_loudness(audio_mono)

            if loudness == -np.inf:
                # Skip, avoid division by zero or extreme gain if loudness is -inf (silence)
                return ""

            # Target loudness (LUFS) - EBU R 128 suggests -23.0 LUFS, but -24 is also common for podcasts/audiobooks
            target_lufs = -24.0
            gain_db = target_lufs - loudness
            gain_linear = 10 ** (gain_db / 20.0)

            # Apply gain
            normalized_audio = audio * gain_linear

            # Peak normalize to prevent clipping (-1.0 dBTP ceiling is common, 0.0 is max)
            # Use pyloudnorm's peak normalization
            normalized_audio = pyln.normalize.peak(normalized_audio, -1.0)

            # Write the normalized audio back to the original file
            # Use soundfile to preserve original subtype if possible (e.g., PCM_16, PCM_24, FLOAT)
            sf.write(wav_file_path, normalized_audio, sample_rate) # sf handles dtype conversion based on file subtype
            return ""

        except Exception as e:
            return f"Error normalizing {wav_file_path}: {e}"
            # TODO should do temp file treatment
