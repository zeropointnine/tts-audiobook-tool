import sounddevice as sd
import wave
import numpy as np
import threading
import pyloudnorm as pyln
import soundfile as sf

from .util import printt

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
        Eats exceptions
        """
        def _play():
            try:
                data, samplerate = sf.read(file_path, dtype='float32')
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                printt(f"Couldn't play audio file: {e}")
        thread = threading.Thread(target=_play, daemon=True)
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
