import subprocess
import sounddevice as sd
import wave
import numpy as np
import threading
import soundfile as sf

from tts_audiobook_tool.util import printt

class SoundUtil:

    @staticmethod
    def encode_to_flac(wav_path: str, flac_path: str) -> bool:

        # TODO: use "make_file" here

        try:
            # Construct the FFmpeg command with proper escaping for filenames
            cmd = [
                'ffmpeg',
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                '-i', wav_path,
                '-c:a', 'flac',
                '-compression_level', '5',
                flac_path
            ]

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True

        except subprocess.CalledProcessError as e:
            printt(str(e.stderr), "error")
            return False
        except Exception as e:
            printt(str(e), "error")
            return False


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

    # ---

