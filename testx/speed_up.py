import librosa
import soundfile as sf
from audiotsm import wsola
from audiotsm.io.array import ArrayReader, ArrayWriter

# Speed up voice clone sample

fn = r"C:\w\voice\amanda lee jingliu\amanda lee jingliu 2.wav"

y, sr = librosa.load(fn, sr=None)

# Set up the reader and writer
reader = ArrayReader(y.reshape(1, -1)) # audiotsm expects a 2D array
writer = ArrayWriter(channels=1)

# The WSOLA processor
# The frame_length and synthesis_hop parameters can be tuned, but defaults are often good.
tsm = wsola(channels=1, speed=1.25)

# Process the audio
tsm.run(reader, writer)
y_fast = writer.data.flatten() # Get the processed audio back as a 1D array

# Save the output
sf.write("test1.wav", y_fast, sr)

print("done")
