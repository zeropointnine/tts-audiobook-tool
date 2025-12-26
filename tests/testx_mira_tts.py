"""
Testing Mira TTS with app dependencies
Using that project's reference code but also activating faster-whisper since that's the main pain point
"""

import time

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.whisper_util import WhisperUtil
start_time = time.time()

from mira.model import MiraTTS # type: ignore

import torch
import soundfile as sf


# ----------------------------------------

def invoke_faster_whisper(sound: Sound):
    result = WhisperUtil.transcribe_to_words(sound, language_code="en")
    assert(isinstance(result, list))
    print("transcript:\n", WhisperUtil.get_flat_text_from_words(result), "\n")

# ----------------------------------------

voice_clone_path = "/home/lee/Documents/w/voice/erica mendez/sen-last-ch-6.wav" 

prompts = [
    'The sun had not yet risen.',
    'The sea was indistinguishable from the sky, except that the sea was slightly creased as if a cloth had wrinkles in it.',
    'Gradually as the sky whitened a dark line lay on the horizon dividing the sea from the sky and the grey cloth became barred with thick strokes moving, one after another, beneath the surface, following each other, pursuing each other,',
    'perpetually.',
    'As they neared the shore each bar rose, heaped itself, broke and swept a thin veil of white water across the sand.',
    'The wave paused, and then drew out again, sighing like a sleeper whose breath comes and goes unconsciously.',
    'Gradually the dark bar on the horizon became clear as if the sediment in an old wine-bottle has sunk and left the glass green.',
    'Behind it, too, the sky cleared as if the white sediment there had sunk, or as if the arm of a woman couched beneath the horizon had raised a lamp and flat bars of white,',
    'green and yellow spread across the sky like the blades of a fan.',
    'Then she raised her lamp higher and the air seemed to become fibrous and to tear away from the green surface flickering and flaming in red and yellow fibres like the smoky fire that roars from a bonfire.',
    'Gradually the fibres of the burning bonfire were fused into one haze, one incandescence which lifted the weight of the woollen grey sky on top of it and turned it to a million atoms of soft blue.',
    'The surface of the sea slowly became transparent and lay rippling and sparkling until the dark stripes were almost rubbed out.',
    'Slowly the arm that held the lamp raised it higher and then higher until a broad flame became visible; an arc of fire burnt on the rim of the horizon, and all round it the sea blazed gold.',
    'The light struck upon the trees in the garden, making one leaf transparent and then another.',
    'One bird chirped high up; there was a pause; another chirped lower down.',
    'The sun sharpened the walls of the house, and rested like the tip of a fan upon a white blind and made a blue finger-print of shadow under the leaf by the bedroom window.',
    'The blind stirred slightly, but all within was dim and unsubstantial.',
    'The birds sang their blank melody outside.'
]

mira_tts = MiraTTS('YatharthS/MiraTTS') 

BATCH_MODE = False

if not BATCH_MODE:

    context_tokens = mira_tts.encode_audio(voice_clone_path)

    print(f"\nwarm-up time: {time.time() - start_time}\n") # 9s
    start_time = time.time()

    for prompt in prompts:
        audio_tensor = mira_tts.generate(prompt, context_tokens)    
        audio_np = audio_tensor.to(torch.float32).cpu().numpy()
        sf.write(f'output_{time.time()}.wav', audio_np, samplerate=48000)
        print("inference complete:\n", prompt, "\n")

        audio_tensor = mira_tts.generate(prompts, context_tokens)    
        audio_np = audio_tensor.to(torch.float32).cpu().numpy()

        sound = Sound(audio_np, 48000)
        invoke_faster_whisper(sound)

        sf.write(f'output_{time.time()}.wav', audio_np, samplerate=48000)            

    print(f"\ninference time: {time.time() - start_time}\n") # 20s

else:

    context_tokens = [mira_tts.encode_audio(voice_clone_path)]

    print(f"\nwarm-up time: {time.time() - start_time}\n")
    start_time = time.time()

    audio_tensor = mira_tts.batch_generate(prompts, context_tokens)
    audio_np = audio_tensor.to(torch.float32).cpu().numpy()

    sound = Sound(audio_np, 48000)
    invoke_faster_whisper(sound)

    sf.write(f'output_{time.time()}.wav', audio_np, samplerate=48000)            

    print(f"\ninference time: {time.time() - start_time}\n") # 4.3s
