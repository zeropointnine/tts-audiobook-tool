from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.tts import Tts


groups = [
    ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02052] [a795abd1d6a4188f] [sen_last_ch_combined] Professor_Carter_asked.flac"),
    ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02069] [9225610f76d54f58] [sen_last_ch_combined] Professor_Carter_asked.flac"),
    ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02076] [6b5f1e7d8c7fa9c6] [sen_last_ch_combined] Professor_Carter_asked.flac"),
    ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02655] [e3fe13e9721e19c8] [sen_last_ch_combined] Professor_Carter_asked.flac")
]

groups = [
    # whisper returns value in the middle of the silence btw spurious noise and word, consistently
    # ("Never wake up?", r"C:\workspace\_bad outputs to address\noise before start - Never wake up.flac")

    # whisper returns time at very start of matched word  * only succeeded in doing this first time out of many attempts!
    # ("Just watching", r"C:\workspace\_bad outputs to address\noise before start - Just_watching.flac"),

    # whisper fails to isolate first word here
    # ("The girl was startled", r"C:\workspace\_bad outputs to address\noise before start, subtle - The_girl_was_startled.flac"),

    # or here
    # ("Oh", r"C:\workspace\_bad outputs to address\noise before start, subtle - oh.flac")
]

# for ref_text, path in groups:

#     sound =SoundFileUtil.load(path)
#     if isinstance(sound, str):
#         print("NO:", sound)
#         continue

#     whisper_data = SoundUtil.transcribe(Tts.get_openai_whisper(), sound)
#     if isinstance(whisper_data, str):
#         print("NO:", whisper_data)
#         continue

#     print("-------------------------------")
#     print(path)
#     print("Duration", sound.duration)
#     print("Reference text:", ref_text)
#     print("Transcription:", TranscribeUtil.get_pretty_whisper_data_string(whisper_data))

#     start = TranscribeUtil.get_semantic_match_start_time_trim(ref_text, whisper_data, sound)
#     print("start", start)

#     end = TranscribeUtil.get_semantic_match_end_time_trim(ref_text, whisper_data, sound)
#     print("end", end)

