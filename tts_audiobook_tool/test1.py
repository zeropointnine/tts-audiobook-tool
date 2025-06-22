import os
from pathlib import Path

from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.constants import WHISPER_SAMPLERATE
from tts_audiobook_tool.l import L
from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.validate_util import ValidateUtil

L.init("temp")

if False:

    dir = r"C:\workspace\test1\segments\New Folder"
    LoudnessNormalizationUtil.print_lra_directory(dir)

if False:

    project = Project.load_using_dir_path(r"C:\workspace\test1")
    assert(isinstance(project, Project))

    whisper = Shared.get_whisper()

    dir = r"C:\workspace\test1\segments - trimmed"
    files = os.listdir(dir)

    for i, file in enumerate(files):

        path = os.path.join(dir, file)
        o =SoundFileUtil.load(path, WHISPER_SAMPLERATE)
        assert(isinstance(o, tuple))
        data, sr = o
        duration = len(data) / sr

        segment = project.text_segments[i]

        whisper_data = whisper.transcribe(data, word_timestamps=True)

        print("-------------------------")
        print(Path(path).stem)
        print(segment.text.strip())

        print()
        ValidateUtil.print_whisper_data(whisper_data)

        o = ValidateUtil.get_semantic_match_end_trim_time(segment.text, whisper_data, duration)
        print("end trim time", o, "duration", f"{(len(data) / sr):.2f}" )

if False:

    dir = r"C:\workspace\test1\segments - gen only - chatterbox"

    for item in os.listdir(dir):

        path = os.path.join(dir, item)
        print()
        print(path)

        sound =SoundFileUtil.load(path)
        if isinstance(sound, str):
            print("PROBLEM")
            continue

        print('duration', sound.duration)

        a, b = SilenceUtil.get_start_end_silence_file(path)
        if not a and not b:
            print("nothing to trim")
            continue
        print(f"trim {a:.2f} {b:.2f}")

        data = SoundUtil.trim(sound, a, b)
        p = Path(path)
        dest_path = p.with_stem(p.stem + " TRIMMED")
        err = SoundFileUtil.save_flac(str(dest_path), sound)
        print("error if any:", err)

        print()

if False:

    # Semantic trim, start and end


    # Whisper fails on all the funny sounds before "professor", lol
    items = [
        ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02052] [a795abd1d6a4188f] [sen_last_ch_combined] Professor_Carter_asked.flac"),
        ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02069] [9225610f76d54f58] [sen_last_ch_combined] Professor_Carter_asked.flac"),
        ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02076] [6b5f1e7d8c7fa9c6] [sen_last_ch_combined] Professor_Carter_asked.flac"),
        ("professor carter asked", r"C:\workspace\_bad outputs to address\noise before [02655] [e3fe13e9721e19c8] [sen_last_ch_combined] Professor_Carter_asked.flac")
    ]

    items = [
        # whisper returns value in the middle of the silence btw spurious noise and word, consistently
        # ("Never wake up?", r"C:\workspace\_bad outputs to address\noise before start - Never wake up.flac")

        # whisper returns time at very start of matched word  * only succeeded in doing this first time out of many attempts!
        # ("Just watching", r"C:\workspace\_bad outputs to address\noise before start - Just_watching.flac"),

        # whisper fails to isolate first word here
        # ("The girl was startled", r"C:\workspace\_bad outputs to address\noise before start, subtle - The_girl_was_startled.flac"),

        # or here
        # ("Oh", r"C:\workspace\_bad outputs to address\noise before start, subtle - oh.flac")
    ]

    for ref_text, path in items:

        sound =SoundFileUtil.load(path)
        if isinstance(sound, str):
            print("NO:", sound)
            continue

        whisper_data = SoundUtil.transcribe(Shared.get_whisper(), sound)
        if isinstance(whisper_data, str):
            print("NO:", whisper_data)
            continue

        print("-------------------------------")
        print(path)
        print("Duration", sound.duration)
        print("Reference text:", ref_text)
        print("Transcription:", TranscribeUtil.get_pretty_whisper_data_string(whisper_data))

        start = TranscribeUtil.get_semantic_match_start_time_trim(ref_text, whisper_data, sound)
        print("start", start)

        end = TranscribeUtil.get_semantic_match_end_time_trim(ref_text, whisper_data, sound)
        print("end", end)

if False:

    # Repeats

    items = [
        ("something something", r"C:\workspace\_bad outputs to address\repeat phrase chatterbox - Echika_s_Your_Forma_analyzed_the_creature_it_was.flac"),
        ("There's actually something i need to show you", r"C:\workspace\_bad outputs to address\repeat phrase oute.flac")
    ]

    for ref_text, path in items:

        sound =SoundFileUtil.load(path)
        if isinstance(sound, str):
            print("NO:", sound)
            continue

        whisper_data = SoundUtil.transcribe(Shared.get_whisper(), sound)
        if isinstance(whisper_data, str):
            print("NO:", whisper_data)
            continue

        print("-------------------------------")
        print(path)
        print("Reference text:", ref_text)
        print("Transcription:", TranscribeUtil.get_pretty_whisper_data_string(whisper_data))
        print("Bad repeats:", TranscribeUtil.find_bad_repeats(ref_text, whisper_data))

if False:

    # Loudness normalization...

    print("Starting...")
    path1 = r"C:\workspace\loudness-test\chatterbox pre-normalization.flac"
    path2 = r"C:\workspace\loudness-test\chatterbox post-normalization.flac"
    err = LoudnessNormalizationUtil.normalize_file(path1, path2)
    if not err:
        print("success")
    else:
        print("error:", err)


# ---------------

