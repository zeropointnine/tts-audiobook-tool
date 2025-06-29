import os
import time
import copy

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_sound_segments import *
from tts_audiobook_tool.validate_util import ValidateActionType, ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_items(
            project: Project,
            indices_to_generate: set[int],
            items_to_regenerate: dict[int, str],
            play_on_save: bool
    ) -> bool:
        """
        indices_to_generate and items_to_regenerate are mutually exclusive

        "Regenerate" here simply means that the currently existing, failed file
        for the given index will be deleted first.

        Returns True if ended because interrupted
        """

        # not great
        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, AUDIO_SEGMENTS_SUBDIR)

        if indices_to_generate:
            is_regenerate = False
            # Convert list to dict for compatibility
            items = {}
            for index in indices_to_generate:
                items[index] = ""
        else:
            is_regenerate = True
            items = items_to_regenerate

        did_interrupt = False
        start_time = time.time()
        count = 0
        Shared.mode = "generating"

        for i, path in sorted(items.items()):

            printt()
            print_item_heading(
                is_regenerate, project.text_segments[i].text, i, count, len(items)
            )

            if is_regenerate and path and os.path.exists(path):
                try:
                    Path(path).unlink()
                except:
                    ...

            GenerateUtil.generate_item_full(
                index=i,
                project=project,
                whisper_model=Shared.get_whisper(),
                play_on_save=play_on_save
            )

            count += 1
            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                did_interrupt = True
                break

        Shared.clear_whisper()

        printt()
        printt(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()

        return did_interrupt

    @staticmethod
    def generate_item_full(
        index: int,
        project: Project,
        whisper_model,
        play_on_save: bool
    ) -> None:
        """
        Full workflow for generating an item
        (ie, generate audio, trim silence, validate, fix, retry, giveup)
        """

        MAX_PASSES = 2
        pass_num = 1

        while True:

            # Generate
            sound = GenerateUtil.generate(index, project, True)
            if isinstance(sound, str):
                printt(f"{sound}")
                pass_num += 1
                if pass_num > MAX_PASSES:
                    printt(f"{COL_ERROR}Giving up on item")
                    break
                else:
                    printt(f"{COL_ERROR}Will regenerate")
                    continue

            # Post process
            sound = GenerateUtil.post_process(sound)

            # Transcribe and potentially take 'validation action'
            whisper_data = SoundUtil.transcribe(whisper_model, sound)
            if isinstance(whisper_data, str):
                printt(f"{COL_ERROR}Unexpected whisper result, skipping item: {whisper_data}")
                continue

            save_path = HashFileUtil.make_segment_file_path(index, project)
            validate_action = ValidateUtil.validate_and_save(
                sound,
                project.text_segments[index].text,
                whisper_data,
                save_path,
                (pass_num == 2),
                play_on_save=play_on_save
            )

            if pass_num == 1:

                match validate_action.action:
                    case ValidateActionType.SAVED:
                        printt(f"Saved {save_path}")
                        return
                    case ValidateActionType.TRIMMED_AND_SAVED:
                        printt(validate_action.message)
                        printt(f"Saved {save_path}")
                        return
                    case ValidateActionType.INVALID_DIDNT_SAVE:
                        printt(validate_action.message)
                        printt(f"{COL_ERROR}Will regenerate")
                        pass_num = 2
                        continue
                    case ValidateActionType.ACTION_FAILED:
                        printt(validate_action.message)
                        printt(f"{COL_ERROR}Unexpected error")
                        return
                    case _:
                        L.e("Shouldn't get here")
                        return

            else: # pass_num == 2:

                match validate_action.action:
                    case ValidateActionType.SAVED:
                        printt(f"Saved {save_path}")
                        return
                    case ValidateActionType.TRIMMED_AND_SAVED:
                        printt(validate_action.message)
                        printt(f"Saved {save_path}")
                        return
                    case ValidateActionType.INVALID_SAVED_ANYWAY:
                        printt(validate_action.message)
                        printt(f"{COL_ERROR}Failed again on second attempt, keeping anyway")
                        printt(f"Saved {save_path}")
                        return
                    case ValidateActionType.ACTION_FAILED:
                        printt(validate_action.message)
                        printt(f"{COL_ERROR}Unexpected error")
                        return
                    case _:
                        L.e("Shouldn't get here")
                        break

    @staticmethod
    def generate_post_process_save(
        index: int,
        project: Project
    ) -> tuple[str, str]:
        """
        Generates, does post process step and saves
        Ie, skips error detect / error fix steps.

        Returns saved file path, error string
        """

        sound = GenerateUtil.generate(index, project)
        if isinstance(sound, str):
            return "", f"Couldn't generate audio clip: {sound}"

        sound = GenerateUtil.post_process(sound)

        flac_path = HashFileUtil.make_segment_file_path(index, project)
        err = SoundFileUtil.save_flac(flac_path, sound)
        if err:
            return "", err
        else:
            return flac_path, err

    @staticmethod
    def post_process(sound: Sound) -> Sound:

        # Trim all silence from ends of audio clip
        sound = SilenceUtil.trim_silence(sound)
        SoundFileUtil.debug_save("after trim silence", sound)

        # Prevent 0-byte audio data as a failsafe
        if len(sound.data) == 0:
            sound = sound._replace(data=np.array([0], dtype=sound.data.dtype))

        return sound


    @staticmethod
    def generate(
        index: int,
        project: Project,
        print_info: bool=True
    ) -> Sound | str:
        """
        Returns model-generated normalized sound data, in model's native samplerate.
        """

        start_time = time.time()
        text_segment = project.text_segments[index]

        if Shared.is_oute():
            sound = GenerateUtil.generate_oute(
                text_segment.text,
                project.oute_voice_json,
                project.oute_temperature)
        else:
            sound = GenerateUtil.generate_chatterbox(
                text_segment.text,
                project)

        if not sound:
            return "Model failed"

        num_nans = np.sum(np.isnan(sound.data))
        if num_nans > 0:
            return "Model outputted NaN, discarding"

        if print_info:
            elapsed = time.time() - start_time or 1.0
            s = f"Audio duration: {COL_ACCENT}{sound.duration:.1f}s{COL_DEFAULT}, inference time: {COL_ACCENT}{elapsed:.1f}s"
            multi = sound.duration / elapsed
            s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
            printt(s)

        SoundFileUtil.debug_save("after gen", sound)

        return sound

    @staticmethod
    def generate_oute(
        prompt: str,
        voice: dict,
        temperature: float = -1
    ) -> Sound | None:

        oute = Shared.get_oute()

        # First, clone GENERATION_CONFIG from config file
        from outetts.models.config import SamplerConfig # type: ignore
        from tts_audiobook_tool.config_oute import GENERATION_CONFIG
        try:
            from .config_oute_dev import GENERATION_CONFIG
        except ImportError:
            pass
        gen_config = copy.deepcopy(GENERATION_CONFIG)

        gen_config.text = prompt
        gen_config.speaker = voice
        if temperature != -1:
            gen_config.sampler_config = SamplerConfig(temperature)

        try:
            output = oute.generate(config=gen_config)
            audio = output.audio.cpu().clone().squeeze().numpy()
            return Sound(audio, output.sr)

        except Exception as e:
            printt(f"{COL_ERROR}Oute model error: {e}")
            return None

    @staticmethod
    def generate_chatterbox(
        prompt: str,
        project: Project
    ) -> Sound | None:
        """
        Returns normalized sound data and sample rate
        Returns None on fail
        """

        chatterbox = Shared.get_chatterbox()

        d = {}
        if project.chatterbox_voice_file_name:
            # Rem, this is actually optional
            path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
            d["audio_prompt_path"] = path
        if project.chatterbox_exaggeration != -1:
            d["exaggeration"] = project.chatterbox_exaggeration
        if project.chatterbox_cfg != -1:
            d["cfg_weight"] = project.chatterbox_cfg
        if project.chatterbox_temperature != -1:
            d["temperature"] = project.chatterbox_temperature

        try:
            data = chatterbox.generate(prompt, **d)
            data = data.numpy().squeeze()
            data = SoundUtil.normalize(data, headroom_db=1.0)
            return Sound(data, chatterbox.sr)

        except Exception as e:
            print(f"{COL_ERROR}Chatterbox model error: {e}\a", "error")
            return None

# ---

def print_item_heading(is_regenerate: bool, text: str, index: int, count: int, total: int) -> None:
    verb = "Regenerating" if is_regenerate else "Generate"
    s  = f"{COL_ACCENT}[{COL_DEFAULT}{count+1}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
    s += f"{COL_ACCENT}{verb} audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
    printt(s)
    printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
    printt()
