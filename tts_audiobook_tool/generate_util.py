import os
import time
import copy

from tts_audiobook_tool.concat_submenu import ConcatSubmenu
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import *
from tts_audiobook_tool.validate_util import ValidateItem, ValidateResult, ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_validate_fix_items(
            state: State,
            indices: list[int],
            mode: str
    ) -> None:
        """
        mode:
            "generate"
            "generate-and-fix"
            "validate-and-fix": presupposes pre-existing files, skips the first of two potential generate passes
        """

        if not mode in ["generate", "generate-and-fix", "validate-and-fix"]:
            raise ValueError("Bad value for mode")

        index_to_path = ProjectDirUtil.get_indices_and_paths(state)

        did_interrupt = False
        start_time = time.time()
        count = 0
        Shared.mode = "generating"

        for i in indices:

            printt()
            print_item_heading(state.project.text_segments[i].text, i, count, len(indices))

            if mode == "generate":
                _ = GenerateUtil.generate_and_make_flac(index=i, state=state, is_retry=False)

            elif mode == "generate-and-fix":
                GenerateUtil.generate_validate_fix_item(
                    index=i, state=state, whisper_model=Shared.get_whisper())

            else: # == "validate-and-fix"
                flac_path = index_to_path[i]
                GenerateUtil.generate_validate_fix_item(
                        index=i, state=state, whisper_model=Shared.get_whisper(), skip_generate_file_path=flac_path)

            count += 1
            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                did_interrupt = True
                break

        printt()
        printt(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()

        Shared.clear_whisper()

        if did_interrupt:
            ask("Press enter: \a")
        else:
            hotkey = ask_hotkey(f"Press enter or {make_hotkey_string("C")} to concatenate files now: ")
            if hotkey == "c":
                ConcatSubmenu.submenu(state)

    @staticmethod
    def generate_validate_fix_item(
        index: int,
        state: State,
        whisper_model,
        skip_generate_file_path: str=""
    ) -> None:
        """
        skip_generate_file_path
            is used to "validate pre-existing item and fix" (as opposed to "generate, validate and fix")
        """

        pass_num = 1

        while True:

            # Make audio file
            if not skip_generate_file_path:
                file_path = GenerateUtil.generate_and_make_flac(index=index, state=state, is_retry=(pass_num > 1))
                if not file_path:
                    # Unexpected error making audio file
                    break
            else:
                # Validate-and-then-fix
                file_path = skip_generate_file_path
                skip_generate_file_path = ""

            # Validate audio file
            item = ValidateItem(index, file_path, state.project.text_segments[index].text)
            should_fix_delete = (pass_num == 1)
            result, message = ValidateUtil.validate_item(item, should_fix_delete, whisper_model)

            if pass_num == 1:

                match result:
                    case ValidateResult.VALIDATED_AND_TAGGED:
                        printt("Validated")
                        break
                    case ValidateResult.FAILED_AND_CORRECTED:
                        printt(message)
                        printt(f"{COL_OK}Corrected")
                        break
                    case ValidateResult.FAILED_AND_DELETED:
                        printt(message)
                        printt(f"{COL_ERROR}Will regenerate")
                        pass_num = 2
                        continue
                    case _:
                        L.e("Shouldn't get here")
                        break

            else: # pass_num == 2:

                match result:
                    case ValidateResult.VALIDATED_AND_TAGGED:
                        printt(f"{COL_OK}")
                        printt("Validated on second attempt")
                        break
                    case ValidateResult.FAILED_AND_CORRECTED:
                        printt(message)
                        printt(f"{COL_OK}Corrected on second attempt")
                        break
                    case ValidateResult.FAILED_ONLY:
                        printt(message)
                        printt(f"{COL_ERROR}Failed again on second attempt, continuing")
                        break
                    case _:
                        L.e("Shouldn't get here")
                        break

        # ...

    @staticmethod
    def generate_and_make_flac(
        index: int,
        state: State,
        is_retry: bool=False
    ) -> str:
        """
        Generates temp wav file, converts to flac, deletes temp wav
        Prints info
        Returns file path on success, else empty string

        # TODO: should return (path, error), mutually exclusive
        """

        if not state.project.has_voice:
            return ""

        temp_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.project.text_segments[index]

        start_time = time.time()

        if Shared.is_oute():
            is_success = GenerateUtil.generate_wav_file_oute(
                temp_wav_path,
                text_segment.text,
                state.project.oute_voice_json,
                state.project.oute_temperature)
        else:
            is_success = GenerateUtil.generate_wav_file_chatterbox(
                temp_wav_path,
                text_segment.text,
                state.project)

        if not is_success:
            delete_temp_file(temp_wav_path)
            return ""

        # Print time info
        elapsed = time.time() - start_time
        audio_seconds = estimated_wav_seconds(temp_wav_path)
        s = f"Audio duration: {COL_ACCENT}{audio_seconds:.1f}s{COL_DEFAULT}, inference time: {COL_ACCENT}{elapsed:.1f}s"
        if elapsed > 0:
            multi = audio_seconds / elapsed
            s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
        printt(s)

        # Normalize
        if state.prefs.should_normalize:
            printt("Performing loudness normalization")
            temp_normalized_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
            err = LoudnessNormalizationUtil.normalize(temp_wav_path, temp_normalized_wav_path)
            if err:
                delete_temp_file(temp_wav_path)
                delete_temp_file(temp_normalized_wav_path)
                return ""
        else:
            temp_normalized_wav_path = ""

        src_wav_path = temp_normalized_wav_path if temp_normalized_wav_path else temp_wav_path
        is_success = SoundFileUtil.encode_to_flac(src_wav_path, flac_path)

        if not is_success:
            delete_temp_file(temp_wav_path)
            if temp_normalized_wav_path:
                delete_temp_file(temp_normalized_wav_path)
            return ""

        if not is_retry:
            printt(f"Saved: {flac_path}")

        delete_temp_file(temp_wav_path)
        if temp_normalized_wav_path:
            delete_temp_file(temp_normalized_wav_path)

        # TODO: this should get triggered _after_ regenerate/fix, but can't rly unless much refactor ugh
        if state.prefs.play_on_generate:
            SoundFileUtil.play_flac_async(flac_path)

        return flac_path

    @staticmethod
    def generate_wav_file_oute(
        dest_file_path: str,
        prompt: str,
        voice: dict,
        temperature: float = -1
    ) -> bool:

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
            output.save(dest_file_path)
        except Exception as e:
            printt(f"Oute model error: {e}\a", "error")
            return False
        return True

    @staticmethod
    def generate_wav_file_chatterbox(
        dest_file_path: str,
        prompt: str,
        project: Project
    ) -> bool:

        import torchaudio as ta
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
            wav = chatterbox.generate(prompt, **d)
            ta.save(dest_file_path, wav, chatterbox.sr)
        except Exception as e:
            printt(f"Chatterbox model error: {e}\a", "error")
            return False
        return True

# ---

def print_item_heading(text: str, index: int, count: int, total: int) -> None:
    s  = f"{COL_ACCENT}[{COL_DEFAULT}{count+1}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
    s += f"{COL_ACCENT}Generating audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
    printt(s)
    printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
    printt()
