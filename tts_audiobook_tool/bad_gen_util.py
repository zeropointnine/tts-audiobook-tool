
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from .util import *

class BadGenUtil:

    @staticmethod
    def ask_delete(state: State):
        """
        Check for audio generations that are abnormally long relative to its character count, and asks to delete.
        """
        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)

        # Parallel lists meh
        text_segments = []
        durations = []
        file_paths = []

        for index, file_path in dic.items():
            text_segment = state.text_segments[index]
            seconds = AppUtil.get_flac_file_duration(file_path)
            b = isinstance(seconds, float) and seconds >= 1.0 and len(text_segment) >= 30 # TODO length filter not ideal but yea
            if b:
                file_paths.append(file_path)
                text_segments.append(text_segment)
                durations.append(seconds)

        if len(text_segments) < 50:
            ask("Need about 50+ audio segments to do evaluation. Press enter: ")
            return

        total_chars = 0
        total_duration = 0.0
        for i in range(0, len(text_segments)):
            total_chars += len(text_segments[i])
            total_duration += durations[i]
        avg_spc = total_duration / total_chars

        def get_multiplier(i: int) -> float:
            text_segment = text_segments[i]
            seconds = durations[i]
            num_chars = len(text_segment)
            spc = seconds / num_chars
            return spc / avg_spc

        sus_indices = []
        for i in range(0, len(text_segments)):
            multiplier = get_multiplier(i)
            if multiplier > 2:
                sus_indices.append(i)

        if not sus_indices:
            ask("Nothing glaringly sus found. Press enter: ")
            return

        printt(f"{len(sus_indices)} suspected bad gens found:\n")

        for i in sus_indices:
            printt(f"{get_multiplier(i):.2f}x  {file_paths[i]}")
        printt()

        b = ask_confirm(f"Press {make_hotkey_string("Y")} to delete these files (You will need to regenerate them): ")
        if b:
            bad_count = 0
            for i in sus_indices:
                file_path = file_paths[i]
                try:
                    Path(file_path).unlink()
                except:
                    bad_count += 1
            s = "Done." if not bad_count else f"Couldn't delete {bad_count} of {len(sus_indices)} files."
            s += " Press enter to continue: "
            ask(s)
