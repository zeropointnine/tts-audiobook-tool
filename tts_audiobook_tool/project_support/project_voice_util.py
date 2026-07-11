from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.constants import OUTE_DEFAULT_VOICE_JSON_FILE_PATH
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.oute_util import OuteUtil
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR, ellipsize_path_for_menu, printt

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Sound
    from tts_audiobook_tool.project import Project


class ProjectVoiceUtil:
    """
    Voice/model-specific helpers for `Project`.
    """

    @staticmethod
    def voice_values(value) -> list[str]:
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str) and item]
        return []

    @staticmethod
    def primary_voice_value(project: Project, attr: str) -> str:
        if not attr:
            return ""
        values = ProjectVoiceUtil.voice_values(getattr(project, attr, ""))
        return values[0] if values else ""

    @staticmethod
    def primary_voice_transcript(project: Project, attr: str) -> str:
        return ProjectVoiceUtil.primary_voice_value(project, attr)

    @staticmethod
    def current_voice_value(project: Project, attr: str, voice_rotation_index: int) -> str:
        if not attr:
            return ""
        values = ProjectVoiceUtil.voice_values(getattr(project, attr, ""))
        if not values:
            return ""
        return values[voice_rotation_index % len(values)]

    @staticmethod
    def make_voice_file_name_tag(voice_file_name: str, file_tag: str) -> str:
        if not voice_file_name:
            return "none"

        voice_file_name = Path(voice_file_name).stem
        postfix_decorator = "_" + file_tag
        if voice_file_name.endswith(postfix_decorator):
            voice_file_name = voice_file_name[:-len(postfix_decorator)]

        return app_text.sanitize_for_filename(voice_file_name[:30])

    @staticmethod
    def current_voice_reference_pair(
            project: Project,
            voice_attr: str,
            transcript_attr: str,
            voice_rotation_index: int,
    ) -> tuple[str, str]:
        voices = ProjectVoiceUtil.voice_values(getattr(project, voice_attr, ""))
        if not voices:
            return "", ""

        index = voice_rotation_index % len(voices)
        transcripts = ProjectVoiceUtil.voice_values(getattr(project, transcript_attr, "")) if transcript_attr else []
        transcript = transcripts[index] if index < len(transcripts) else ""
        return voices[index], transcript

    @staticmethod
    def voice_reference_pairs(project: Project, voice_attr: str, transcript_attr: str = "") -> list[tuple[str, str]]:
        voices = ProjectVoiceUtil.voice_values(getattr(project, voice_attr, ""))
        transcripts = ProjectVoiceUtil.voice_values(getattr(project, transcript_attr, "")) if transcript_attr else []
        return [(voice, transcripts[i] if i < len(transcripts) else "") for i, voice in enumerate(voices)]

    @staticmethod
    def load_oute_voice_json(project: Project) -> None:
        voice_path = os.path.join(project.dir_path, project.oute_voice_file_name)
        if not project.oute_voice_file_name or not os.path.exists(voice_path):
            result = OuteUtil.load_oute_voice_json(OUTE_DEFAULT_VOICE_JSON_FILE_PATH)
            if isinstance(result, str):
                from tts_audiobook_tool import ask
                ask.ask_error(result)
            else:
                project.set_oute_voice_and_save(result, "default")
        else:
            result = OuteUtil.load_oute_voice_json(voice_path)
            if isinstance(result, str):
                printt(f"Problem loading Oute voice json file {project.oute_voice_file_name}: {result}")
                printt()
            else:
                project.oute_voice_json = result

    @staticmethod
    def set_voice_and_save(
            project: Project,
            sound: Sound,
            voice_file_stem: str,
            transcript: str,
            tts_type: TtsModelType,
            is_secondary: bool=False,
            append: bool=False,
    ) -> str:
        dest_file_name = f"{voice_file_stem}_{tts_type.value.file_tag}.flac"
        dest_path = Path(project.dir_path) / dest_file_name

        err = SoundFileUtil.save_flac(sound, str(dest_path))
        if err:
            return err

        info = tts_type.value
        voice_file_name_attr = info.voice_target_attr
        voice_transcript_attr = info.voice_transcript_attr

        with project.batch():
            if tts_type == TtsModelType.INDEXTTS2 and is_secondary:
                project.indextts2_emo_voice_file_name = dest_file_name
            else:
                if not voice_file_name_attr:
                    raise Exception(f"Unsupported tts type {tts_type}")
                if append:
                    values = ProjectVoiceUtil.voice_values(getattr(project, voice_file_name_attr, ""))
                    setattr(project, voice_file_name_attr, values + [dest_file_name])
                else:
                    setattr(project, voice_file_name_attr, [dest_file_name])

            if voice_transcript_attr:
                if append:
                    transcripts = ProjectVoiceUtil.voice_values(getattr(project, voice_transcript_attr, ""))
                    setattr(project, voice_transcript_attr, transcripts + [transcript])
                else:
                    setattr(project, voice_transcript_attr, [transcript] if transcript else [])

            if tts_type == TtsModelType.POCKET:
                project.pocket_predefined_voice = ""

        return ""

    @staticmethod
    def remove_voice_at_index_and_save(project: Project, tts_type: TtsModelType, index: int) -> str:
        info = tts_type.value
        voice_file_name_attr = info.voice_target_attr
        voice_transcript_attr = info.voice_transcript_attr
        if not voice_file_name_attr:
            raise ValueError(f"Unsupported tts_type: {tts_type}")

        voices = ProjectVoiceUtil.voice_values(getattr(project, voice_file_name_attr, ""))
        if index < 0 or index >= len(voices):
            raise IndexError(f"Voice sample index out of range: {index}")

        removed = voices.pop(index)
        with project.batch():
            setattr(project, voice_file_name_attr, voices)

            if voice_transcript_attr:
                transcripts = ProjectVoiceUtil.voice_values(getattr(project, voice_transcript_attr, ""))
                if index < len(transcripts):
                    transcripts.pop(index)
                setattr(project, voice_transcript_attr, transcripts)

            if tts_type == TtsModelType.POCKET and not voices:
                project.pocket_predefined_voice = ""

        return removed

    @staticmethod
    def clear_voice_and_save(project: Project, tts_type: TtsModelType, is_secondary: bool=False) -> None:
        info = tts_type.value
        voice_file_name_attr = info.voice_target_attr
        voice_transcript_attr = info.voice_transcript_attr

        with project.batch():
            if tts_type == TtsModelType.INDEXTTS2 and is_secondary:
                project.indextts2_emo_voice_file_name = ""
            else:
                if not voice_file_name_attr:
                    raise ValueError(f"Unsupported tts_type: {tts_type}")
                setattr(project, voice_file_name_attr, [])

            if voice_transcript_attr:
                setattr(project, voice_transcript_attr, [])

            if tts_type == TtsModelType.POCKET:
                project.pocket_predefined_voice = ""

    @staticmethod
    def get_voice_label(project: Project) -> str:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelType.POCKET:
            if project.pocket_predefined_voice:
                return project.pocket_predefined_voice
            value = ProjectVoiceUtil.primary_voice_value(project, "pocket_voice_file_name")
            if not value:
                return "none"
            return ellipsize_path_for_menu(value.removesuffix("_pocket.flac"))

        value = ProjectVoiceUtil.primary_voice_value(project, Tts.get_type().value.voice_target_attr)
        if not value:
            return "none"
        value = value.removesuffix(f"_{Tts.get_type().value.file_tag}.flac")
        return ellipsize_path_for_menu(value)

    @staticmethod
    def has_voice(project: Project) -> bool:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelType.POCKET:
            return bool(project.pocket_predefined_voice or ProjectVoiceUtil.primary_voice_value(project, "pocket_voice_file_name"))
        value = ProjectVoiceUtil.primary_voice_value(project, Tts.get_type().value.voice_target_attr)
        return bool(value)

    @staticmethod
    def emo_vector_to_string(project: Project) -> str:
        if not project.indextts2_emo_vector or sum(project.indextts2_emo_vector) == 0:
            return "none"
        strings = []
        for item in project.indextts2_emo_vector:
            string = f"{item:.1f}".replace(".0", "")
            strings.append(string)
        return ",".join(strings)

    @staticmethod
    def verify_voice_files_exist(project: Project) -> bool:
        from tts_audiobook_tool.tts import Tts
        model_type = Tts.get_type()
        info = model_type.value

        attribs: list[str] = []
        if info.voice_target_attr:
            attribs.append(info.voice_target_attr)

        if model_type == TtsModelType.INDEXTTS2:
            attribs.append("indextts2_emo_voice_file_name")

        attribs = [attrib for attrib in attribs if attrib.endswith("_voice_file_name")]
        if not attribs:
            return False

        warnings = []
        for attrib in attribs:
            file_names = ProjectVoiceUtil.voice_values(getattr(project, attrib, ""))
            valid_file_names = []
            for file_name in file_names:
                file_path = os.path.join(project.dir_path, file_name)
                if not os.path.exists(file_path):
                    warnings.append((attrib, file_name, "file not found"))
                    continue

                err = SoundFileUtil.is_valid_sound_file(file_path)
                if err:
                    warnings.append((attrib, file_name, err))
                    continue

                valid_file_names.append(file_name)

            if len(valid_file_names) != len(file_names):
                setattr(project, attrib, valid_file_names)

        if warnings:
            printt(f"{COL_ERROR}Warning/info: {COL_DEFAULT}Problem with saved voice clone file(s) for current model {COL_ACCENT}{info.ui['proper_name']}{COL_DEFAULT}")
            for attrib, file_name, reason in warnings:
                printt(f"- {COL_ACCENT}{attrib}{COL_DEFAULT}: {file_name}")
                printt(f"  {COL_DIM}{reason}{COL_DEFAULT}")
            printt("Clearing saved reference(s) and continuing.")
            printt()

        return bool(warnings)

    @staticmethod
    def get_batch_size(project: Project) -> int:
        from tts_audiobook_tool.tts import Tts
        field = Tts.get_type().value.batch_size_attr
        if not field:
            return 1
        if not hasattr(project, field):
            raise ValueError(f"Unrecognized attribute {field}")
        value = getattr(project, field)
        if value == -1:
            value = PROJECT_BATCH_SIZE_DEFAULT
        elif value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        return value

    @staticmethod
    def set_batch_size(project: Project, value: int) -> None:
        from tts_audiobook_tool.tts import Tts
        field = Tts.get_type().value.batch_size_attr
        if not field:
            raise ValueError(f"No support for batch_size for the current model")
        if not hasattr(project, field):
            raise ValueError(f"Unrecognized attribute {field}")
        if value > PROJECT_BATCH_SIZE_MAX:
            value = PROJECT_BATCH_SIZE_MAX
        setattr(project, field, value)

    @staticmethod
    def is_language_cjk(project: Project) -> bool:
        if project.language_code in ["zh", "ja", "ko"]:
            return True
        if project.language_code.startswith(("zh-", "ja-", "ko-")):
            return True
        return False
