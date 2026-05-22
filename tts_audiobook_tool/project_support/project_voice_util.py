from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tts_audiobook_tool.constants_config import PROJECT_BATCH_SIZE_DEFAULT, PROJECT_BATCH_SIZE_MAX
from tts_audiobook_tool.constants import OUTE_DEFAULT_VOICE_JSON_FILE_PATH
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.tts_models.oute_util import OuteUtil
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import COL_ACCENT, COL_DEFAULT, COL_DIM, COL_ERROR, ellipsize_path_for_menu, printt

if TYPE_CHECKING:
    from tts_audiobook_tool.app_types import Sound
    from tts_audiobook_tool.project import Project


class ProjectVoiceUtil:
    """
    Voice/model-specific helpers for `Project`.
    """

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
            tts_type: TtsModelInfos,
            is_secondary: bool=False,
    ) -> str:
        dest_file_name = f"{voice_file_stem}_{tts_type.value.file_tag}.flac"
        dest_path = Path(project.dir_path) / dest_file_name

        err = SoundFileUtil.save_flac(sound, str(dest_path))
        if err:
            return err

        info = tts_type.value
        voice_file_name_attr = info.voice_file_name_attr
        voice_transcript_attr = info.voice_transcript_attr

        with project.batch():
            if tts_type == TtsModelInfos.INDEXTTS2 and is_secondary:
                project.indextts2_emo_voice_file_name = dest_file_name
            else:
                if not voice_file_name_attr:
                    raise Exception(f"Unsupported tts type {tts_type}")
                setattr(project, voice_file_name_attr, dest_file_name)

            if voice_transcript_attr:
                setattr(project, voice_transcript_attr, transcript)

            if tts_type == TtsModelInfos.POCKET:
                project.pocket_predefined_voice = ""

        return ""

    @staticmethod
    def clear_voice_and_save(project: Project, tts_type: TtsModelInfos, is_secondary: bool=False) -> None:
        info = tts_type.value
        voice_file_name_attr = info.voice_file_name_attr
        voice_transcript_attr = info.voice_transcript_attr

        with project.batch():
            if tts_type == TtsModelInfos.INDEXTTS2 and is_secondary:
                project.indextts2_emo_voice_file_name = ""
            else:
                if not voice_file_name_attr:
                    raise ValueError(f"Unsupported tts_type: {tts_type}")
                setattr(project, voice_file_name_attr, "")

            if voice_transcript_attr:
                setattr(project, voice_transcript_attr, "")

            if tts_type == TtsModelInfos.POCKET:
                project.pocket_predefined_voice = ""

    @staticmethod
    def get_voice_label(project: Project) -> str:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelInfos.POCKET:
            if project.pocket_predefined_voice:
                return project.pocket_predefined_voice
            value = project.pocket_voice_file_name
            if not value:
                return "none"
            return ellipsize_path_for_menu(value.removesuffix("_pocket.flac"))

        value = getattr(project, Tts.get_type().value.voice_file_name_attr, "")
        if not value:
            return "none"
        value = value.removesuffix(f"_{Tts.get_type().value.file_tag}.flac")
        return ellipsize_path_for_menu(value)

    @staticmethod
    def has_voice(project: Project) -> bool:
        from tts_audiobook_tool.tts import Tts
        if Tts.get_type() == TtsModelInfos.POCKET:
            return bool(project.pocket_predefined_voice or project.pocket_voice_file_name)
        value = getattr(project, Tts.get_type().value.voice_file_name_attr, "")
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
        if info.voice_file_name_attr:
            attribs.append(info.voice_file_name_attr)

        if model_type == TtsModelInfos.INDEXTTS2:
            attribs.append("indextts2_emo_voice_file_name")

        attribs = [attrib for attrib in attribs if attrib.endswith("_voice_file_name")]
        if not attribs:
            return False

        warnings = []
        for attrib in attribs:
            file_name = getattr(project, attrib, "")
            if not file_name:
                continue

            file_path = os.path.join(project.dir_path, file_name)
            if not os.path.exists(file_path):
                warnings.append((attrib, file_name, "file not found"))
                setattr(project, attrib, "")
                continue

            err = SoundFileUtil.is_valid_sound_file(file_path)
            if err:
                warnings.append((attrib, file_name, err))
                setattr(project, attrib, "")

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
        field = Tts.get_type().value.batch_size_project_field
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
        field = Tts.get_type().value.batch_size_project_field
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