from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs
from tts_audiobook_tool.tts_models.tts_model_info import TtsModelInfos


class VoiceMossShared:

    @staticmethod
    def append_voice_items(items: list[MenuItem], state: State) -> None:
        items.append(
            MenuItem(
                VoiceMenuShared.make_resolved_voice_label,
                lambda _, __: VoiceMenuShared.ask_and_set_voice_file(state, TtsModelInfos.MOSS)
            )
        )
        if state.project.moss_voice_file_name:
            items.append(VoiceMenuShared.make_clear_voice_item(state, TtsModelInfos.MOSS))

    @staticmethod
    def get_temperature_attr(arch_type: MossConfigs) -> str:
        return "moss_local_temperature" if arch_type == MossConfigs.LOCAL else "moss_delay_temperature"

    @staticmethod
    def get_top_p_attr(arch_type: MossConfigs) -> str:
        return "moss_local_top_p" if arch_type == MossConfigs.LOCAL else "moss_delay_top_p"

    @staticmethod
    def get_top_k_attr(arch_type: MossConfigs) -> str:
        return "moss_local_top_k" if arch_type == MossConfigs.LOCAL else "moss_delay_top_k"

    @staticmethod
    def make_temperature_item(state: State, arch_type: MossConfigs) -> MenuItem:
        arch_values = arch_type.value
        return VoiceMenuShared.make_temperature_item(
            state=state,
            attr=VoiceMossShared.get_temperature_attr(arch_type),
            base_label=f"{arch_values.arch_name} temperature",
            default_value=arch_values.temperature_default,
            min_value=arch_values.temperature_min,
            max_value=arch_values.temperature_max,
            hint=HINT_MOSS_TEMPERATURE,
        )

    @staticmethod
    def make_audio_top_p_item(state: State, arch_type: MossConfigs) -> MenuItem:

        arch_values = arch_type.value

        return MenuUtil.make_number_item(
            state=state,
            attr=VoiceMossShared.get_top_p_attr(arch_type),
            base_label=f"{arch_values.arch_name} audio top-p",
            default_value=arch_values.audio_top_p_default,
            is_minus_one_default=True,
            num_decimals=2,
            prompt=f"Enter Audio top-p",
            min_value=arch_values.audio_top_p_min,
            max_value=arch_values.audio_top_p_max
        )

    @staticmethod
    def make_audio_top_k_item(state: State, arch_type: MossConfigs) -> MenuItem:

        arch_values = arch_type.value

        return MenuUtil.make_number_item(
            state=state,
            attr=VoiceMossShared.get_top_k_attr(arch_type),
            base_label=f"{arch_values.arch_name} audio top-k",
            default_value=arch_values.audio_top_k_default,
            is_minus_one_default=True,
            num_decimals=0,
            prompt=f"Enter Audio top-k",
            min_value=arch_values.audio_top_k_min,
            max_value=arch_values.audio_top_k_max
        )
