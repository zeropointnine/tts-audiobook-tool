from tts_audiobook_tool import ask
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.constants_hints import HINT_MOSS_TEMPERATURE
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel, MossConfigs
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class VoiceMossShared:

    @staticmethod
    def append_voice_items(
            items: list[MenuItem], state: State, model_type: TtsModelType
    ) -> None:
        items.extend(
            VoiceMenuShared.make_voice_sample_items(state, model_type)
        )

    @staticmethod
    def get_setting_prefix(arch_type: MossConfigs) -> str:
        return arch_type.value.setting_prefix

    @staticmethod
    def get_temperature_ref(arch_type: MossConfigs) -> SettingRef:
        return SettingRef("moss_local", f"{VoiceMossShared.get_setting_prefix(arch_type)}_temperature")

    @staticmethod
    def get_top_p_ref(arch_type: MossConfigs) -> SettingRef:
        return SettingRef("moss_local", f"{VoiceMossShared.get_setting_prefix(arch_type)}_top_p")

    @staticmethod
    def get_top_k_ref(arch_type: MossConfigs) -> SettingRef:
        return SettingRef("moss_local", f"{VoiceMossShared.get_setting_prefix(arch_type)}_top_k")

    @staticmethod
    def get_seed_ref(arch_type: MossConfigs) -> SettingRef:
        return SettingRef("moss_local", f"{VoiceMossShared.get_setting_prefix(arch_type)}_seed")

    @staticmethod
    def make_temperature_item(state: State, arch_type: MossConfigs) -> MenuItem:
        arch_values = arch_type.value
        target = VoiceMossShared.get_temperature_ref(arch_type)
        label = MenuUtil.make_number_label(
            project=state.project,
            target=target,
            base_label=f"{arch_values.arch_name} temperature",
            default_value=arch_values.temperature_default,
            num_decimals=2,
        )

        def edit(current: State, _: MenuItem) -> None:
            hints.show_hint_if_necessary(current.prefs, HINT_MOSS_TEMPERATURE)
            temperature = MossBaseModel.get_generation_params(current.project, arch_type)[0]
            ask.ask_number_and_save(
                saveable=current.project,
                target=target,
                prompt=f"Enter {arch_values.arch_name} temperature",
                min_value=arch_values.temperature_min,
                max_value=arch_values.temperature_max,
                default_value=arch_values.temperature_default,
                success_prefix="Value set:",
                prefill_value=temperature,
            )

        return MenuItem(label, edit)

    @staticmethod
    def make_audio_top_p_item(state: State, arch_type: MossConfigs) -> MenuItem:

        arch_values = arch_type.value

        return MenuUtil.make_number_item(
            state=state,
            target=VoiceMossShared.get_top_p_ref(arch_type),
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
            target=VoiceMossShared.get_top_k_ref(arch_type),
            base_label=f"{arch_values.arch_name} audio top-k",
            default_value=arch_values.audio_top_k_default,
            is_minus_one_default=True,
            num_decimals=0,
            prompt=f"Enter Audio top-k",
            min_value=arch_values.audio_top_k_min,
            max_value=arch_values.audio_top_k_max
        )
