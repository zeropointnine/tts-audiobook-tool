from tts_audiobook_tool.menus.voice.voice_moss_shared import VoiceMossShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs


def test_moss_menu_setting_refs_use_variant_config():
    assert VoiceMossShared.get_temperature_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_temperature")
    assert VoiceMossShared.get_top_p_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_top_p")
    assert VoiceMossShared.get_top_k_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_top_k")
    assert VoiceMossShared.get_seed_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_seed")
    assert VoiceMossShared.get_temperature_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_temperature")
    assert VoiceMossShared.get_top_p_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_top_p")
    assert VoiceMossShared.get_top_k_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_top_k")
    assert VoiceMossShared.get_seed_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_seed")
    assert VoiceMossShared.get_temperature_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_temperature")
    assert VoiceMossShared.get_top_p_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_top_p")
    assert VoiceMossShared.get_top_k_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_top_k")
    assert VoiceMossShared.get_seed_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_seed")
