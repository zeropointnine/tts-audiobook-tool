from tts_audiobook_tool.menus.model.model_moss_shared import ModelMossShared
from tts_audiobook_tool.project_support.model_settings import SettingRef
from tts_audiobook_tool.tts_models.moss_base_model import MossConfigs


def test_moss_menu_setting_refs_use_variant_config():
    assert ModelMossShared.get_temperature_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_temperature")
    assert ModelMossShared.get_top_p_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_top_p")
    assert ModelMossShared.get_top_k_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_top_k")
    assert ModelMossShared.get_seed_ref(MossConfigs.DELAY) == SettingRef("moss_local", "delay_seed")
    assert ModelMossShared.get_temperature_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_temperature")
    assert ModelMossShared.get_top_p_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_top_p")
    assert ModelMossShared.get_top_k_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_top_k")
    assert ModelMossShared.get_seed_ref(MossConfigs.LOCAL) == SettingRef("moss_local", "local_seed")
    assert ModelMossShared.get_temperature_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_temperature")
    assert ModelMossShared.get_top_p_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_top_p")
    assert ModelMossShared.get_top_k_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_top_k")
    assert ModelMossShared.get_seed_ref(MossConfigs.LOCAL_V15) == SettingRef("moss_local", "local_v15_seed")
