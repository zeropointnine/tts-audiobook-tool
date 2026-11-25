"""Per-voice speaking-rate override: Disabled, or a 1-based discrete bucket."""
from __future__ import annotations

from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import make_menu_label, print_feedback, printt

LABEL_SPEAKING_RATE = "Speaking-rate override"
DISABLED = "Disabled"


def make_speaking_rate_items(state: State, index: int, model_id: str, bucket_count: int) -> list[MenuItem]:
    voices = state.project.voice_references
    if not 0 <= index < len(voices):
        return []
    bucket = voices[index].speaking_rate_buckets.get(model_id)
    value = f"Bucket {bucket}" if bucket is not None else DISABLED
    return [MenuItem(
        make_menu_label(LABEL_SPEAKING_RATE, value, DISABLED),
        lambda s, _: ask_speaking_rate_bucket(s, index, model_id, bucket_count),
    )]


def ask_speaking_rate_bucket(state: State, index: int, model_id: str, bucket_count: int) -> None:
    current = state.project.voice_references[index].speaking_rate_buckets.get(model_id)
    printt(f"Enter speaking-rate bucket, from 1 (slowest) to {bucket_count} (fastest), or 0 for Disabled:")
    value = ask.ask_input(prefill=str(current) if current is not None else "0").strip()
    if not value:
        return
    try:
        number = int(value)
    except ValueError:
        ask.ask_error("Enter a whole-number bucket, or 0 for Disabled")
        return
    if not 0 <= number <= bucket_count:
        ask.ask_error(f"Enter a bucket from 1 to {bucket_count}, or 0 for Disabled")
        return
    bucket = number or None
    error = ProjectVoiceUtil.set_voice_speaking_rate_bucket_at_index_and_save(
        state.project, index, model_id, bucket)
    if error:
        ask.ask_error(error)
    elif bucket != current:
        print_feedback(f"{LABEL_SPEAKING_RATE} set to: {f'Bucket {bucket}' if bucket is not None else DISABLED}")
