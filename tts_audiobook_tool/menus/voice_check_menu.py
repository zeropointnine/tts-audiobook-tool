from tts_audiobook_tool import ask
from tts_audiobook_tool.menus.menu_util import MenuItem, MenuUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.voice_check import VoiceCheck


class VoiceCheckMenu:

    @staticmethod
    def make_menu_item(state: State) -> MenuItem:
        return MenuItem(
            make_menu_label("Voice similarity check", VoiceCheck.is_enabled()),
            lambda _, __: VoiceCheckMenu.menu(state)
        )

    @staticmethod
    def print_confirmation_line() -> None:
        """ For the generate start screen """
        if VoiceCheck.is_enabled():
            printt(f"- Voice similarity check: {COL_OK}Enabled{COL_DEFAULT} (Qwen3 voice clone)")

    @staticmethod
    def make_description() -> str:
        return (
            f"Qwen3-TTS voice clone only. Compares the voice of each generated segment\n"
            f"with the voice sample (speaker embedding similarity, 0-1). If the similarity\n"
            f"is below the threshold, the line is generated again.\n"
            f"Short segments have lower similarity values, so the threshold depends on\n"
            f"the segment duration (plus the threshold offset):\n"
            f"  {VoiceCheckMenu._make_threshold_table_string()}\n"
            f"- From attempt 2 on, the temperature is lowered by {VoiceCheck.RETRY_TEMPERATURE_DELTA:.1f} per attempt,\n"
            f"  down to the min temperature. The project's temperature setting is not changed.\n"
            f"  Short lines have a higher minimum (letters/digits: {VoiceCheck._make_length_rule_string()}),\n"
            f"  since low temperatures can make their generation run away until the\n"
            f"  generation timeout, which stops the run.\n"
            f"- Once the temperature is at the minimum, the threshold is lowered by "
            f"{VoiceCheck.THRESHOLD_STEP:.3f}\n"
            f"  for each further attempt.\n"
            f"- After the max number of attempts, the line is tagged as failed.\n"
            f"While enabled, the generation length is limited according to the text length,\n"
            f"so that a generation that does not end cannot run into the generation timeout.\n"
            f"Works with or without speech-to-text validation.\n"
        )

    @staticmethod
    def menu(state: State) -> None:

        def item_maker(_: State) -> list[MenuItem]:
            return [
                MenuItem(
                    lambda _: make_menu_label("Enabled", VoiceCheck.is_enabled(), False),
                    lambda _, __: VoiceCheckMenu.enabled_menu(state)
                ),
                MenuItem(
                    lambda _: make_menu_label(
                        "Threshold offset", VoiceCheck.get_threshold_offset(),
                        VoiceCheck.THRESHOLD_OFFSET_DEFAULT, num_decimals=3
                    ),
                    lambda _, __: VoiceCheckMenu.ask_threshold_offset()
                ),
                MenuItem(
                    lambda _: make_menu_label(
                        "Max attempts", VoiceCheck.get_max_attempts(), VoiceCheck.MAX_ATTEMPTS_DEFAULT
                    ),
                    lambda _, __: VoiceCheckMenu.ask_max_attempts()
                ),
                MenuItem(
                    lambda _: make_menu_label(
                        f"Min temperature (lines >= {VoiceCheck.RETRY_TEMPERATURE_MIN_BY_LENGTH[-1][0]} letters)", VoiceCheck.get_min_temperature(),
                        VoiceCheck.RETRY_TEMPERATURE_MIN_DEFAULT, num_decimals=1
                    ),
                    lambda _, __: VoiceCheckMenu.ask_min_temperature()
                ),
            ]

        MenuUtil.menu(
            state,
            "Voice similarity check",
            item_maker,
            subheading=lambda _: VoiceCheckMenu.make_description(),
            breadcrumb="Voice similarity check",
        )

    @staticmethod
    def enabled_menu(state: State) -> None:

        def on_select(value: bool) -> None:
            if VoiceCheck.is_enabled() != value:
                err = VoiceCheck.set_enabled(value)
                if err:
                    printt(f"{COL_ERROR}{err}")
            print_feedback(f"Set to:", str(VoiceCheck.is_enabled()))

        MenuUtil.options_menu(
            state=state,
            heading_text="Voice similarity check",
            labels=["True", "False"],
            values=[True, False],
            current_value=VoiceCheck.is_enabled(),
            default_value=False,
            on_select=on_select
        )

    @staticmethod
    def _make_threshold_table_string() -> str:
        """ Eg: "<1.0s: 0.947, <1.5s: 0.960, ..., >=5.0s: 0.978" (offset applied) """
        offset = VoiceCheck.get_threshold_offset()
        parts = []
        previous = 0.0
        for max_duration, _ in VoiceCheck.THRESHOLD_BY_DURATION:
            threshold = VoiceCheck.get_duration_threshold(previous, offset)
            label = f"<{max_duration:.1f}s" if max_duration != float("inf") else f">={previous:.1f}s"
            parts.append(f"{label}: {threshold:.3f}")
            previous = max_duration
        return ", ".join(parts)

    @staticmethod
    def ask_threshold_offset() -> None:
        value = VoiceCheckMenu._ask_number(
            "Threshold offset (added to all duration thresholds)",
            VoiceCheck.get_threshold_offset(),
            VoiceCheck.THRESHOLD_OFFSET_MIN, VoiceCheck.THRESHOLD_OFFSET_MAX,
            VoiceCheck.THRESHOLD_OFFSET_DEFAULT,
            is_int=False,
        )
        if value is None:
            return
        err = VoiceCheck.set_threshold_offset(value)
        if err:
            ask.ask_error(err)
            return
        print_feedback("Threshold offset set to:", f"{VoiceCheck.get_threshold_offset():+.3f}")

    @staticmethod
    def ask_max_attempts() -> None:
        value = VoiceCheckMenu._ask_number(
            "Max attempts",
            VoiceCheck.get_max_attempts(),
            VoiceCheck.MAX_ATTEMPTS_MIN, VoiceCheck.MAX_ATTEMPTS_MAX, VoiceCheck.MAX_ATTEMPTS_DEFAULT,
            is_int=True,
        )
        if value is None:
            return
        err = VoiceCheck.set_max_attempts(int(value))
        if err:
            ask.ask_error(err)
            return
        print_feedback("Max attempts set to:", str(VoiceCheck.get_max_attempts()))

    @staticmethod
    def ask_min_temperature() -> None:
        value = VoiceCheckMenu._ask_number(
            "Min temperature",
            VoiceCheck.get_min_temperature(),
            VoiceCheck.RETRY_TEMPERATURE_MIN_MIN, VoiceCheck.RETRY_TEMPERATURE_MIN_MAX,
            VoiceCheck.RETRY_TEMPERATURE_MIN_DEFAULT,
            is_int=False,
        )
        if value is None:
            return
        err = VoiceCheck.set_min_temperature(value)
        if err:
            ask.ask_error(err)
            return
        print_feedback("Min temperature set to:", f"{VoiceCheck.get_min_temperature():.2f}")

    @staticmethod
    def _ask_number(
        prompt: str,
        current: float,
        min_value: float,
        max_value: float,
        default_value: float,
        is_int: bool,
    ) -> float | None:
        """ Returns the entered value, or None if unchanged, empty or invalid """
        prefill = str(int(current)) if is_int else f"{current:.3f}"
        printt(f"{prompt}: {COL_DIM}(valid range: {min_value} to {max_value}; default: {default_value})")
        string = ask.ask_input(prefill=prefill)
        if not string or string == prefill:
            return None
        try:
            value = int(string) if is_int else float(string)
        except ValueError:
            ask.ask_error("Bad value")
            return None
        if not (min_value <= value <= max_value):
            ask.ask_error("Out of range")
            return None
        return value
