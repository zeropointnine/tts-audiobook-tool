"""
Startup messaging: tells users about changes made since the last time they ran the app.

To announce a change:
  1. Increment STARTUP_VERSION_CODE
  2. Append a (new_code, message) tuple to STARTUP_VERSION_MESSAGES

At startup, messages whose code is greater than the code stored in Prefs
(`startup_version_code`) are shown, then the stored code is updated.
Users with no stored code (brand-new users, or users updating to the first
version that has this feature) are not shown any messages.
"""

from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT

STARTUP_VERSION_CODE = 6

# (version code, message), ascending by version code
STARTUP_VERSION_MESSAGES: list[tuple[int, str]] = [
    (
        2,
        f"Room reverb has been moved to the {COL_ACCENT}Voice{COL_DEFAULT} menu,\n"
        "and can now be controlled on a per-voice basis."
    ),
    (
        3,
        f"The \"Voice selection mode\" setting has been removed (including \"auto-advance\").\n"
        "Text lines now always use their assigned voice. "
    ),
    (
        4,
        f"Voices > Auto-identify characters:\n"
        "Automatically identify speaking lines by character using LLM."
    ),
    (
        5,
        "Improved loudness normalization: \"Create > Loudness\" now defaults\n"
        "to \"Normal\", and the whole-file normalization pass has been removed."
    ),
    (
        6,
        "Added per-voice volume control."
    )
]
