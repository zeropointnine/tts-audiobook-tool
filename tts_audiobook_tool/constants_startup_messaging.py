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

STARTUP_VERSION_CODE = 1

# (version code, message), ascending by version code
STARTUP_VERSION_MESSAGES: list[tuple[int, str]] = [

]
