from __future__ import annotations

import re
import subprocess
import sys
from typing import cast

import sounddevice as sd

from tts_audiobook_tool.util import make_error_string


class SoundInputDeviceInfo:
    """
    Convenience class for user-facing description of the sound input device.

    The base source of truth is sounddevice / PortAudio. Platform-specific
    helpers are used only to improve the displayed name of the default input
    device when the OS can provide a friendlier label:
    - Linux: pactl / wpctl
    - macOS: SwitchAudioSource, if available

    On Windows we instead pick a preferred host API (WASAPI > DirectSound >
    WDM-KS > MME) and use its default input device, since MME's default is the
    virtual "Microsoft Sound Mapper - Input" rather than the real hardware mic.

    All OS-specific lookups are best-effort and must fail safely so microphone
    availability and app behavior continue to work even when those commands are
    unavailable.
    """

    @staticmethod
    def has_input_device() -> bool:
        try:
            return SoundInputDeviceInfo._get_selected_input_device()[0] is not None
        except Exception:
            return False

    @staticmethod
    def get_check_error() -> str:
        try:
            if SoundInputDeviceInfo.has_input_device():
                return ""
            devices = sd.query_devices()
            if not devices:
                return "No audio devices found."
            return "No microphone / input audio device is available."
        except Exception as e:
            return f"Couldn't verify microphone input device: {make_error_string(e)}"

    @staticmethod
    def get_input_device_description() -> str:
        try:
            selected = SoundInputDeviceInfo._get_selected_input_device()
            device_idx, device_info, source = selected
            if device_info is None:
                return "unavailable"

            name = SoundInputDeviceInfo._get_preferred_input_device_name(
                device_info=device_info,
                source=source,
            )
            hostapi_name = SoundInputDeviceInfo._get_hostapi_name(device_info)
            return f"{name} (device index {device_idx}, host API {hostapi_name}) ({source})"
        except Exception as e:
            return f"unable to query details ({make_error_string(e)})"

    @staticmethod
    def _get_selected_input_device() -> tuple[int | None, dict[str, object] | None, str]:

        devices = sd.query_devices()
        if not devices:
            return None, None, "unavailable"

        default_input = SoundInputDeviceInfo._get_default_input_device_index()
        if default_input is not None and default_input >= 0:
            candidate = sd.query_devices(default_input)
            if SoundInputDeviceInfo._is_input_device(candidate):
                return default_input, cast(dict[str, object], candidate), "default"

        for i, candidate in enumerate(devices):
            if SoundInputDeviceInfo._is_input_device(candidate):
                return i, cast(dict[str, object], candidate), "detected"

        return None, None, "unavailable"

    @staticmethod
    def _get_default_input_device_index() -> int | None:
        if sys.platform == "win32":
            idx = SoundInputDeviceInfo._get_windows_preferred_default_input_index()
            if idx is not None:
                return idx

        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)) and default_device:
            default_input = default_device[0]
            return default_input if isinstance(default_input, int) else None
        return None

    @staticmethod
    def _get_windows_preferred_default_input_index() -> int | None:
        preferred_order = ["Windows WASAPI", "Windows DirectSound", "Windows WDM-KS", "MME"]
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            return None

        by_name: dict[str, dict[str, object]] = {}
        for api in hostapis:
            if isinstance(api, dict):
                api_name = api.get("name")
                if isinstance(api_name, str):
                    by_name[api_name] = cast(dict[str, object], api)

        for name in preferred_order:
            api = by_name.get(name)
            if api is None:
                continue
            idx = api.get("default_input_device")
            if not isinstance(idx, int) or idx < 0:
                continue
            try:
                info = sd.query_devices(idx)
            except Exception:
                continue
            if SoundInputDeviceInfo._is_input_device(info):
                return idx
        return None

    @staticmethod
    def _is_input_device(device_info: object) -> bool:
        return isinstance(device_info, dict) and device_info.get("max_input_channels", 0) > 0

    @staticmethod
    def _get_preferred_input_device_name(
        device_info: dict[str, object],
        source: str,
    ) -> str:
        if source == "default":
            os_default_name = SoundInputDeviceInfo._get_os_default_input_device_name()
            if os_default_name:
                return os_default_name
        return str(device_info.get("name", "Unknown device"))

    @staticmethod
    def _get_os_default_input_device_name() -> str | None:
        if sys.platform == "linux":
            return SoundInputDeviceInfo._get_linux_default_source_display_name()
        if sys.platform == "darwin":
            return SoundInputDeviceInfo._get_macos_default_input_device_name()
        return None

    @staticmethod
    def _get_linux_default_source_display_name() -> str | None:

        default_source_lines = SoundInputDeviceInfo._run_command_lines(["pactl", "get-default-source"])
        if default_source_lines:
            default_source_name = default_source_lines[0].strip()
            if default_source_name:
                info_lines = SoundInputDeviceInfo._run_command_lines(["pactl", "list", "sources"])
                current_name = ""
                for line in info_lines:
                    stripped = line.strip()
                    if stripped.startswith("Name: "):
                        current_name = stripped.removeprefix("Name: ").strip()
                        continue
                    if current_name == default_source_name and stripped.startswith("Description: "):
                        description = stripped.removeprefix("Description: ").strip()
                        if description:
                            return description

        status_lines = SoundInputDeviceInfo._run_command_lines(["wpctl", "status"])
        in_sources = False
        for line in status_lines:
            stripped = line.strip()
            if stripped == "├─ Sources:" or stripped == "└─ Sources:":
                in_sources = True
                continue
            if in_sources and (stripped.startswith("├─ ") or stripped.startswith("└─ ")):
                break
            if in_sources and stripped.startswith("*"):
                match = re.match(r"\*\s+\d+\.\s+(.*?)\s+\[", stripped)
                if match:
                    name = match.group(1).strip()
                    if name:
                        return name

        return None

    @staticmethod
    def _get_macos_default_input_device_name() -> str | None:
        lines = SoundInputDeviceInfo._run_command_lines(["SwitchAudioSource", "-t", "input", "-c"])
        for line in lines:
            name = line.strip()
            if name:
                return name
        return None

    @staticmethod
    def _get_hostapi_name(device_info: dict[str, object]) -> str:
        hostapi_idx = device_info.get("hostapi")
        hostapi_name = "Unknown host API"
        if isinstance(hostapi_idx, int):
            try:
                hostapi_info = sd.query_hostapis(hostapi_idx)
                if isinstance(hostapi_info, dict):
                    hostapi_name = str(hostapi_info.get("name", hostapi_name))
            except Exception:
                pass
        return hostapi_name

    @staticmethod
    def _run_command_lines(command: list[str]) -> list[str]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except Exception:
            return []
        return [line.rstrip("\n") for line in result.stdout.splitlines()]