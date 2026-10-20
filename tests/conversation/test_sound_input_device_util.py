import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tts_audiobook_tool.conversation.sound_input_device_util as module
from tts_audiobook_tool.conversation.sound_input_device_util import (
    SoundInputDeviceInfo,
)


class FakeSd:
    def __init__(self, devices, default_input, hostapis=None):
        self.devices = devices
        self._hostapis = hostapis if hostapis is not None else [{"name": "FakeHost"}]
        self.default = SimpleNamespace(device=(default_input, 0))

    def query_devices(self, idx=None):
        if idx is None:
            return self.devices
        return self.devices[idx]

    def query_hostapis(self, idx=None):
        if idx is None:
            return self._hostapis
        return self._hostapis[idx]


def device(name, in_ch=0, out_ch=0, hostapi=0) -> dict:
    return {"name": name, "max_input_channels": in_ch, "max_output_channels": out_ch,
            "hostapi": hostapi}


def install_sd(monkeypatch, devices, default_input, hostapis=None) -> None:
    monkeypatch.setattr(module, "sd", FakeSd(devices, default_input, hostapis))


def run_fake(monkeypatch, results: dict, failures=()) -> None:
    def fake_run(command, **kwargs):
        key = " ".join(command)
        if key in failures:
            raise RuntimeError("command failed")
        return SimpleNamespace(stdout=results.get(key, ""))

    monkeypatch.setattr(module.subprocess, "run", fake_run)


def test_is_input_device() -> None:
    assert SoundInputDeviceInfo._is_input_device(device("m", in_ch=2)) is True
    assert SoundInputDeviceInfo._is_input_device(device("o", out_ch=2)) is False
    assert SoundInputDeviceInfo._is_input_device("not a dict") is False


def test_selected_input_device_prefers_valid_default(monkeypatch) -> None:
    install_sd(monkeypatch, [device("in mic", in_ch=2), device("out", out_ch=2)], 0)

    idx, info, source = SoundInputDeviceInfo._get_selected_input_device()

    assert (idx, source) == (0, "default")
    assert info["name"] == "in mic"


def test_selected_input_device_falls_back_to_first_input_device(monkeypatch) -> None:
    install_sd(
        monkeypatch,
        [device("out only", out_ch=2), device("backup mic", in_ch=1), device("other")],
        0,
    )

    idx, info, source = SoundInputDeviceInfo._get_selected_input_device()

    assert (idx, source) == (1, "detected")
    assert info["name"] == "backup mic"


def test_selected_input_device_unavailable_when_no_devices(monkeypatch) -> None:
    install_sd(monkeypatch, [], -1)

    assert SoundInputDeviceInfo._get_selected_input_device() == (None, None, "unavailable")


def test_has_input_device_false_when_querying_fails(monkeypatch) -> None:
    def raising_query_devices(idx=None):
        raise OSError("no port")

    monkeypatch.setattr(module, "sd", SimpleNamespace(query_devices=raising_query_devices))

    assert SoundInputDeviceInfo.has_input_device() is False


def test_get_check_error_reports_no_devices(monkeypatch) -> None:
    install_sd(monkeypatch, [], -1)

    assert SoundInputDeviceInfo.get_check_error() == "No sound input devices found."


def test_get_check_error_reports_no_input_channels(monkeypatch) -> None:
    install_sd(monkeypatch, [device("out only", out_ch=2)], 0)

    assert (
        SoundInputDeviceInfo.get_check_error()
        == "No microphone / sound input device is available."
    )


def test_get_check_error_reports_query_failure(monkeypatch) -> None:
    def raising_query_devices(idx=None):
        raise OSError("no port")

    monkeypatch.setattr(module, "sd", SimpleNamespace(query_devices=raising_query_devices))

    assert SoundInputDeviceInfo.get_check_error().startswith("Couldn't verify")


@pytest.mark.parametrize("platform_name", ["linux"])
def test_input_device_description_uses_pactl_name_on_linux(monkeypatch, platform_name) -> None:
    install_sd(monkeypatch, [device("pcm in", in_ch=2, hostapi=0)], 0,
               hostapis=[{"name": "FakeHost"}])
    monkeypatch.setattr(sys, "platform", platform_name)
    run_fake(monkeypatch, {
        "pactl get-default-source": "alsa_input.PC.output\n",
        "pactl list sources": (
            "Source #0\n\tName: alsa_input.PC.output\n"
            "\tDescription: Internal Microphone\n"
        ),
    })

    description = SoundInputDeviceInfo.get_input_device_description()

    assert description == "Internal Microphone (device index 0, host API FakeHost) (default)"


def test_linux_name_falls_back_to_wpctl(monkeypatch) -> None:
    install_sd(monkeypatch, [device("pcm in", in_ch=2, hostapi=0)], 0,
               hostapis=[{"name": "FakeHost"}])
    monkeypatch.setattr(sys, "platform", "linux")
    run_fake(
        monkeypatch,
        {"wpctl status": "Global\n├─ Sources:\n* 1. MyMic [3.0]\n└─ Sinks:\n"},
        failures=["pactl get-default-source", "pactl list sources"],
    )

    description = SoundInputDeviceInfo.get_input_device_description()

    assert description == "MyMic (device index 0, host API FakeHost) (default)"


def test_input_device_description_uses_switch_audiosource_on_macos(monkeypatch) -> None:
    install_sd(monkeypatch, [device("mic", in_ch=2, hostapi=0)], 0,
               hostapis=[{"name": "MacHost"}])
    monkeypatch.setattr(sys, "platform", "darwin")
    run_fake(monkeypatch, {"SwitchAudioSource -t input -c": "MacBook Pro Microphone\n"})

    description = SoundInputDeviceInfo.get_input_device_description()

    assert description == "MacBook Pro Microphone (device index 0, host API MacHost) (default)"


def test_windows_preferred_default_input_prefers_wasapi(monkeypatch) -> None:
    devices = [device("out"), device("wasapi mic", in_ch=1), device("mme mic", in_ch=1)]
    install_sd(monkeypatch, devices, -1, hostapis=[
        {"name": "MME", "default_input_device": 2},
        {"name": "Windows WASAPI", "default_input_device": 1},
    ])

    assert SoundInputDeviceInfo._get_windows_preferred_default_input_index() == 1


def test_windows_preferred_default_input_uses_first_available_in_order(monkeypatch) -> None:
    devices = [device("out"), device("mme mic", in_ch=1)]
    install_sd(monkeypatch, devices, -1, hostapis=[
        {"name": "MME", "default_input_device": 1},
    ])

    assert SoundInputDeviceInfo._get_windows_preferred_default_input_index() == 1


def test_windows_preferred_default_input_none_when_no_input_defaults(monkeypatch) -> None:
    devices = [device("out")]
    install_sd(monkeypatch, devices, -1, hostapis=[
        {"name": "Windows WASAPI", "default_input_device": -1},
    ])

    assert SoundInputDeviceInfo._get_windows_preferred_default_input_index() is None


def test_get_hostapi_name_resolves_index_and_handles_errors(monkeypatch) -> None:
    install_sd(monkeypatch, [device("mic", in_ch=1, hostapi=0)], 0,
               hostapis=[{"name": "ALSA"}, {"name": "Pulse"}])

    assert SoundInputDeviceInfo._get_hostapi_name({"hostapi": 1}) == "Pulse"
    assert SoundInputDeviceInfo._get_hostapi_name({"hostapi": 99}) == "Unknown host API"
    assert SoundInputDeviceInfo._get_hostapi_name({"hostapi": "weird"}) == "Unknown host API"
    assert SoundInputDeviceInfo._get_hostapi_name({}) == "Unknown host API"


def test_get_hostapi_name_handles_query_failure(monkeypatch) -> None:
    class FailingSd(FakeSd):
        def query_hostapis(self, idx=None):
            raise OSError("no ports")

    monkeypatch.setattr(module, "sd", FailingSd([device("x")], -1))

    assert SoundInputDeviceInfo._get_hostapi_name({"hostapi": 0}) == "Unknown host API"