import torch

from tts_audiobook_tool.app_support import app_memory


def test_gc_does_not_initialize_a_new_parent_cuda_context(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: calls.append("ipc"))

    app_memory.gc_ram_vram()

    assert calls == []
