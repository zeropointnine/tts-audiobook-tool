import gc
import weakref

import pytest
import torch

pytest.importorskip("chatterbox")

from chatterbox.models.t3.inference.alignment_stream_analyzer import (
    AlignmentStreamAnalyzer,
)

from tts_audiobook_tool.app_types import DeviceType, Sound
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxType
from tts_audiobook_tool.tts_models.chatterbox_model import ChatterboxModel


class FakeSelfAttn(torch.nn.Module):
    pass


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeSelfAttn()


class FakeConfig:
    output_attentions = False
    _attn_implementation = "sdpa"


class FakeTfmr:
    def __init__(self):
        self.layers = [FakeLayer() for _ in range(16)]
        self.config = FakeConfig()


class FakeT3:
    def __init__(self):
        self.tfmr = FakeTfmr()


class FakeChatterbox:
    def __init__(self):
        self.t3 = FakeT3()


def make_analyzer(chatterbox) -> AlignmentStreamAnalyzer:
    return AlignmentStreamAnalyzer(
        chatterbox.t3.tfmr,
        None,
        text_tokens_slice=(2, 8),
        alignment_layer_idx=9,
        eos_idx=0,
    )


def num_hooks(chatterbox) -> list[int]:
    return [
        len(chatterbox.t3.tfmr.layers[i].self_attn._forward_hooks)
        for i in ChatterboxModel._ALIGNED_ATTN_LAYER_INDICES
    ]


def test_strip_removes_analyzer_hooks_and_keeps_unrelated_hooks():
    chatterbox = FakeChatterbox()

    # Two generations' worth of analyzers, as the library registers them.
    analyzers = [make_analyzer(chatterbox) for _ in range(2)]

    # An unrelated hook on one of the same layers must survive the strip.
    keep_layer = chatterbox.t3.tfmr.layers[ChatterboxModel._ALIGNED_ATTN_LAYER_INDICES[0]].self_attn
    keep_layer.register_forward_hook(lambda module, inputs, output: None)

    assert num_hooks(chatterbox) == [3, 2, 2]

    ChatterboxModel._strip_alignment_analyzer_hooks(chatterbox)

    assert num_hooks(chatterbox) == [1, 0, 0]
    assert len(analyzers) == 2


def test_strip_makes_orphaned_analyzers_collectable():
    chatterbox = FakeChatterbox()
    analyzer = make_analyzer(chatterbox)
    ref = weakref.ref(analyzer)
    del analyzer

    # Still alive: the transformer's hook closures reference it.
    gc.collect()
    assert ref() is not None

    ChatterboxModel._strip_alignment_analyzer_hooks(chatterbox)
    gc.collect()
    assert ref() is None


def test_strip_is_noop_for_unrelated_shapes():
    # A library layout change (missing t3/tfmr) must not raise.
    class Empty:
        pass

    ChatterboxModel._strip_alignment_analyzer_hooks(Empty())
    ChatterboxModel._strip_alignment_analyzer_hooks(None) # type: ignore[arg-type]


def test_wrapper_generate_leaves_no_analyzer_hooks():
    class HookRegisteringChatterbox(FakeChatterbox):
        """Mimics the library: each generate() registers a fresh analyzer."""

        def generate(self, text: str, **dic) -> torch.Tensor:
            make_analyzer(self)
            return torch.zeros(4800)

    model = ChatterboxModel.__new__(ChatterboxModel)
    model._device_type = DeviceType.CPU
    model._model_type = ChatterboxType.MULTILINGUAL
    chatterbox = HookRegisteringChatterbox()
    model._chatterbox = chatterbox

    for _ in range(3):
        result = model.generate(text="Test sentence.", voice_path="", seed=1)
        assert isinstance(result, Sound)

    assert num_hooks(chatterbox) == [0, 0, 0]
