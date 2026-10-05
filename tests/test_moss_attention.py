from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tts_audiobook_tool.tts_models.moss_attention import configure_moss_local_v15_attention


class FakeAttention:
    def __init__(self):
        self.flash_calls = []
        self.sdpa_calls = []

    def _flash_attention(self, **kwargs):
        self.flash_calls.append(kwargs)
        query, key, mask = kwargs["query"], kwargs["key"], kwargs["attention_mask"]
        if (
            mask is not None
            and query.shape[1] != key.shape[1]
            and not bool(mask[:, -query.shape[1]:].any(dim=-1).all())
        ):
            raise RuntimeError("shape '[2, 8, 4, 128]' is invalid for input of size 4096")
        return torch.ones_like(query)

    def _sdpa_attention(self, **kwargs):
        self.sdpa_calls.append(kwargs)
        query, key, value = (kwargs[name].transpose(1, 2) for name in ("query", "key", "value"))
        query_positions = torch.arange(query.shape[-2]) + key.shape[-2] - query.shape[-2]
        causal = torch.arange(key.shape[-2])[None, :] <= query_positions[:, None]
        mask = causal[None, None] & kwargs["attention_mask"][:, None, None, :]
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
        ).transpose(1, 2)


def make_model(attention):
    return SimpleNamespace(
        config=SimpleNamespace(model_type="moss_tts_local", gpt2_config=object()),
        transformer=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )


def decode_kwargs(mask):
    return dict(
        query=torch.randn(len(mask), 1, 2, 8),
        key=torch.randn(len(mask), len(mask[0]), 2, 8),
        value=torch.randn(len(mask), len(mask[0]), 2, 8),
        attention_mask=torch.tensor(mask, dtype=torch.bool),
        packed_metadata=None,
    )


@pytest.mark.parametrize("batch_size", [2, 3])
def test_finished_batch_row_falls_back_to_sdpa(batch_size):
    attention = FakeAttention()
    kwargs = decode_kwargs([[1, 1, 0]] + [[1, 1, 1]] * (batch_size - 1))
    with pytest.raises(RuntimeError, match="invalid for input"):
        attention._flash_attention(**kwargs)
    attention.flash_calls.clear()

    configure_moss_local_v15_attention(make_model(attention))
    output = attention._flash_attention(**kwargs)

    assert output.shape == kwargs["query"].shape
    assert torch.isfinite(output).all()
    assert attention.flash_calls == []
    assert len(attention.sdpa_calls) == 1
    assert attention.sdpa_calls[0]["attention_mask"] is kwargs["attention_mask"]


@pytest.mark.parametrize("mask", [[[1, 1, 1]], [[0, 1, 1], [1, 1, 1]]])
def test_single_item_and_active_batches_keep_flash(mask):
    attention = FakeAttention()
    configure_moss_local_v15_attention(make_model(attention))
    kwargs = decode_kwargs(mask)
    assert torch.equal(attention._flash_attention(**kwargs), torch.ones_like(kwargs["query"]))
    assert len(attention.flash_calls) == 1
    assert attention.sdpa_calls == []


def test_prefill_and_unmasked_attention_keep_flash():
    attention = FakeAttention()
    configure_moss_local_v15_attention(make_model(attention))
    kwargs = decode_kwargs([[0, 1, 1], [1, 1, 1]])
    kwargs["query"] = torch.randn(2, 3, 2, 8)
    attention._flash_attention(**kwargs)
    kwargs["attention_mask"] = None
    attention._flash_attention(**kwargs)
    assert len(attention.flash_calls) == 2
    assert attention.sdpa_calls == []


def test_packed_attention_uses_original_flash_path():
    attention = FakeAttention()
    configure_moss_local_v15_attention(make_model(attention))
    kwargs = decode_kwargs([[1, 1, 1], [1, 1, 1]])
    metadata = object()
    kwargs["packed_metadata"] = metadata
    attention._flash_attention(**kwargs)
    assert attention.flash_calls[0]["packed_metadata"] is metadata
    assert attention.sdpa_calls == []


def test_each_layer_falls_back_independently():
    first, second = FakeAttention(), FakeAttention()
    model = make_model(first)
    model.transformer.layers.append(SimpleNamespace(self_attn=second))
    configure_moss_local_v15_attention(model)
    kwargs = decode_kwargs([[1, 1, 0], [1, 1, 1]])
    first._flash_attention(**kwargs)
    second._flash_attention(**kwargs)
    assert len(first.sdpa_calls) == len(second.sdpa_calls) == 1


@pytest.mark.parametrize("config", [
    None,
    SimpleNamespace(model_type="moss_tts_delay"),
    SimpleNamespace(model_type="moss_tts_local"),
])
def test_other_architectures_are_untouched(config):
    # No transformer here: the compatibility code must not touch these models.
    configure_moss_local_v15_attention(SimpleNamespace(config=config))
