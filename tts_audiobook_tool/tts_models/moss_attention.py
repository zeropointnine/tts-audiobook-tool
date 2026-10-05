"""Instance-scoped compatibility for MOSS Local v1.5's remote-code attention."""

from functools import partial
from typing import Any


def _flash_attention_with_empty_query_fallback(
    attention: Any,
    original_flash_attention: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    packed_metadata: Any,
) -> Any:
    # Cached generation leaves finished rows in the batch, with their next
    # query masked out. Some Flash Attention 2 kernels cannot handle a mixture
    # of empty/nonempty query sequences (e.g. a batch of two after one ends).
    # Only that path needs SDPA; preserve normal flash prefill/decoding.
    if (
        packed_metadata is None
        and attention_mask is not None
        and query.shape[1] != key.shape[1]
        and not bool(attention_mask[:, -query.shape[1]:].any(dim=-1).all())
    ):
        return attention._sdpa_attention(
            query=query, key=key, value=value, attention_mask=attention_mask,
        )
    return original_flash_attention(
        query=query, key=key, value=value,
        attention_mask=attention_mask, packed_metadata=packed_metadata,
    )


def configure_moss_local_v15_attention(model: Any) -> None:
    """Install once after loading, without changing upstream classes or caches."""
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "moss_tts_local" or not hasattr(config, "gpt2_config"):
        return

    # This is the v1.5 global Qwen decoder, not the local GPT2 depth decoder.
    for layer in model.transformer.layers:
        attention = layer.self_attn
        attention._flash_attention = partial(
            _flash_attention_with_empty_query_fallback,
            attention,
            attention._flash_attention,
        )
