from __future__ import annotations
from typing import ClassVar

from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind, TtsModelSpec


class _ModelCatalogMeta(type):
    """Retain legacy iteration over the ordered TOML-backed model catalog."""

    def __iter__(cls):
        # Legacy Enum iteration compatibility; app code uses all().
        return iter(getattr(cls, "_catalog").values())



class TtsModelType(metaclass=_ModelCatalogMeta):
    """Stable model identity backed by an ordered, replaceable spec catalog."""

    _catalog: dict[str, TtsModelType]
    _specs: dict[str, TtsModelSpec]
    _builtin_specs: dict[str, TtsModelSpec]
    NONE: ClassVar[TtsModelType]
    AUK_SERVER: ClassVar[TtsModelType]
    AUK_FLASH_SERVER: ClassVar[TtsModelType]
    CHATTERBOX: ClassVar[TtsModelType]
    DOTS: ClassVar[TtsModelType]
    FISH_S1: ClassVar[TtsModelType]
    FISH_S2: ClassVar[TtsModelType]
    FISH_S2_SERVER: ClassVar[TtsModelType]
    GLM: ClassVar[TtsModelType]
    HIGGS_V2: ClassVar[TtsModelType]
    HIGGS_V3_SERVER: ClassVar[TtsModelType]
    INDEXTTS2: ClassVar[TtsModelType]
    MIRA: ClassVar[TtsModelType]
    MOSS: ClassVar[TtsModelType]
    MOSS_DELAY_SERVER: ClassVar[TtsModelType]
    MOSS_LOCAL_SERVER: ClassVar[TtsModelType]
    OMNIVOICE: ClassVar[TtsModelType]
    POCKET: ClassVar[TtsModelType]
    QWEN3TTS: ClassVar[TtsModelType]
    QWEN3TTS_SERVER: ClassVar[TtsModelType]
    VIBEVOICE: ClassVar[TtsModelType]
    ZONOS2_SERVER: ClassVar[TtsModelType]

    @classmethod
    def reset_catalog(cls) -> None:
        """Restore built-ins during startup (also supports isolated startup tests)."""
        cls._catalog = {id: cls._catalog[id] for id in cls._builtin_specs}
        cls._specs = dict(cls._builtin_specs)

    def __init__(self, id: str):
        self.id = id

    def can_batch(self) -> bool:
        """Whether this model stores a batch size / concurrency value.

        Storage presence is derived from the model-settings registry, not
        from a project field name on the spec.
        """
        from tts_audiobook_tool.project_support.model_settings import REGISTRY
        return REGISTRY.orchestration_binding(self.id) is not None

    @property
    def value(self) -> TtsModelSpec:
        return self._specs[self.id]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TtsModelType) and self.id == other.id

    def __repr__(self) -> str:
        return f"TtsModelType({self.id!r})"

    @classmethod
    def all(cls) -> list[TtsModelType]:
        return list(cls._catalog.values())

    @classmethod
    def overlay_spec(cls, spec: TtsModelSpec) -> None:
        """Replace an existing definition without replacing its stable handle."""
        if spec.id not in cls._catalog:
            raise ValueError(f"Unknown model ID for overlay: {spec.id}")
        cls._specs[spec.id] = spec

    @classmethod
    def register_spec(cls, spec: TtsModelSpec) -> TtsModelType:
        if spec.id in cls._catalog:
            raise ValueError(f"Duplicate model ID: {spec.id}")
        handle = cls(spec.id)
        cls._catalog[spec.id] = handle
        cls._specs[spec.id] = spec
        return handle

    # ---

    @staticmethod
    def get_by_id(id: str) -> TtsModelType:
        return TtsModelType._catalog.get(id, TtsModelType.NONE)

    @staticmethod
    def recommended_range_string(range: tuple[int, int, str]) -> str:
        if range[1] == range[0]:
            s = f"up to {range[1]}"
        else:
            s = f"{range[0]}-{range[1]}"
        if range[2]:
            s += f" ({range[2]})"
        return s

    @staticmethod
    def all_file_tags() -> set[str]:
        return {item.value.file_tag for item in TtsModelType.all()}

    @staticmethod
    def get_items_by_backend(kind: TtsBackendKind) -> list[TtsModelType]:
        return [item for item in TtsModelType.all() if item.value.backend_kind == kind]

    @staticmethod
    def get_local_items() -> list[TtsModelType]:
        return TtsModelType.get_items_by_backend(TtsBackendKind.LOCAL)

    @staticmethod
    def get_sgl_omni_items() -> list[TtsModelType]:
        return TtsModelType.get_items_by_backend(TtsBackendKind.SGL_OMNI)

    @staticmethod
    def is_backend(item: TtsModelType, kind: TtsBackendKind) -> bool:
        """
        Predicate: does catalog member `item` have backend kind `kind`?
        (The primary API is the spec field, `TtsModelSpec.backend_kind`.)
        """
        return item.value.backend_kind == kind

    @staticmethod
    def is_valid_sgl_omni_type(value: TtsModelType | None) -> bool:
        """
        Predicate: is `value` a real SGL-Omni-backed catalog member,
        i.e. acceptable as the app's SGL-Omni TTS type?
        """
        return value is not None and value.value.backend_kind == TtsBackendKind.SGL_OMNI

    @staticmethod
    def find_tts_type_using_sgl_omni_model_id(model_id: str) -> TtsModelType | None:
        """
        Chooses a TtsModelType member using the model id returned by
        the SGL-Omni models endpoint (typically an hf repo id),
        using simple substring comparison.

        If more than one variant's substring matches, the longest
        (most specific) match wins; ties fall back to catalog order.
        """
        if not model_id:
            return None

        model_id = model_id.lower().strip()

        best: tuple[int, TtsModelType] | None = None
        for item in TtsModelType.get_sgl_omni_items():
            substring = item.value.sgl_omni_model_id_substring.lower()
            if substring and substring in model_id:
                if best is None or len(substring) > best[0]:
                    best = (len(substring), item)

        return best[1] if best else None


def _install_builtin_catalog() -> None:
    # The dependency-light parser shares model_spec types, so either the
    # catalog or this module can be imported first without a cycle.
    from tts_audiobook_tool.tts_models.model_catalog import load_catalog

    builtins, _, _ = load_catalog()
    expected = {name for name in TtsModelType.__annotations__ if name.isupper()}
    actual = {symbol for symbol, _ in builtins}
    if expected != actual:
        raise ValueError(f"Model catalog built-in handles mismatch: missing {sorted(expected - actual)}, unknown {sorted(actual - expected)}")
    catalog: dict[str, TtsModelType] = {}
    specs: dict[str, TtsModelSpec] = {}
    for symbol, spec in builtins:
        handle = TtsModelType(spec.id)
        setattr(TtsModelType, symbol, handle)
        catalog[spec.id] = handle
        specs[spec.id] = spec
    TtsModelType._catalog = catalog
    TtsModelType._specs = specs
    TtsModelType._builtin_specs = dict(specs)


_install_builtin_catalog()
