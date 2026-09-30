from __future__ import annotations

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
    _initial_catalog: dict[str, TtsModelType]
    _initial_specs: dict[str, TtsModelSpec]

    @classmethod
    def reset_catalog(cls) -> None:
        """Restore the imported catalog, removing optional startup definitions."""
        cls._catalog = dict(cls._initial_catalog)
        cls._specs = dict(cls._initial_specs)

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
    def install_spec(cls, spec: TtsModelSpec) -> TtsModelType:
        """Install metadata uniformly, retaining an existing identity if present."""
        handle = cls._catalog.get(spec.id)
        if handle is None:
            handle = cls(spec.id)
            cls._catalog[spec.id] = handle
        cls._specs[spec.id] = spec
        return handle

    @classmethod
    def register_spec(cls, spec: TtsModelSpec) -> TtsModelType:
        if spec.id in cls._catalog:
            raise ValueError(f"Duplicate model ID: {spec.id}")
        return cls.install_spec(spec)

    # ---

    @staticmethod
    def require_by_id(id: str) -> TtsModelType:
        """Return a registered canonical handle, raising for an unknown ID."""
        try:
            return TtsModelType._catalog[id]
        except KeyError:
            raise ValueError(f"Unknown TTS model ID: {id!r}") from None

    @staticmethod
    def get_by_id(id: str) -> TtsModelType:
        """Return a canonical handle or the registered ``none`` placeholder.

        Use for saved selections and external IDs that may be unavailable.
        Callers should preserve the original ID rather than saving the fallback.
        """
        return TtsModelType._catalog.get(id, TtsModelType._catalog["none"])

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
        """Compatibility helper; SGL matching belongs to its backend detector."""
        from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
        matches = detect_sgl_omni_models([{"id": model_id}])
        return matches[0][0] if matches else None


def _install_catalog() -> None:
    # The dependency-light parser shares model_spec types, so either the
    # catalog or this module can be imported first without a cycle.
    from tts_audiobook_tool.tts_models.model_catalog import load_catalog

    specs, _, _ = load_catalog()
    TtsModelType._catalog = {}
    TtsModelType._specs = {}
    for spec in specs:
        TtsModelType.install_spec(spec)
    TtsModelType._initial_catalog = dict(TtsModelType._catalog)
    TtsModelType._initial_specs = dict(TtsModelType._specs)


_install_catalog()
