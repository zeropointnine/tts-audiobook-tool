"""Switches between per-voice LoRA adapters on a loaded model whose adapters are merged into the weights.

Model-agnostic: a model supplies its parameters, where its untouched base weights can be read from
(`BaseWeightSource`), and a planner that turns a `VoiceLora` into a validated `LoraMergePlan`
(format-specific, eg `breeze_lora.prepare_lora`).

A merged adapter can't be subtracted back out exactly (`bf16(W + d) - d != W`), so switching restores
the adapter's target weights from a clean copy and merges the next adapter:

    activate(B) while A is merged:  restore A's targets from base, merge B
    activate(None):                 restore only

Weights are written in place (`copy_`), so parameter storage never moves and captured CUDA graphs that
read the weights stay valid. Switching to the LoRA that is already active (same target, strength, and
adapter file contents) does nothing; the first merge onto pristine weights reads nothing from `base`.

The base copy (`BaseWeightCache`) is built lazily, at the first restore. It is pinned in RAM when there
appears to be enough memory (`should_pin_base_weights`), otherwise read from the model's files on every
restore. `end_run()` releases it (and the prepared plans); the next restore rebuilds it.

Merges always compute the delta on the weight's device (`apply_merge(delta_device="target")`), so a
voice's weights don't depend on the order of switches.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal, Protocol

import torch

from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge

GIB = 1024 ** 3
# RAM that must remain available after pinning: the larger of this and PIN_MARGIN_FRACTION of total RAM
PIN_MARGIN_MIN_BYTES = 4 * GIB
PIN_MARGIN_FRACTION = 0.15

BaseCacheMode = Literal["undecided", "pinned", "files"]
LoraKey = tuple[str, float]


def should_pin_base_weights(nbytes: int, device_type: str, ram: tuple[int, int] | None) -> bool:
    """Whether to keep `nbytes` of base weights pinned in RAM (`ram`: available, total bytes).
    Only for CUDA (pinning speeds up host-to-device copies; on CPU/MPS it would just duplicate the
    weights), and only when RAM stays comfortably available afterwards."""
    if device_type != "cuda" or ram is None:
        return False
    available, total = ram
    return available >= nbytes + max(PIN_MARGIN_MIN_BYTES, PIN_MARGIN_FRACTION * total)


def _param_nbytes(param: torch.Tensor) -> int:
    return param.numel() * param.element_size()


class BaseWeightSource(Protocol):
    def read(self, params: dict[str, torch.nn.Parameter], names: Iterable[str]) -> Iterator[tuple[str, torch.Tensor]]:
        """Yields (name, CPU tensor) for each name: the base value, cast to the parameter's dtype.
        Raises if a name can't be found or doesn't match the parameter's shape."""
        ...


class SafetensorsBaseWeights:
    """Base weights read from the model's own safetensors files: sharded (`model.safetensors.index.json`)
    or a single `model.safetensors`. `key_for_param` maps a model parameter name to its checkpoint key
    (identity by default). Values are cast to the parameter's dtype, as loading did (eg OmniVoice's fp32
    checkpoint loads as fp16 on CUDA); the cast is deterministic, so the restore is exact."""

    INDEX_FILE_NAME = "model.safetensors.index.json"
    SINGLE_FILE_NAME = "model.safetensors"

    def __init__(self, model_dir: str, key_for_param: Callable[[str], str] | None = None):
        self._model_dir = model_dir
        self._key_for_param = key_for_param or (lambda name: name)
        self._weight_map: dict[str, str] | None = None

    def _file_for_key(self, key: str) -> str:
        if self._weight_map is None:
            index_path = os.path.join(self._model_dir, self.INDEX_FILE_NAME)
            if os.path.isfile(index_path):
                with open(index_path, encoding="utf-8") as f:
                    self._weight_map = dict(json.load(f)["weight_map"])
            else:
                self._weight_map = {}
        file_name = self._weight_map.get(key) if self._weight_map else self.SINGLE_FILE_NAME
        if file_name is None:
            raise KeyError(f"Base weight {key} isn't in the model's checkpoint index")
        return os.path.join(self._model_dir, file_name)

    def read(self, params: dict[str, torch.nn.Parameter], names: Iterable[str]) -> Iterator[tuple[str, torch.Tensor]]:
        from safetensors import safe_open

        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self._file_for_key(self._key_for_param(name)), []).append(name)
        for path, file_names in by_file.items():
            with safe_open(path, "pt", device="cpu") as f:
                for name in file_names:
                    tensor = f.get_tensor(self._key_for_param(name))
                    param = params[name]
                    if tuple(tensor.shape) != tuple(param.shape):
                        raise ValueError(f"Base weight {name} has shape {tuple(tensor.shape)}, "
                                         f"model has {tuple(param.shape)}")
                    yield name, tensor.to(param.dtype)


class BaseWeightCache:
    """
    Untouched values of the weights adapters write, for restoring them. The mode is decided at the first
    restore (`should_pin_base_weights`, sized by the names restored now plus `expected_names`, ie every
    prepared adapter's targets) and kept until `release()`:
        "pinned"  read once into pinned RAM; names first needed later are pinned too while RAM allows
        "files"   read from `source` on every restore (the OS file cache usually keeps that fast)
    A failed pin falls back to "files".
    """

    def __init__(
            self,
            source: BaseWeightSource,
            *,
            pin_policy: Callable[[int, str, tuple[int, int] | None], bool] = should_pin_base_weights,
            ram_info: Callable[[], tuple[int, int] | None] | None = None,
            pin: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self._source = source
        self._pin_policy = pin_policy
        self._ram_info = ram_info or _default_ram_info
        self._pin = pin or (lambda tensor: tensor.pin_memory())
        self._pinned: dict[str, torch.Tensor] = {}
        self.mode: BaseCacheMode = "undecided"
        self.mode_reason = ""

    @property
    def pinned_bytes(self) -> int:
        return sum(_param_nbytes(tensor) for tensor in self._pinned.values())

    def _try_pin(self, params: dict[str, torch.nn.Parameter], names: list[str]) -> bool:
        pinned: dict[str, torch.Tensor] = {}
        try:
            for name, tensor in self._source.read(params, names):
                pinned[name] = self._pin(tensor)
        except Exception as e:
            if isinstance(e, (KeyError, ValueError)):
                raise
            self.mode_reason = f"pinning failed ({type(e).__name__}: {e})"
            return False
        self._pinned.update(pinned)
        return True

    @torch.no_grad()
    def restore(self, params: dict[str, torch.nn.Parameter], names: Iterable[str],
                expected_names: Iterable[str] = ()) -> None:
        names = sorted(set(names))
        if not names:
            return
        if self.mode == "undecided":
            wanted = sorted(set(names) | set(expected_names))
            nbytes = sum(_param_nbytes(params[name]) for name in wanted)
            device_type = params[names[0]].device.type
            if self._pin_policy(nbytes, device_type, self._ram_info()):
                self.mode = "pinned" if self._try_pin(params, wanted) else "files"
                if self.mode == "pinned":
                    self.mode_reason = f"{nbytes / GIB:.1f} GB pinned in RAM"
            else:
                self.mode = "files"
                self.mode_reason = "not enough free RAM to pin" if device_type == "cuda" else "not CUDA"
        if self.mode == "pinned":
            missing = [name for name in names if name not in self._pinned]
            if missing:
                nbytes = sum(_param_nbytes(params[name]) for name in missing)
                if self._pin_policy(nbytes, params[missing[0]].device.type, self._ram_info()):
                    self._try_pin(params, missing)

        from_files = [name for name in names if name not in self._pinned]
        uses_cuda = False
        for name in names:
            if name in self._pinned:
                target = params[name]
                uses_cuda |= target.device.type == "cuda"
                target.data.copy_(self._pinned[name], non_blocking=True)
        for name, tensor in self._source.read(params, from_files):
            target = params[name]
            uses_cuda |= target.device.type == "cuda"
            target.data.copy_(tensor)
        if uses_cuda:
            # Pinned sources are copied asynchronously; don't let the next kernel (or a release) race them
            torch.cuda.synchronize()

    def release(self) -> None:
        self._pinned.clear()
        self.mode = "undecided"
        self.mode_reason = ""


def _default_ram_info() -> tuple[int, int] | None:
    from tts_audiobook_tool.app_support.app_memory import get_available_ram
    return get_available_ram()


@dataclass(frozen=True)
class PreparedLora:
    """A validated adapter, ready to merge. `identity` covers its tensor contents and any metadata
    affecting the merge (eg alpha), so changes at the same path trigger a re-merge;
    `display` names it in messages."""
    plan: LoraMergePlan
    identity: str
    display: str


def lora_key(lora: VoiceLora) -> LoraKey:
    return lora.target, lora.strength


class LoraSwitcher:
    """
    The active per-voice LoRA of one loaded model (see the module docstring).

    `planner(lora)` reads and validates an adapter against the model without changing it; it raises
    `LoraLoadError` (or anything else) for an unusable one. Plans and planning errors are cached per
    (target, strength) until `end_run()`.

    A failure while restoring or merging leaves the weights in an unknown state: the switcher is then
    `broken` and every later `activate()` fails, so generation stops (the model must be reloaded).
    """

    def __init__(
            self,
            model: torch.nn.Module,
            base: BaseWeightCache,
            planner: Callable[[VoiceLora], PreparedLora],
            log: Callable[[str], None] | None = None,
    ):
        self._model = model
        self._base = base
        self._planner = planner
        self._log = log or _default_log
        self._prepared: dict[LoraKey, PreparedLora] = {}
        self._errors: dict[LoraKey, str] = {}
        self._dirty: set[str] = set()
        self._active_key: LoraKey | None = None
        self._active_identity = ""
        self._active_display = ""
        self.broken = ""

    @property
    def active_key(self) -> LoraKey | None:
        return self._active_key

    @property
    def active_display(self) -> str:
        return self._active_display

    @property
    def base(self) -> BaseWeightCache:
        return self._base

    def prepare(self, lora: VoiceLora) -> str:
        """Validates `lora` (cached); returns an error message or ""."""
        key = lora_key(lora)
        if key in self._prepared:
            return ""
        if key in self._errors:
            return self._errors[key]
        try:
            self._prepared[key] = self._planner(lora)
        except LoraLoadError as e:
            self._errors[key] = str(e)
        except Exception as e:
            self._errors[key] = f"Couldn't read adapter: {type(e).__name__}: {e}"
        return self._errors.get(key, "")

    def activate(self, lora: VoiceLora | None) -> str:
        """Makes `lora` (None: no LoRA) the merged adapter. Returns an error message or "";
        on a planning error the weights are unchanged."""
        if self.broken:
            return self.broken
        key = lora_key(lora) if lora is not None else None
        prepared: PreparedLora | None = None
        if lora is not None:
            error = self.prepare(lora)
            if error:
                return error
            prepared = self._prepared[lora_key(lora)]
        if key == self._active_key and (prepared is None or prepared.identity == self._active_identity):
            return ""

        start = time.perf_counter()
        restored = bool(self._dirty)
        try:
            params = dict(self._model.named_parameters())
            if self._dirty:
                expected = set().union(*(p.plan.target_names for p in self._prepared.values()))
                self._base.restore(params, self._dirty, expected)
                self._dirty = set()
            self._active_key, self._active_identity, self._active_display = None, "", ""
            if prepared is not None:
                # Tracked before merging: a failed merge leaves these targets partly written
                self._dirty = set(prepared.plan.target_names)
                apply_merge(self._model, prepared.plan, delta_device="target")
                self._active_key, self._active_identity, self._active_display = key, prepared.identity, prepared.display
        except Exception as e:
            self.broken = (f"Couldn't switch LoRA ({type(e).__name__}: {e}); "
                           "model weights are in an unknown state, so the model must be reloaded")
            return self.broken

        elapsed = time.perf_counter() - start
        what = f"{prepared.display} (strength {lora.strength:g})" if prepared is not None and lora else "none"
        message = f"LoRA: {what}"
        if restored:
            message += f" - restored base weights ({self._base.mode_reason or self._base.mode})"
        self._log(f"{message}, {elapsed:.1f}s")
        return ""

    def release_base(self) -> None:
        """Releases only the base copy (eg after a chat synthesis); prepared plans are kept."""
        self._base.release()

    def end_run(self) -> None:
        """Releases the base copy, the prepared plans and cached errors (an adapter is re-read and
        re-validated at its next use). The merged adapter stays merged."""
        self._base.release()
        self._prepared.clear()
        self._errors.clear()


def _default_log(message: str) -> None:
    from tts_audiobook_tool.constants import COL_DIM
    from tts_audiobook_tool.util import printt
    printt(f"{COL_DIM}{message}")
