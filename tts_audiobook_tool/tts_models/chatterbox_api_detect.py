"""Lightweight Chatterbox API compatibility detection.

This module deliberately inspects installed source instead of importing
Chatterbox, which would also import Torch during startup dependency checks.
"""

from __future__ import annotations

import ast
from importlib import metadata
from pathlib import Path


class ChatterboxApiDetect:
    DISTRIBUTION_NAME = "chatterbox-tts"
    MULTILINGUAL_MODULE_PATH = "chatterbox/mtl_tts.py"
    V3_CHECKPOINT_NAME = "t3_mtl23ls_v3.safetensors"

    @staticmethod
    def has_required_v3_features() -> bool:
        """Return whether the installed package exposes the required V3-era API."""
        try:
            dist = metadata.distribution(ChatterboxApiDetect.DISTRIBUTION_NAME)
        except metadata.PackageNotFoundError:
            return False

        source_entry = next(
            (
                entry
                for entry in (dist.files or ())
                if entry.as_posix() == ChatterboxApiDetect.MULTILINGUAL_MODULE_PATH
            ),
            None,
        )
        if source_entry is None:
            return False

        try:
            source_path = Path(str(dist.locate_file(source_entry)))
            source = source_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False

        return ChatterboxApiDetect.source_has_required_v3_features(source)

    @staticmethod
    def source_has_required_v3_features(source: str) -> bool:
        """Inspect ``chatterbox.mtl_tts`` source for the required feature cluster."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False

        model_class = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "ChatterboxMultilingualTTS"
            ),
            None,
        )
        if model_class is None:
            return False

        loader = next(
            (
                node
                for node in model_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "from_pretrained"
            ),
            None,
        )
        if loader is None:
            return False

        loader_parameters = (
            loader.args.posonlyargs
            + loader.args.args
            + loader.args.kwonlyargs
        )
        has_t3_model_parameter = any(
            parameter.arg == "t3_model" for parameter in loader_parameters
        )
        has_model_resolver = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_resolve_multilingual_t3_model"
            for node in tree.body
        )
        has_v3_checkpoint = any(
            isinstance(node, ast.Constant)
            and node.value == ChatterboxApiDetect.V3_CHECKPOINT_NAME
            for node in ast.walk(tree)
        )

        return (
            has_t3_model_parameter
            and has_model_resolver
            and has_v3_checkpoint
        )
