from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, ClassVar

from tts_audiobook_tool.util import make_error_string


class JsonArtifactType(Enum):
    PROJECT = "project settings"
    PROJECT_TEXT = "project text"
    PREFS = "preferences"


class JsonSaveUtil:
    """
    Safely persists the application's primary JSON artifacts.

    Methodology:
    - Serialize the complete payload in memory before creating any output file, so
      payload construction and JSON encoding failures cannot truncate a valid file.
    - Write the serialized data to a uniquely named temporary sibling, flush Python's
      buffer, and fsync the file before atomically replacing the destination.
    - Hold a reentrant, artifact-specific lock across payload construction and the
      complete write/replace sequence so threads cannot interleave saves of the same
      artifact type. Reentrancy also avoids deadlock if a save path is nested later.
    - Remove abandoned temporary files on failure and preserve the previous destination.

    Artifact types select synchronization and diagnostic context, while callers retain
    responsibility for destination paths. This keeps persistence mechanics independent
    from Project and Prefs and avoids circular imports. The locks are intentionally
    process-local; cross-process locking, backups, and multi-file transactions are
    separate concerns and are not provided here.
    """

    locks: ClassVar = {
        artifact_type: threading.RLock()
        for artifact_type in JsonArtifactType
    }

    @staticmethod
    def save(
        artifact_type: JsonArtifactType,
        path: str | os.PathLike[str],
        payload_factory: Callable[[], Any],
    ) -> str:
        """Return an error string on failure, otherwise atomically replace ``path``."""
        if not isinstance(artifact_type, JsonArtifactType):
            return f"Error saving JSON artifact: invalid artifact type {artifact_type!r}"

        destination = Path(path)
        temporary_path: Path | None = None
        lock = JsonSaveUtil.locks[artifact_type]

        with lock:
            try:
                payload = payload_factory()
                serialized = json.dumps(payload, indent=4, allow_nan=False)

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as file:
                    temporary_path = Path(file.name)
                    file.write(serialized)
                    file.flush()
                    os.fsync(file.fileno())

                os.replace(temporary_path, destination)
                temporary_path = None
                return ""
            except Exception as exception:
                return (
                    f"Error saving {artifact_type.value}: "
                    f"{make_error_string(exception)}"
                )
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        pass
