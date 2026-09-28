# Python environment

- When running Python commands in this workspace and inspecting virtual environment library code, use `venv-base` by default. If the question at hand relates to a TTS model's library code, use that model's venv — the venv→model mapping is in `.agents/venv-models.md`.

- Use **Pyright** as the project's primary Python static analyzer and for reproducing editor/Pylance diagnostics.


# Project directories and files

Use *_probe.py for disposable, task-local diagnostic scripts created during investigation to answer a specific question; remove them when no longer needed.

Durable developer-facing scripts that demonstrate a mostly working subsystem, verify assumptions, or serve as rerunnable reference material belong in `testx/` instead. Run scripts in `testx/` like so:

```bash
./venv-base/bin/python -m testx.llm_util
```

