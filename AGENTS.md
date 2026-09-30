# Agent operation

When inspecting large source files, prefer targeted searches and bounded line-range reads when the available tools support them. Read the entire file only when its full context is genuinely needed.

# Testing and validation

Once appropriate validation has passed, stop. Do not expand testing merely to increase confidence when the remaining risk is low relative to the scope of the change.

The full project test suite is relatively expensive. Treat it as broad validation, not the default completion step for every edit.

# Python environment

- Treat `pyproject.toml` as effectively irrelevant to dependency and tooling discovery; its only agent-relevant content is the configuration concerning the `tests` directory.

# Developer documentation

`docs-dev/` contains developer- and LLM-facing architecture notes, subsystem explanations, and design context. Consult relevant documents when investigating or changing a subsystem. Use them as supporting context.

# Development scripts

`testx/` directory contains durable developer-facing scripts that demonstrate a mostly working subsystem, verify assumptions, or serve as rerunnable reference material belong in `testx/`. Run scripts in `testx/` like so:

```bash
./some-venv/bin/python -m testx.llm_util
```

# Probe scripts

Use *_probe.py for disposable, task-local diagnostic scripts created during investigation to answer a specific question; remove them when no longer needed.

