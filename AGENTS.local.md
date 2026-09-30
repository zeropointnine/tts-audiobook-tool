# Python environment

- When running Python commands in this workspace and inspecting virtual environment library code, use `venv-base` by default. If the question at hand relates to a local TTS model's library code, use that model's venv — the venv→model mapping is in `.agents/venv-local-models.md`.

- Use **Pyright** as the project's primary Python static analyzer and for reproducing editor/Pylance diagnostics.


# audio.cpp

The audio.cpp local repository lives in `/d/p/audio.cpp-repo`. Treat this local checkout as the primary reference for audio.cpp behavior and capabilities. In particular, consult:

- `/d/p/audio.cpp-repo/model_specs/` for model capability/spec definitions
- `/d/p/audio.cpp-repo/app/server/` for server/API behavior
- `/d/p/audio.cpp-repo/docs/` for model and usage documentation
- `/d/p/audio.cpp-repo/community_models/` for community model implementations
- the relevant model implementation/source files when behavior is unclear
