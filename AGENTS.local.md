# Python environment

- When running Python commands in this workspace and inspecting virtual environment library code, use `venv-base` by default. If the question at hand relates to a local TTS model's library code, use that model's venv — the venv→model mapping is in `.agents/venv-local-models.md`.

- Use **Pyright** as the project's primary Python static analyzer and for reproducing editor/Pylance diagnostics.


# audio.cpp

The audio.cpp local repository lives in `/d/p/audio.cpp-repo`. Treat this local checkout as the primary reference for audio.cpp behavior and capabilities.

# tests

When writing unit tests, add short comment explaining what the test illustrates illustrating unless the test is relatively trivial.

# git commit messages

When formulating git commit messages, the "headline" should start lowercase. The body text should use a bulleted list format which is relatively terse. Don't bother mentioning tests or documentation updates.