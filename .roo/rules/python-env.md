# Python Virtual Environment Rules
The project contains multiple virtual environments (e.g., {{workspace}}/venv-[name]). Do not scan or assume dependencies from random venv folders. Always check which Python interpreter is actively selected in VS Code (via .vscode/settings.json or terminal context) and restrict your code searches, imports, and executions to that specific environment.

## Verification
If you are unsure if the venv is active, run `which python` (Unix) or `where python` (Windows) to verify the path points to the `venv-[name]` folder before executing complex logic.