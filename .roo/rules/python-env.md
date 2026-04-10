# Python Environment & Interpreter Rules

1. **Locate Interpreter:** Before executing any Python scripts, installing packages, or searching for library definitions, check `.vscode/settings.json` for the `"python.defaultInterpreterPath"` key.

2. **Virtual Environment Validation:** 
   - If the path is defined in `settings.json`, use that specific executable for all terminal commands (e.g., use `/path/to/venv/bin/python` instead of just `python`).
   - If not defined in settings, look for common workspace venv folders: `.venv` or `venv`.

3. **Library Context:** When I ask you to "look at the source code" of a dependency or "find where X is defined," refer to the `lib/python3.x/site-packages` (Unix) or `Lib/site-packages` (Windows) directory inside the identified virtual environment path.

4. **Execution:** Always prefix python commands with the full path to the environment's interpreter to ensure the correct `site-packages` are loaded and to avoid using the global system Python.

5. **Consistency:** If you notice the current terminal session is not using the interpreter specified in `.vscode/settings.json`, notify me or use the absolute path to the correct interpreter.