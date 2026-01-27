import json
import os


class OuteUtil:

    @staticmethod
    def load_oute_voice_json(path: str) -> dict | str:
        """
        Returns voice dict or error string
        Note how we are intentionally not using oute interface `load_speaker()`
        """
        if not os.path.exists(path):
            return f"Doesn't exist: {path}"
        try:
            with open(path, 'r', encoding='utf-8') as file:
                voice = json.load(file)
        except Exception as e:
            return f"Error loading voice json file: {e}"
        if not isinstance(voice, dict):
            return f"Voice json object unexpected type: {type(voice)}"
        # TODO: do some extra validation here since we are avoiding using "interface.load_speaker()"

        return voice
