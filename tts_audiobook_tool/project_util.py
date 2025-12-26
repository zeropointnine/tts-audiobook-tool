import json
from tts_audiobook_tool.util import make_error_string


class ProjectUtil:
    """
    Helper functions for `Project`
    """
    
    @staticmethod
    def parse_word_substitutions_json_string(inp: str) -> dict[str, str] | str:
        """ Returns list of two-string lists or user-facing error message """
        try:
            import ast
            dic = ast.literal_eval(inp) # less strict than json decode, which requires double-quotes
        except Exception as e:
            return make_error_string(e)
        if not isinstance(dic, dict):
            return "Bad type"
        if len(dic) == 0:
            return "No items entered"
        for key, value in dic.items():
            if not key:
                return f"Found empty string"
            if not value:
                return f"Found empty string for {key}"
            if not isinstance(key, str):
                return f"Not a string: {key}"
            if not isinstance(value, str):
                return f"Not a string: {value}"
            # fyi, not checking for extra-long strings, punctuation, etc

        # Massage
        result = {}
        for key, value in dic.items():
            key = key.lower().strip() # search-word is treated as case-insensitive
            value = value.strip() # replacement-word is not
            result[key] = value
        return result
