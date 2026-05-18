"""
Generic text/string utility functions
"""

import re
from urllib.parse import urlencode


def strip_ansi_codes(s: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    osc_hyperlink_escape = re.compile(r'\x1b]8;;.*?\x1b\\')
    s = osc_hyperlink_escape.sub('', s)
    return ansi_escape.sub('', s)

def make_terminal_hyperlink(url: str, text: str = "", is_file: bool=False) -> str:
    display = text or url
    link = f"file://{url}" if is_file else url
    return f"\x1b]8;;{link}\x1b\\{display}\x1b]8;;\x1b\\"

def strip_quotes_around_path_string(s: str) -> str:
    if len(s) >= 2:
        first = s[0]
        last = s[-1]
        if (first == "'" and last == "'") or (first == "\"" and last == "\""):
            s = s[1:-1]
    return s

def make_random_hex_string(num_hex_chars: int=32) -> str:
    import random
    return f"{random.getrandbits(num_hex_chars * 4):0{num_hex_chars}x}"

def make_url_with_params(base_url: str, params: dict) -> str:
    """ Builds a properly encoded URL """
    if params:
        query_string = urlencode(params, doseq=True)
        return f"{base_url}?{query_string}"
    return base_url

def load_text_file(path: str, errors: str="strict") -> str:
    """ 
    Load text file of potentially unknown provenance or format 
    
    param errors:
        is passed to the decode(errors=) function.
        rem:
            "strict" is the default, which will raise an exception
            "ignore" will filter out unknown characters
            "replace" will replace unknown characters with the standard mystery character U+FFFD
    """
    import chardet
    try:
        # 1. Open as binary (rb) to get raw bytes, not text
        with open(path, 'rb') as f:
            raw_data = f.read()

        # 2. Detect the encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence'] # (Optional) strictly for debugging

        # 3. Handle edge case: if chardet is confused, default to utf-8
        if encoding is None:
            encoding = 'utf-8'

        # 4. Decode using the detected encoding
        transcript = raw_data.decode(encoding, errors=errors)
        
        # print(f"Loaded with encoding: {encoding} (Confidence: {confidence})")
        return transcript

    except Exception as e:
        return ""
