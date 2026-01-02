import os
import json

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil

# ---

DIR = "/home/lee/Documents/w/w/luminous/segments"

def load_jsons(dir: str) -> list[dict]:

    items: list[dict] = []
    for filename in sorted(os.listdir(dir)):
        if not ".debug.json" in filename:
            continue
        filepath = os.path.join(DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading {filename}: {e}")
            continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        items.append(data)
    return items

# ---

print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)

items = load_jsons(DIR)
print(f"{len(items)} items loaded")
print()

for item in items:

    fails = ValidateUtil.get_word_errors(item["normalized_source"], item["normalized_transc"], "en", False)
    if not fails: 
        continue

    for key in item:
        # if key in ["index_1b", "normalized_source", "normalized_transc", "result_desc"]:
        print(f"{COL_DIM}{key:>24}: {COL_DEFAULT}{str(item[key]).strip()}")
    
    print("Word fails:", fails)
    print()
