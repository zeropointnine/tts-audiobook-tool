import os
import json

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.gen_info import GenInfo
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.transcribe_util import TranscribeUtil

# ---

DIR = "/home/lee/Documents/w/w/_error rate survey/higgs 500-1000/segments"

def load_geninfos(dir: str) -> list[GenInfo]:

    items: list[GenInfo] = []
    for filename in sorted(os.listdir(DIR)):
        if "geninfo" in filename:
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
            item = GenInfo.from_dict(data)
            items.append(item)
    return items

def count_fixed_not_fixed(items: list[GenInfo]) -> tuple[int, int]:

    results: dict[int, str] = {} # value is concatenation of validation result classes

    for item in items:
        if not item.index in results:
            results[item.index] = item.validation_result_class
        else:
            results[item.index] += " " + item.validation_result_class
    
    num_fixed = 0
    num_not_fixed = 0

    for value in results.values():
        if len(value.split(" ")) > 1:
            if "PassResult" in value:
                num_fixed += 1
            else:
                num_not_fixed += 1
    
    return num_fixed, num_not_fixed

# ---

print(Ansi.CLEAR_SCREEN_AND_SCROLLBACK)

items = load_geninfos(DIR)
print(f"{len(items)} items loaded")
print()


seen_item_indices = set()
count = 0

for item in items:

    if item.validation_result_class != "FailResult":
        continue

    # if item.index in seen_item_indices:
    #     continue
    # seen_item_indices.add(item.index)
    count += 1

    a, b = TextNormalizer.normalize_source_and_transcript(item.source, item.transcript, "en") 
    fails = TranscribeUtil.count_word_failures(a, b, "en")

    print("index:     :", item.index + 1)
    print("prompt     :", item.prompt)
    print("source raw :", item.source.strip())
    print("transc raw :", item.transcript.strip())
    print("source norm:", a)
    print("transc norm:", b)
    print("fails      :", fails)
    if fails != item.validation_num_word_fails:
        print("WTF", item.validation_num_word_fails)
    print()

print("num items:", count)

print("fixed/not fixed:", count_fixed_not_fixed(items) )