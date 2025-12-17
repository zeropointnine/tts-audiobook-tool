import re

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.phrase import Reason
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.util import printt



if False:

    # Test num2words
    # --------------

    from num2words import num2words

    while True:
        text = input("Enter text: ")
        text = re.sub(
                r'\d+', 
                lambda x: x.group() if int(x.group()) > 999 else num2words(int(x.group()), lang="de"), 
                text
            )        
        print(text)

if True:

    # Test PhraseGrouper various

    text = """
hello? yes?

yes well. so. and so on. plus! plus! plus! plus!

"""
    print(text)
    print()

    groups = PhraseGrouper.text_to_groups(text, 40, SegmentationStrategy.NORMAL)

    groups = PhraseGrouper.merge_short_sentences(groups, 40)
    printt("")
    PhraseGrouper.print_groups(groups)
