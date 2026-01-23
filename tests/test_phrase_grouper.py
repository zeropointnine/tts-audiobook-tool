import unittest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.phrase_grouper import PhraseGrouper

class TestPhraseGrouper(unittest.TestCase):

    def test_phrase_grouper(self):

        items = [
            (SAMPLE1, []),
        ]

        for inp, answer in items:
          groups = PhraseGrouper.text_to_groups(inp, 20, SegmentationStrategy.MULTI_SENTENCE, "en")
          print()
          print("-" * 80)
          print("input:")
          print("-" * 80)
          print()
          print(inp)
          print()

          print("-" * 80)
          print("result:")
          print("-" * 80)
          print()
          PhraseGrouper.print_groups(groups)
          # TODO:
          # self.assertTrue(groups == answer)
          print()


if __name__ == '__main__':
    unittest.main()

# ---

SAMPLE1 = """
Hitagi Senjogahara occupies the position of “the girl who’s always ill” in our class. She’s not expected to participate in P.E., of course, and is even allowed to suffer morning and school-wide assemblies in the shade, alone, as a precaution against anemia or something. Though we’ve been in the same class my first, my second, and this, my third and final year of high school, I’ve never once seen her engaged in any sort of vigorous activity. She’s a regular at the nurse’s room, and she arrives late, leaves early, or simply doesn’t show up to school because she has to visit her primary care hospital, time and again. To the point where it’s rumored in jest that she lives there.

Though “always ill,” she is by no means sickly. She’s graceful, like her thin lines could snap at a touch, and has this evanescent air, which must be why some of the boys refer to her as “the cloistered princess” half-jokingly, half-seriously. You could say earnestly. That phrase and its connotations aptly describe Senjogahara, I agree.

Senjogahara is always alone reading a book in one corner of the classroom. At times that book is an imposing hardcover, and at others it’s a comic that could permanently damage your intellect to judge from its cover design. She seems to be one of those voracious readers. Maybe she doesn’t care as long as there are words in it, maybe she has some sort of clear standard.
"""
