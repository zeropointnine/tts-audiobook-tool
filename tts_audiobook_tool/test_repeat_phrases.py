import re

class TextUtils:
    @staticmethod
    def get_repeats(string: str) -> list[str]:
        """
        Finds words or phrases that repeat one or more times. The repeated
        word or phrase must be adjacent to each other.

        Args:
            string: The input string to search for repeats.

        Returns:
            A list of unique repeated words or phrases found in the string.
            The order of the returned list is not guaranteed.

        Example:
            "I want to go to New York New York because it's a city city"
            Returns: ['new york', 'to', 'city']
        """
        # 1. Pre-process the string: lowercase and remove punctuation
        # This makes the matching more robust (e.g., "Go, go" becomes "go go")
        clean_string = re.sub(r'[^\w\s]', '', string.lower())
        words = clean_string.split()

        # Not enough words for any possible repeat
        if len(words) < 2:
            return []

        found_repeats = set()
        n = len(words)

        # 2. Iterate through all possible phrase lengths
        # The longest possible repeating phrase can only be half the total length
        for phrase_len in range(1, (n // 2) + 1):

            # 3. Slide a window across the list of words to find adjacent phrases
            for i in range(n - 2 * phrase_len + 1):
                phrase1 = words[i : i + phrase_len]
                phrase2 = words[i + phrase_len : i + 2 * phrase_len]

                # 4. If two adjacent phrases are identical, add to our set
                if phrase1 == phrase2:
                    found_repeats.add(" ".join(phrase1))

        return list(found_repeats)

# Example 1: Simple word and phrase repeats
text1 = "I want to go to New York New York because it's a city city"
repeats1 = TextUtils.get_repeats(text1)
print(f"Text: \"{text1}\"")
print(f"Repeats: {repeats1}\n")
# Expected (order may vary): ['to', 'new york', 'city']

# Example 2: Multiple repeats of the same word
text2 = "Let's go go go to the store"
repeats2 = TextUtils.get_repeats(text2)
print(f"Text: \"{text2}\"")
print(f"Repeats: {repeats2}\n")
# Expected: ['go']

# Example 3: No adjacent repeats
text3 = "hello world hello"
repeats3 = TextUtils.get_repeats(text3)
print(f"Text: \"{text3}\"")
print(f"Repeats: {repeats3}\n")
# Expected: []

# Example 3: No adjacent repeats
text4 = "1 2 3 4 two years ago a friend of mine two years ago a friend of mine hello "
repeats4 = TextUtils.get_repeats(text4)
print(f"Text: \"{text4}\"")
print(f"Repeats: {repeats4}\n")
# Expected: []

###

ref_set = set()
trans_set = { "new york", "los angeles" }
trans_only = trans_set - ref_set
print("a", trans_only)

ref_set = { "new york", "los angeles" }
trans_set = { "new york", "los angeles" }
trans_only = trans_set - ref_set
print("b", trans_only)

ref_set = { "new york" }
trans_set = { "new york", "los angeles", "nothingburger" }
trans_only = trans_set - ref_set
print("c", trans_only)
