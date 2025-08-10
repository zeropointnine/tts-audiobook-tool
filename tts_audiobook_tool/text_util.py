import re
import string

class TextUtil:

    @staticmethod
    def expand_int_words_in_text(text: str) -> str:
        """
        Expands any "int words" (from 0 to 999) with spelled-out numbers.
        Eg, "Repeat 28 times." -> "Repeat twenty-eight times."
        """

        # This regex splits the text by whitespace, keeping the whitespace delimiters.
        tokens = re.split(r'(\s+)', text)

        expanded_tokens = []

        # Create a regex pattern for leading and trailing punctuation.
        punct_chars = re.escape(string.punctuation)
        leading_punct_re = re.compile(r'^([' + punct_chars + r']*)')
        trailing_punct_re = re.compile(r'([' + punct_chars + r']*)$')

        for token in tokens:
            if not token:
                continue

            # Pass through whitespace tokens.
            if token.isspace():
                expanded_tokens.append(token)
                continue

            # Find leading punctuation.
            leading_match = leading_punct_re.match(token)
            leading_punct = leading_match.group(1) if leading_match else ''

            # Find trailing punctuation.
            trailing_match = trailing_punct_re.search(token)
            trailing_punct = trailing_match.group(1) if trailing_match else ''

            # Extract the core word.
            start_index = len(leading_punct)
            end_index = len(token) - len(trailing_punct)

            # Handle cases where token is only punctuation or where matches overlap.
            if start_index >= end_index:
                expanded_tokens.append(token)
                continue

            core_word = token[start_index:end_index]

            # Expand the core word using the other static method.
            expanded_core = TextUtil._expand_int_word_or_pass_through(core_word)

            # Reconstruct the token with original punctuation.
            new_token = leading_punct + expanded_core + trailing_punct
            expanded_tokens.append(new_token)

        return "".join(expanded_tokens)

    @staticmethod
    def _expand_int_word_or_pass_through(word: str) -> str:

        word = word.strip()
        is_strict_int = word.isdigit() and (word == '0' or word[0] != '0') # digits only, no leading zeros before other numbers
        if not is_strict_int:
            return word

        n = int(word)
        if not (0 <= n <= 999):
            return word

        if n == 0:
            return "zero"

        # Define the building blocks for the number words
        LESS_THAN_20 = [
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        ]
        TENS = [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]

        parts = []

        # Step 1: Handle the hundreds place
        if n >= 100:
            # e.g., for 999, this gets the '9' from 999 // 100
            hundreds_digit = n // 100
            parts.append(f"{LESS_THAN_20[hundreds_digit]} hundred")
            # Get the remainder, e.g., 999 % 100 = 99
            n %= 100

        # Step 2: Handle the tens and ones place
        if n >= 20:
            # e.g., for 99, this gets the 'ninety' from TENS[99 // 10]
            parts.append(TENS[n // 10])
            # Get the remainder, e.g., 99 % 10 = 9
            n %= 10
            # If there's a one's digit, add it with a hyphen
            if n > 0:
                parts[-1] += f"-{LESS_THAN_20[n]}"
        elif n > 0:
            # This handles numbers from 1 to 19
            parts.append(LESS_THAN_20[n])

        # Step 3: Join the parts with spaces
        return " ".join(parts)
