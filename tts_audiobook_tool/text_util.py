class TextUtil:


    @staticmethod
    def number_string_to_words(s: str) -> str:
        try:
            n = int(s.strip())
            return TextUtil.number_to_words(n)
        except:
            return s

    @staticmethod
    def number_to_words(n: int) -> str:
        """
        Converts an integer from 0 to 999 into its English word representation if applicable.
        """
        if not isinstance(n, int) or not (0 <= n <= 999):
            return str(n)

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
