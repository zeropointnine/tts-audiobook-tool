class ParseUtil:

    @staticmethod
    def parse_ranges_string(string: str, max_one_indexed: int) -> tuple[ set[int], list[str] ]:
        """
        Expects a comma-delimited list of one-indexed ints and/or int ranges. Eg, "1, 3, 6-8"
        Returns tuple of zero-indexed index values and warning strings (eg, 1,3,6,7,8)
        """

        ints = []
        warnings: list[str] = []

        tokens = split_and_strip(string, ",")
        if not tokens:
            return (set(), [])

        for token in tokens:
            if token.isdigit():
                value = int(token)
                if value < 1 or value > max_one_indexed:
                    warnings.append(f"Out of range: {value}")
                else:
                    ints.append(value - 1)
            else:
                items = ParseUtil._parse_range_string(token, max_one_indexed)
                if not items:
                    warnings.append(f"Bad value: {token}")
                else:
                    ints.extend(items)

        ints = set(ints)

        return ints, warnings

    @staticmethod
    def make_ranges_string(zero_indexed_ints: set[int], max_one_indexed: int) -> str:
        """
        Returns a string of one-indexed values in this format: "1, 3-5, 7-10". Or just "all".
        """
        if not zero_indexed_ints:
            return "none"

        ints_list = sorted(list(set(zero_indexed_ints)))
        one_indexed_parts: list[ int | tuple[int, int] ] = []

        i = 0
        while i < len(ints_list):
            start = ints_list[i]
            end = start
            while i + 1 < len(ints_list) and ints_list[i+1] == end + 1:
                end = ints_list[i+1]
                i += 1

            if start == end:
                one_indexed_parts.append(start + 1)
            else:
                one_indexed_parts.append((start + 1, end + 1))
            i += 1

        if len(one_indexed_parts) == 1:
            item = one_indexed_parts[0]
            if isinstance(item, tuple):
                if item[0] == 1 and item[1] == max_one_indexed:
                    return "all"

        strings = []
        for item in one_indexed_parts:
            if isinstance(item, int):
                strings.append(str(item))
            else:
                strings.append(f"{item[0]}-{item[1]}")
        return ", ".join(strings)

    @staticmethod
    def _parse_range_string(string: str, max_one_indexed: int) -> list[int]:
        """
        Expects a string like "5-10" of one-indexed values. Or, "-5" or "5-".
        Returns zero-indexed list of expanded ints.
        Or empty string on parse error.
        """
        if not string:
            return []

        # All ints up to n (eg, "-5")
        if string[0] == "-":
            string = string[1:]
            if not string.isdigit():
                return []
            value = int(string)
            if value < 1:
                return []
            if value > max_one_indexed:
                value = max_one_indexed
            return [i for i in range(0, value)]

        # All ints from n to max (eg, "5-")
        if string[-1] == "-":
            string = string[:-1]
            if not string.isdigit():
                return []
            value = int(string)
            if value < 1:
                return []
            if value > max_one_indexed:
                return []
            return [i for i in range(value -1 , max_one_indexed)]

        # Eg, "5-10"
        tokens = string.split("-")
        if len(tokens) != 2:
            return []
        a = tokens[0]
        b = tokens[1]
        if not a.isdigit() or not b.isdigit():
            return []
        a = int(a)
        b = int(b)
        if b < a:
            return []
        if a < 1 or b < 1:
            return []
        if b > max_one_indexed:
            b = max_one_indexed
        a -= 1
        b -= 1
        return [i for i in range(a, b + 1)]

    @staticmethod
    def parse_range_string_normal(string: str, max_one_indexed) -> tuple[int, int] | str:
        """
        Parses a string which represents a range of one-indexed ints.
        Expects one of the following formats: x-y; x-; -y; z; "all"/"a"
        Returns returns tuple of ints, or parse error message; (0,0) signifies "all"
        """

        string = string.strip()

        if string.lower() in ["all", "a"]:
            return (0,0)

        strings = string.split("-")
        strings = [string.strip() for string in strings]

        if len(strings) == 0:
            return "No input"

        if len(strings) > 2:
            return "Bad format"

        if len(strings) == 1:
            # Single item means "start at a"
            str_a = strings[0]
            str_b = str(max_one_indexed)
        else: # len == 2
            str_a, str_b = strings
            if not str_a: # eg, "-10", meaning 1-10
                str_a = str(1)
            if not str_b: # eg, "5-", meaning 5 to the end
                str_b = str(max_one_indexed)

        if not str_a.isdigit():
            return f"Must be a number {str_a}"
        if not str_b.isdigit():
            return f"Must be a number {str_b}"

        int_a = int(str_a)
        int_b = int(str_b)
        if int_a < 1 or int_a > max_one_indexed:
            return f"Out of range: {int_a}"
        if int_b < 1:
            return f"Out of range: {int_b}"
        if int_b > max_one_indexed:
            int_b = max_one_indexed # silently clamp
        if int_a > int_b:
            return "Bad values"

        return (int_a, int_b)

# ---

def split_and_strip(s: str, delimiter: str) -> list[str]:
    return [item.strip() for item in s.split(delimiter) if item and item.strip()]
