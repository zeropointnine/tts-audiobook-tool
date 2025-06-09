class ParseUtil:

    @staticmethod
    def parse_one_indexed_ranges_string(string: str, max_one_indexed: int) -> tuple[ set[int], list[str] ]:
        """
        Expects a comma-delimited list of one-indexed ints and/or int ranges.
        Eg, "1, 2, 5-10"

        Returns tuple of zero-indexed index values and warning strings.
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
                items = parse_one_indexed_range_string(token, max_one_indexed)
                if not items:
                    warnings.append(f"Bad value: {token}")
                else:
                    ints.extend(items)

        ints = set(ints)

        return ints, warnings

    @staticmethod
    def make_one_indexed_ranges_string(zero_indexed_ints: set[int], max_one_indexed: int) -> str:
        """
        Returns a string of one-indexed values in this format: "1, 3-5, 7-10"
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

# ---

def split_and_strip(s: str, delimiter: str) -> list[str]:
    return [item.strip() for item in s.split(delimiter) if item and item.strip()]

def parse_one_indexed_range_string(string: str, max_one_indexed: int) -> list[int]:
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
