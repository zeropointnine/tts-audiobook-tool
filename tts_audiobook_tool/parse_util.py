class ParseUtil:

    @staticmethod
    def parse_int_list(string: str) -> tuple[ list[int], list[str] ]:
        """
        Expects a comma-delimited list of ints and/or int ranges.
        Eg, "1, 2, 5-10"
        Returns tuple of values and warning strings.
        """

        values = []
        warnings: list[str] = []

        tokens = split_and_strip(string, ",")
        if not tokens:
            return ([], [])

        for token in tokens:
            if token.isdigit():
                values.append(int(token))
            else:
                items = parse_int_range_string(token)
                if not items:
                    warnings.append(f"Bad value: {token}")
                else:
                    values.extend(items)

        return values, warnings

# ---

def parse_int_range_string(string: str) -> list[int]:
    """
    Expects a string like "5-10" and returns list of ints in that range, else empty list
    """
    tokens = string.split("-")
    if len(tokens) != 2:
        return []
    a = tokens[0]
    b = tokens[1]
    if not a.isdigit() or not b.isdigit():
        return []
    a = int(a)
    b = int(b)
    result = []
    if a <= b:
        for i in range(a, b + 1):
            result.append(i)
    else:
        for i in range(a, b - 1, -1):
            result.append(i)
    return result

def split_and_strip(s: str, delimiter: str) -> list[str]:
    """Use this to prevent unintended whitespace and empty items"""
    return [item.strip() for item in s.split(delimiter) if item and item.strip()]  # note "if s"
