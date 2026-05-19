import os

import xxhash


def calc_hash_string(string: str) -> str:
    return xxhash.xxh3_64(string).hexdigest()


def calc_hash_file(path: str, print_progress_text: str = "") -> tuple[str, str]:
    """ Returns hash and error string, mutually exclusive"""

    if not os.path.exists(path):
        return "", f"File not found: {path}"
    if os.path.isdir(path):
        return "", f"Is not a file: {path}"
    if not os.access(path, os.R_OK):
        return "", f"No read permission for file: {path}"

    if print_progress_text:
        print_progress_text = print_progress_text.strip()

    hasher = xxhash.xxh64()
    file_size = os.path.getsize(path)
    processed = 0

    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
                processed += len(chunk)
                if print_progress_text:
                    print(f"\r{print_progress_text} {processed / file_size:.1%} ", end="")
    except Exception as e:
        return "", f"Error while hashing file: {e}"

    if print_progress_text:
        print("\r", end="")

    return hasher.hexdigest(), ""


def is_app_hash(hash: str) -> bool:
    """ The app uses 16-character hex string for hash values """
    return len(hash) == 16 and all(c in "0123456789abcdefABCDEF" for c in hash)