from pathlib import Path


STYLE_ACCENT = "#ffaa44"
STYLE_ERROR = "#ff0000"
STYLE_DIM = "#888888"
STYLE_OK = "#00ff00"
STYLE_DEFAULT = "default"


def load_css(*filenames: str) -> str:
    """Load multiple TCSS files into one variable-substitution scope."""
    textual_dir = Path(__file__).parent
    return "\n".join(
        (textual_dir / filename).read_text(encoding="utf-8")
        for filename in filenames
    )
