"""Text normalization for knowledge-base documents."""
import re


def clean_text(text: str) -> str:
    """Trim and collapse whitespace without changing document wording."""
    return re.sub(r"\s+", " ", str(text)).strip()
