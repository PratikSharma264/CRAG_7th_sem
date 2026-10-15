"""Sentence-aware text chunking."""
MAX_CHUNK_SIZE = 2000


def create_chunks(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """Split text at sentence boundaries where possible, respecting max_size."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in text.split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        piece = sentence if sentence.endswith(".") else sentence + "."
        if len(piece) > max_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(piece[i:i + max_size] for i in range(0, len(piece), max_size))
        elif not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_size:
            current += " " + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks
