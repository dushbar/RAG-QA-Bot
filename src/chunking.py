"""Recursive character-based text splitter.

Tries each separator in order (paragraph -> line -> sentence -> word -> char)
and only falls through to a finer-grained separator when a piece is still
too big, so chunks break on natural boundaries whenever possible.
"""
from typing import List

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    if not text or not text.strip():
        return []
    chunks = _recursive_split(text, chunk_size, chunk_overlap, DEFAULT_SEPARATORS)
    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    separator = separators[-1]
    remaining_separators: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    splits = text.split(separator) if separator else list(text)

    final_chunks: List[str] = []
    good_splits: List[str] = []
    for piece in splits:
        if len(piece) < chunk_size:
            good_splits.append(piece)
            continue
        if good_splits:
            final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
            good_splits = []
        if remaining_separators:
            final_chunks.extend(_recursive_split(piece, chunk_size, chunk_overlap, remaining_separators))
        else:
            final_chunks.append(piece)  # nothing finer left to split on

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
    return final_chunks


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack small pieces into chunks up to chunk_size, carrying the
    tail of each chunk into the next one so context isn't lost at boundaries.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        extra = len(piece) + (sep_len if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(separator.join(current))
            # keep a trailing slice of the last chunk as overlap
            overlap: List[str] = []
            overlap_len = 0
            for p in reversed(current):
                add = len(p) + (sep_len if overlap else 0)
                if overlap_len + add > chunk_overlap:
                    break
                overlap.insert(0, p)
                overlap_len += add
            current = overlap
            current_len = overlap_len
            extra = len(piece) + (sep_len if current else 0)

        current.append(piece)
        current_len += extra

    if current:
        chunks.append(separator.join(current))
    return chunks
