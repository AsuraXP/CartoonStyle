"""Text normalization and word-span utilities.

Matching between narration text, anchor phrases and TTS boundary events is
done on normalized words: case-folded, with punctuation stripped from the
edges of each token (internal apostrophes and hyphens are kept).  The
original character spans are preserved so matched text can be reported
verbatim from the source narration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Punctuation stripped from token edges when normalizing.
_EDGE_PUNCT = ".,;:!?\"'()[]{}<>\u2014\u2013\u201c\u201d\u2018\u2019\u00ab\u00bb\u2026*#|/\\"


@dataclass(frozen=True)
class Word:
    """One whitespace token of a source text."""

    index: int
    text: str  # verbatim token
    norm: str  # normalized token used for matching
    start: int  # char offset of token start in source text
    end: int  # char offset of token end in source text


def norm_word(token: str) -> str:
    """Normalize one token: casefold and strip edge punctuation."""
    w = token.casefold().strip(_EDGE_PUNCT)
    # Normalize typographic apostrophes / dashes inside words.
    w = w.replace("\u2019", "'").replace("\u2018", "'")
    w = w.replace("\u2014", "-").replace("\u2013", "-")
    return w


def tokenize(text: str) -> List[Word]:
    """Split text into Word tokens with character spans and normalized forms.

    Empty/whitespace-only tokens are skipped.  Tokens that normalize to the
    empty string (pure punctuation) are kept as words only if they carry a
    non-empty raw form; they normalize to "" and are ignored by sequence
    matching via ``significant_indices``.
    """
    words: List[Word] = []
    for m in re.finditer(r"\S+", text):
        raw = m.group(0)
        words.append(
            Word(
                index=len(words),
                text=raw,
                norm=norm_word(raw),
                start=m.start(),
                end=m.end(),
            )
        )
    return words


def significant_indices(words: List[Word]) -> List[int]:
    """Indices of words whose normalized form is non-empty."""
    return [w.index for w in words if w.norm]


def normalize_sequence(words: List[Word]) -> List[str]:
    """Normalized word sequence, skipping punctuation-only tokens."""
    return [w.norm for w in words if w.norm]


def find_phrase_occurrences(words: List[Word], phrase: str) -> List[int]:
    """Find all occurrences of phrase in words.

    Returns a list of *word indices* (into ``words``) where the normalized
    phrase word-sequence begins.  Punctuation-only tokens in the source do
    not break a match.  The returned index is the index into ``words`` of
    the first significant word of the match (the token that starts the
    phrase as spoken).
    """
    phrase_norms = [n for n in (norm_word(t) for t in phrase.split()) if n]
    if not phrase_norms:
        return []

    # Build significant sequence with back-mapping to original indices.
    sig = [w for w in words if w.norm]
    sig_norms = [w.norm for w in sig]
    n = len(phrase_norms)

    hits: List[int] = []
    for i in range(0, len(sig_norms) - n + 1):
        if sig_norms[i : i + n] == phrase_norms:
            hits.append(sig[i].index)
    return hits


def slice_text(text: str, words: List[Word], start_word: int, end_word: int) -> str:
    """Verbatim source slice covering words[start_word:end_word] (end exclusive)."""
    if not words or start_word < 0 or end_word > len(words) or start_word >= end_word:
        return ""
    return text[words[start_word].start : words[end_word - 1].end]


def count_words(text: str) -> int:
    """Number of significant (spoken) words in text."""
    return len([w for w in tokenize(text) if w.norm])
