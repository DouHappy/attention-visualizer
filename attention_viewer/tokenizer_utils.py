from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer


@dataclass
class TokenisationResult:
    token_ids: List[int]
    tokens: List[str]
    text: str
    source_token_ids: List[int]
    prediction_token_ids: List[int]
    source_tokens: List[str]
    prediction_tokens: List[str]
    source_span: Optional[Tuple[int, int]]
    prediction_span: Optional[Tuple[int, int]]


def load_tokenizer(model_name: str, trust_remote_code: bool = False) -> AutoTokenizer:
    """Load a Hugging Face tokenizer with sensible defaults."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


def build_chat_tokens(
    tokenizer: AutoTokenizer,
    source: str,
    prediction: str,
) -> TokenisationResult:
    """Tokenise the chat conversation using the model's chat template."""

    messages = [
        {"role": "user", "content": source},
        {"role": "assistant", "content": prediction},
    ]

    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="np",
    )

    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.reshape(-1).tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    else:
        token_ids = list(token_ids)

    tokens = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]

    source_token_ids = tokenizer.encode(
        source,
        add_special_tokens=False,
        return_tensors=None,
    )
    prediction_token_ids = tokenizer.encode(
        prediction,
        add_special_tokens=False,
        return_tensors=None,
    )

    if isinstance(source_token_ids, np.ndarray):
        source_token_ids = source_token_ids.reshape(-1).tolist()
    source_token_ids = list(source_token_ids)

    if isinstance(prediction_token_ids, np.ndarray):
        prediction_token_ids = prediction_token_ids.reshape(-1).tolist()
    prediction_token_ids = list(prediction_token_ids)

    source_tokens = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in source_token_ids
    ]
    prediction_tokens = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in prediction_token_ids
    ]

    source_span = _find_subsequence(token_ids, source_token_ids)
    prediction_span = _find_subsequence(token_ids, prediction_token_ids)
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return TokenisationResult(
        token_ids=token_ids,
        tokens=tokens,
        text=text,
        source_token_ids=source_token_ids,
        prediction_token_ids=prediction_token_ids,
        source_tokens=source_tokens,
        prediction_tokens=prediction_tokens,
        source_span=source_span,
        prediction_span=prediction_span,
    )


def align_tokens_to_attention(
    token_ids: Sequence[int],
    tokens: Sequence[str],
    attention: np.ndarray,
    pad_token_id: int | None = None,
) -> tuple[List[str], np.ndarray]:
    """Trim padding tokens and attention matrix to align lengths."""

    token_ids = list(token_ids)
    tokens = list(tokens)

    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("Attention matrix must be square (seq_len x seq_len).")

    max_length = min(len(token_ids), len(tokens), attention.shape[0])
    valid_length = max_length

    if pad_token_id is not None:
        while valid_length > 0 and token_ids[valid_length - 1] == pad_token_id:
            valid_length -= 1

    tokens = tokens[:valid_length]
    trimmed_attention = attention[:valid_length, :valid_length]

    return tokens, trimmed_attention


def _find_subsequence(
    haystack: Sequence[int], needle: Sequence[int]
) -> Optional[Tuple[int, int]]:
    """Return the (start, end) indices if *needle* appears in *haystack*."""

    haystack = list(haystack)
    needle = list(needle)

    if not needle:
        return None

    hay_len = len(haystack)
    needle_len = len(needle)
    if needle_len > hay_len:
        return None

    for start in range(hay_len - needle_len + 1):
        if haystack[start : start + needle_len] == needle:
            return start, start + needle_len
    return None


def _compute_edit_distance_masks(
    source_tokens: Sequence[str], prediction_tokens: Sequence[str]
) -> Tuple[List[int], List[int]]:
    """Compute indices participating in the minimal edit distance alignment."""

    m = len(source_tokens)
    n = len(prediction_tokens)

    if m == 0 and n == 0:
        return [], []

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if source_tokens[i - 1] == prediction_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    i, j = m, n
    source_indices: List[int] = []
    prediction_indices: List[int] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if source_tokens[i - 1] == prediction_tokens[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    source_indices.append(i - 1)
                    prediction_indices.append(j - 1)
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            source_indices.append(i - 1)
            i -= 1
            continue
        if j > 0:
            prediction_indices.append(j - 1)
            j -= 1

    source_indices.reverse()
    prediction_indices.reverse()
    return source_indices, prediction_indices


def compute_correction_indices(
    tokenisation: TokenisationResult,
) -> Tuple[List[int], List[int]]:
    """Return global token indices that participate in corrections."""

    if not tokenisation.source_tokens and not tokenisation.prediction_tokens:
        return [], []

    source_local, prediction_local = _compute_edit_distance_masks(
        tokenisation.source_tokens,
        tokenisation.prediction_tokens,
    )

    source_indices: List[int] = []
    prediction_indices: List[int] = []

    if tokenisation.source_span is not None:
        start, _ = tokenisation.source_span
        source_indices = [start + idx for idx in source_local]

    if tokenisation.prediction_span is not None:
        start, _ = tokenisation.prediction_span
        prediction_indices = [start + idx for idx in prediction_local]

    return source_indices, prediction_indices

