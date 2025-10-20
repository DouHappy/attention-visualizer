from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from transformers import AutoTokenizer


@dataclass
class TokenisationResult:
    token_ids: List[int]
    tokens: List[str]
    text: str


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
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return TokenisationResult(token_ids=token_ids, tokens=tokens, text=text)


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

