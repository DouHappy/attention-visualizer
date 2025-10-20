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

    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)

    return TokenisationResult(token_ids=token_ids, tokens=tokens, text=text)


def align_tokens_to_attention(tokens: Sequence[str], expected_length: int) -> List[str]:
    """Ensure the token list matches the attention matrix size."""

    tokens = list(tokens)
    if len(tokens) == expected_length:
        return tokens

    if len(tokens) > expected_length:
        return tokens[-expected_length:]

    # Pad with placeholders to avoid crashing the UI.
    padding = ["<pad>"] * (expected_length - len(tokens))
    return tokens + padding

