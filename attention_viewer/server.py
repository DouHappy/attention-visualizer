from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data_loader import AttentionDataset
from .tokenizer_utils import (
    align_tokens_to_attention,
    build_chat_tokens,
    compute_correction_indices,
    load_tokenizer,
)


def create_app(
    data_dir: str | Path,
    tokenizer_name: str,
    trust_remote_code: bool = False,
) -> FastAPI:
    """Create the FastAPI application that serves the visualisation."""

    dataset = AttentionDataset(data_dir)
    tokenizer = load_tokenizer(tokenizer_name, trust_remote_code=trust_remote_code)

    app = FastAPI(title="Attention Viewer")

    static_dir = Path(__file__).parent / "static"
    templates_dir = Path(__file__).parent / "templates"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(templates_dir / "index.html")

    @app.get("/api/files")
    async def list_files() -> Dict[str, Any]:
        files = [
            {
                "name": file.name,
                "start_id": file.start_id,
                "end_id": file.end_id,
            }
            for file in dataset.files()
        ]
        min_id = min((item["start_id"] for item in files), default=0)
        max_id = max((item["end_id"] for item in files), default=0)
        return {
            "files": files,
            "sample_id_range": {
                "min": min_id,
                "max": max_id - 1 if files else 0,
            },
        }

    @app.get("/api/attention")
    async def get_attention(
        sample_id: int = Query(..., ge=0),
        layer: int = Query(0, ge=0),
        head: int = Query(0, ge=0),
    ) -> Dict[str, Any]:
        try:
            sample = dataset.get_sample(sample_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if layer >= sample.layer_count:
            raise HTTPException(
                status_code=400,
                detail=f"Layer index {layer} is out of range (0-{sample.layer_count - 1}).",
            )
        if head >= sample.head_count:
            raise HTTPException(
                status_code=400,
                detail=f"Head index {head} is out of range (0-{sample.head_count - 1}).",
            )

        attn_matrix: np.ndarray = sample.attentions[layer, head]

        tokenised = build_chat_tokens(tokenizer, sample.source, sample.prediction)
        # Attention matrices in the dataset already omit any system prompt tokens,
        # so we align the visualised tokens by dropping the same prefix.
        visible_offset = tokenised.source_span[0] if tokenised.source_span else 0
        visible_token_ids = tokenised.token_ids[visible_offset:]
        visible_tokens = tokenised.tokens[visible_offset:]

        tokens, trimmed_attention = align_tokens_to_attention(
            visible_token_ids,
            visible_tokens,
            attn_matrix,
            pad_token_id=tokenizer.pad_token_id,
        )
        correction_source, correction_prediction = compute_correction_indices(tokenised)
        if visible_offset:
            correction_source = [idx - visible_offset for idx in correction_source if idx >= visible_offset]
            correction_prediction = [
                idx - visible_offset for idx in correction_prediction if idx >= visible_offset
            ]
        max_length = len(tokens)
        correction_source = [idx for idx in correction_source if idx < max_length]
        correction_prediction = [idx for idx in correction_prediction if idx < max_length]
        attention_with_instruction = trimmed_attention.astype(float)
        if attention_with_instruction.ndim != 2:
            raise ValueError("Expected attention matrix to be 2-dimensional after trimming")

        row_sums = attention_with_instruction.sum(axis=1)
        instruction_column = np.clip(1.0 - row_sums, a_min=0.0, a_max=None)[:, np.newaxis]
        attention = np.concatenate(
            [instruction_column, attention_with_instruction], axis=1
        ).tolist()

        sequence_length = attention_with_instruction.shape[0]
        tokens_with_instruction = ["Instruction"] + tokens

        instruction_offset = 1
        max_token_index = len(tokens_with_instruction)
        correction_source = [
            idx + instruction_offset
            for idx in correction_source
            if idx + instruction_offset < max_token_index
        ]
        correction_prediction = [
            idx + instruction_offset
            for idx in correction_prediction
            if idx + instruction_offset < max_token_index
        ]

        return {
            "sample_id": sample_id,
            "layer": layer,
            "head": head,
            "layer_count": sample.layer_count,
            "head_count": sample.head_count,
            "sequence_length": int(sequence_length),
            "tokens": tokens_with_instruction,
            "attention": attention,
            "source": sample.source,
            "prediction": sample.prediction,
            "corrections": {
                "source_indices": correction_source,
                "prediction_indices": correction_prediction,
            },
            "has_instruction_column": True,
            "file": {
                "name": sample.file.name,
                "start_id": sample.file.start_id,
                "end_id": sample.file.end_id,
                "batch_index": sample.batch_index,
            },
        }

    return app

