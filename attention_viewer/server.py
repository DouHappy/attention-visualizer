from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data_loader import AttentionDataset
from .tokenizer_utils import align_tokens_to_attention, build_chat_tokens, load_tokenizer


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
        attention = attn_matrix.astype(float).tolist()

        tokenised = build_chat_tokens(tokenizer, sample.source, sample.prediction)
        tokens = align_tokens_to_attention(tokenised.tokens, sample.sequence_length)

        return {
            "sample_id": sample_id,
            "layer": layer,
            "head": head,
            "layer_count": sample.layer_count,
            "head_count": sample.head_count,
            "sequence_length": sample.sequence_length,
            "tokens": tokens,
            "attention": attention,
            "source": sample.source,
            "prediction": sample.prediction,
            "file": {
                "name": sample.file.name,
                "start_id": sample.file.start_id,
                "end_id": sample.file.end_id,
                "batch_index": sample.batch_index,
            },
        }

    return app

