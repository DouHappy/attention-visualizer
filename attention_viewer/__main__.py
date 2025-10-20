from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from .server import create_app

cli = typer.Typer(help="Launch the attention visualisation web application.")


@cli.command()
def run(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True, resolve_path=True, help="Directory containing NPZ attention dumps."),
    tokenizer: str = typer.Option(..., help="Hugging Face model name used to load the tokenizer."),
    host: str = typer.Option("0.0.0.0", help="Host interface to bind the web server to."),
    port: int = typer.Option(8000, help="Port to expose the web server on."),
    reload: bool = typer.Option(False, help="Automatically reload when files change."),
    trust_remote_code: bool = typer.Option(False, help="Allow loading tokenizers that require remote code."),
) -> None:
    """Run the FastAPI application via uvicorn."""

    app = create_app(data_dir=data_dir, tokenizer_name=tokenizer, trust_remote_code=trust_remote_code)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
