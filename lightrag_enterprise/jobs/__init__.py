from pathlib import Path


async def run_sync_openrouter_catalog(
    output_path: str | Path = "rag_storage/model_catalog/openrouter_catalog.json",
) -> int:
    from .sync_openrouter_catalog import run_sync_openrouter_catalog as run

    return await run(output_path)

__all__ = ["run_sync_openrouter_catalog"]
