# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A strategy for generating text embeddings using OpenAI's Batch API."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any


try:
    import openai
except Exception:  # pragma: no cover - openai may not be installed in tests
    openai = None  # type: ignore

from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.index.operations.embed_text.strategies.typing import TextEmbeddingResult
from graphrag.index.operations.embed_text.strategies import openai as openai_strategy
from graphrag.index.utils.is_null import is_null
from graphrag.language_model.manager import ModelManager
from graphrag.language_model.protocol.base import EmbeddingModel
from graphrag.logger.progress import ProgressTicker, progress_ticker

logger = logging.getLogger(__name__)


async def run(
    input: list[str],
    callbacks: WorkflowCallbacks,
    cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Generate embeddings using OpenAI's Batch API."""

    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    llm_config = LanguageModelConfig(**args["llm"])
    splitter = openai_strategy._get_splitter(llm_config, args.get("batch_max_tokens", 8191))
    model = ModelManager().get_or_create_embedding_model(
        name="text_embedding",
        model_type=llm_config.type,
        config=llm_config,
        callbacks=callbacks,
        cache=cache,
    )

    texts, input_sizes = openai_strategy._prepare_embed_texts(input, splitter)

    ticker = progress_ticker(
        callbacks.progress,
        1,
        description="generate embeddings progress: ",
    )

    embeddings = await _execute(model, texts, ticker, llm_config)
    embeddings = openai_strategy._reconstitute_embeddings(embeddings, input_sizes)

    return TextEmbeddingResult(embeddings=embeddings)


async def _execute(
    model: EmbeddingModel,
    texts: list[str],
    tick: ProgressTicker,
    llm_config: LanguageModelConfig,
) -> list[list[float]]:
    """Execute the batch embedding request."""

    # If openai library is missing or model is a mock, fall back to direct calls
    if openai is None or llm_config.type.endswith("mock_embedding"):
        result = await model.aembed_batch(texts)
        tick(1)
        return result

    client = openai.AsyncOpenAI(
        api_key=llm_config.api_key,
        organization=llm_config.organization,
        base_url=llm_config.api_base,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        request_file = Path(tmpdir) / "requests.jsonl"
        with request_file.open("w", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"input": text, "model": llm_config.model}) + "\n")

        file = await client.files.create(file=request_file.open("rb"), purpose="batch")
        batch = await client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )

        status = batch.status
        while status in {"validating", "in_progress"}:
            await asyncio.sleep(1)
            batch = await client.batches.retrieve(batch.id)
            status = batch.status
            tick(0)

        if status != "completed":
            msg = f"Batch failed with status {status}"
            raise RuntimeError(msg)

        # Download the output file and parse embeddings
        output = await client.files.content(batch.output_file_id)
        content = (await output.aread()).decode("utf-8").splitlines()

    embeddings = [json.loads(line)["embedding"] for line in content]
    tick(1)
    return embeddings
