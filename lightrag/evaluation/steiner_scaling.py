"""Synthetic stage-2 latency only; no answer-quality or retrieval-speed claim."""

import argparse
import asyncio
import json
import math
import random
import statistics
from pathlib import Path
from time import perf_counter

import tiktoken

from lightrag import QueryParam
from lightrag.evaluation.steiner_ablation import ARMS
from lightrag.operate import _apply_token_truncation
from lightrag.utils import Tokenizer, logger


async def main(args):
    rng = random.Random(3866)
    tokenizer = Tokenizer("cl100k_base", tiktoken.get_encoding("cl100k_base"))
    records = []
    for size in args.sizes:
        n = size // 2
        entities = [
            {
                "entity_name": f"Entity {i}",
                "description": f"Entity {i} belongs to project {i % 5}. "
                + "Background information. " * (i % 5),
            }
            for i in range(n)
        ]
        # One fifth of entities are deliberately isolated.
        connected = max(2, n * 4 // 5)
        relations = [
            {
                "src_tgt": (f"Entity {i % connected}", f"Entity {(i + 1) % connected}"),
                "description": f"Record {i} links two project participants. "
                + "Supporting evidence. " * (i % 3),
            }
            for i in range(size - n)
        ]
        snapshot = {"final_entities": entities, "final_relations": relations}
        param = QueryParam(
            max_entity_tokens=max(40, n * 12),
            max_relation_tokens=max(40, (size - n) * 12),
        )
        for repeat in range(-1, args.repeats):
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                started = perf_counter()
                result = await _apply_token_truncation(
                    snapshot,
                    param,
                    {
                        "tokenizer": tokenizer,
                        "addon_params": {
                            "context_selection": {
                                "strategy": arm,
                                "max_candidates": args.max_candidates,
                            }
                        },
                    },
                )
                elapsed = (perf_counter() - started) * 1000
                records.append(
                    {
                        "candidate_count": size,
                        "arm": arm,
                        "repeat": repeat,
                        "warmup": repeat == -1,
                        "selector_ms": elapsed,
                        "selection": result.get(
                            "selection_metadata",
                            {"solver_calls": 0, "executed_strategy": "rank"},
                        ),
                    }
                )
    summary = []
    for size in args.sizes:
        for arm in ARMS:
            rows = [
                r
                for r in records
                if r["candidate_count"] == size and r["arm"] == arm and not r["warmup"]
            ]
            values = sorted(r["selector_ms"] for r in rows)
            summary.append(
                {
                    "candidate_count": size,
                    "arm": arm,
                    "p50_ms": statistics.median(values),
                    "p95_ms": values[max(0, math.ceil(0.95 * len(values)) - 1)],
                    "fallback_rate": statistics.mean(
                        bool(r["selection"].get("fallback_reason")) for r in rows
                    ),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"kind": "synthetic_selector_scaling", "summary": summary, "runs": records},
            indent=2,
        )
    )
    logger.info("Saved scaling results to %s", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[30, 60, 120, 240])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument(
        "--output", type=Path, default=Path("temp/steiner-scaling.json")
    )
    asyncio.run(main(parser.parse_args()))
