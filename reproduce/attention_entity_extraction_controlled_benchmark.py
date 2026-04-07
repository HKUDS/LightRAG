import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import json_repair
from openai import AsyncOpenAI
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "lightrag" / "prompt.py"
TYPES_PATH = REPO_ROOT / "lightrag" / "types.py"

DEFAULT_LLM_BASE_URL = os.getenv("BENCH_LLM_BASE_URL", "https://api.moonshot.cn/v1")
DEFAULT_LLM_MODEL = os.getenv("BENCH_LLM_MODEL", "kimi-k2-0905-preview")
DEFAULT_LANGUAGE = os.getenv("BENCH_LANGUAGE", "English")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PROMPTS = load_module(PROMPT_PATH, "controlled_benchmark_prompts").PROMPTS
TYPES_MODULE = load_module(TYPES_PATH, "controlled_benchmark_types")
TYPES_MODULE.EntityExtractionResult.model_rebuild(
    _types_namespace={
        "ExtractedEntity": TYPES_MODULE.ExtractedEntity,
        "ExtractedRelationship": TYPES_MODULE.ExtractedRelationship,
    }
)
EntityExtractionResult = TYPES_MODULE.EntityExtractionResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled benchmark for PR2864 entity extraction stability."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "benchmark_entity_extraction_prompt_ab_test_results.json"),
        help="Where to write the JSON result file.",
    )
    parser.add_argument("--chunk-token-size", type=int, default=450)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--run-variants",
        default="",
        help="Comma-separated variant names. Empty means run all variants.",
    )
    return parser.parse_args()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return "\n\n".join(page for page in pages if page)


def chunk_text(content: str, chunk_token_size: int, chunk_overlap: int) -> list[dict]:
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = enc.encode(content)
    step = chunk_token_size - chunk_overlap
    chunks = []
    for index, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start : start + chunk_token_size]
        chunks.append(
            {
                "chunk_order_index": index,
                "tokens": len(chunk_tokens),
                "content": enc.decode(chunk_tokens).strip(),
            }
        )
    return chunks


def build_system_prompt(language: str) -> str:
    return PROMPTS["entity_extraction_json_system_prompt"].format(
        entity_types_guidance=PROMPTS["default_entity_types_guidance"],
        examples="\n".join(PROMPTS["entity_extraction_json_examples"]),
        language=language,
    )


def build_user_prompt(include_type_guidance: bool, input_text: str, language: str) -> str:
    prompt = PROMPTS["entity_extraction_json_user_prompt"]
    if include_type_guidance:
        prompt = prompt.replace(
            "---Data to be Processed---\n<Input Text>",
            "---Data to be Processed---\n---Entity Types---\n{entity_types_guidance}\n\n<Input Text>",
        )
    return prompt.format(
        entity_types_guidance=PROMPTS["default_entity_types_guidance"],
        input_text=input_text,
        language=language,
    )


def normalize_name(name: str) -> str:
    value = (name or "").strip().lower()
    value = re.sub(r"[^\w\s\-]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def make_warning(
    variant: str,
    chunk_idx: int,
    warning_type: str,
    message: str,
    recovered_entities: int = 0,
    recovered_relations: int = 0,
) -> dict:
    return {
        "variant": variant,
        "chunk_order_index": chunk_idx,
        "warning_type": warning_type,
        "message": message,
        "recovered_entities": recovered_entities,
        "recovered_relations": recovered_relations,
    }


def parse_json_content(
    raw_content: str,
    variant: str,
    chunk_idx: int,
    warnings: list[dict],
) -> dict:
    if not raw_content or not raw_content.strip():
        warnings.append(
            make_warning(
                variant,
                chunk_idx,
                "empty_json_content",
                "LLM returned empty content; this chunk may lose all extractions.",
            )
        )
        return {"entities": [], "relationships": []}

    try:
        return json.loads(raw_content)
    except Exception as strict_error:
        try:
            repaired = json_repair.loads(raw_content)
            entities = repaired.get("entities", []) if isinstance(repaired, dict) else []
            relations = (
                repaired.get("relationships", []) if isinstance(repaired, dict) else []
            )
            warnings.append(
                make_warning(
                    variant,
                    chunk_idx,
                    "malformed_json_repaired",
                    f"Strict JSON parse failed: {strict_error}",
                    len(entities) if isinstance(entities, list) else 0,
                    len(relations) if isinstance(relations, list) else 0,
                )
            )
            return repaired if isinstance(repaired, dict) else {"entities": [], "relationships": []}
        except Exception as repair_error:
            warnings.append(
                make_warning(
                    variant,
                    chunk_idx,
                    "malformed_json_dropped",
                    f"Strict JSON parse failed ({strict_error}); repair also failed ({repair_error}).",
                )
            )
            return {"entities": [], "relationships": []}


async def call_json_object(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    variant: str,
    chunk_idx: int,
    warnings: list[dict],
    llm_model: str,
    max_tokens: int,
) -> tuple[dict, str]:
    response = await client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=600,
    )
    raw_content = response.choices[0].message.content or ""
    parsed = parse_json_content(raw_content, variant, chunk_idx, warnings)
    return parsed, raw_content


async def call_schema_mode(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    variant: str,
    chunk_idx: int,
    warnings: list[dict],
    llm_model: str,
    max_tokens: int,
) -> tuple[dict, str]:
    try:
        response = await client.chat.completions.parse(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=EntityExtractionResult,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=600,
        )
        message = response.choices[0].message
        raw_content = getattr(message, "content", None) or ""
        if hasattr(message, "parsed") and message.parsed is not None:
            parsed = json.loads(message.parsed.model_dump_json())
        else:
            parsed = parse_json_content(raw_content, variant, chunk_idx, warnings)
        return parsed, raw_content
    except Exception as exc:
        warnings.append(
            make_warning(
                variant,
                chunk_idx,
                "schema_parse_failed",
                f"Pydantic schema parse failed: {exc}",
            )
        )
        try:
            probe_result, raw_probe = await call_json_object(
                client,
                system_prompt,
                user_prompt,
                variant,
                chunk_idx,
                warnings,
                llm_model,
                max_tokens,
            )
            entities = probe_result.get("entities", [])
            relations = probe_result.get("relationships", [])
            if entities or relations:
                warnings.append(
                    make_warning(
                        variant,
                        chunk_idx,
                        "schema_parse_failed_salvageable",
                        "Diagnostic json_object probe recovered parseable extractions; strict schema may have discarded this chunk.",
                        len(entities) if isinstance(entities, list) else 0,
                        len(relations) if isinstance(relations, list) else 0,
                    )
                )
            return {"entities": [], "relationships": []}, raw_probe
        except Exception as probe_exc:
            warnings.append(
                make_warning(
                    variant,
                    chunk_idx,
                    "schema_probe_failed",
                    f"Diagnostic json_object probe also failed: {probe_exc}",
                )
            )
            return {"entities": [], "relationships": []}, ""


def summarize_variant(variant_name: str, chunk_results: list[dict], warnings: list[dict]) -> dict:
    raw_entity_mentions = 0
    raw_relation_mentions = 0
    unique_entities = set()
    unique_relations = set()
    for item in chunk_results:
        parsed = item["parsed"]
        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
        relations = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        if isinstance(entities, list):
            raw_entity_mentions += len(entities)
            for entity in entities:
                if isinstance(entity, dict):
                    name = normalize_name(str(entity.get("entity_name", "")))
                    if name:
                        unique_entities.add(name)
        if isinstance(relations, list):
            raw_relation_mentions += len(relations)
            for relation in relations:
                if isinstance(relation, dict):
                    src = normalize_name(str(relation.get("source_entity", "")))
                    tgt = normalize_name(str(relation.get("target_entity", "")))
                    if src and tgt:
                        unique_relations.add(tuple(sorted((src, tgt))))
    warning_counts = Counter(item["warning_type"] for item in warnings)
    return {
        "variant": variant_name,
        "chunk_count": len(chunk_results),
        "contiguous_chunks": [item["chunk_order_index"] for item in chunk_results]
        == list(range(len(chunk_results))),
        "raw_entity_mentions": raw_entity_mentions,
        "raw_relation_mentions": raw_relation_mentions,
        "raw_total_volume": raw_entity_mentions + raw_relation_mentions,
        "unique_entity_count": len(unique_entities),
        "unique_relation_count": len(unique_relations),
        "unique_total_volume": len(unique_entities) + len(unique_relations),
        "warning_counts": dict(sorted(warning_counts.items())),
        "warnings": warnings,
    }


async def run_variant(
    client: AsyncOpenAI,
    variant_name: str,
    include_type_guidance: bool,
    json_mode: str,
    chunks: list[dict],
    *,
    language: str,
    llm_model: str,
    max_tokens: int,
    max_concurrency: int,
) -> dict:
    system_prompt = build_system_prompt(language)
    warnings = []
    semaphore = asyncio.Semaphore(max_concurrency)
    start_time = time.time()

    async def process_chunk(chunk: dict) -> dict:
        chunk_idx = chunk["chunk_order_index"]
        user_prompt = build_user_prompt(include_type_guidance, chunk["content"], language)
        async with semaphore:
            if json_mode == "json_object":
                parsed, raw_content = await call_json_object(
                    client,
                    system_prompt,
                    user_prompt,
                    variant_name,
                    chunk_idx,
                    warnings,
                    llm_model,
                    max_tokens,
                )
            else:
                parsed, raw_content = await call_schema_mode(
                    client,
                    system_prompt,
                    user_prompt,
                    variant_name,
                    chunk_idx,
                    warnings,
                    llm_model,
                    max_tokens,
                )
        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
        relations = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        print(
            f"[{variant_name}] chunk {chunk_idx + 1}/{len(chunks)} -> "
            f"{len(entities) if isinstance(entities, list) else 0}E "
            f"{len(relations) if isinstance(relations, list) else 0}R",
            flush=True,
        )
        return {
            "chunk_order_index": chunk_idx,
            "tokens": chunk["tokens"],
            "parsed": parsed,
            "raw_content_preview": raw_content[:600],
        }

    chunk_results = await asyncio.gather(*(process_chunk(chunk) for chunk in chunks))
    chunk_results.sort(key=lambda item: item["chunk_order_index"])
    summary = summarize_variant(variant_name, chunk_results, warnings)
    summary["elapsed_s"] = round(time.time() - start_time, 4)
    summary["chunks"] = chunk_results
    return summary


async def main() -> None:
    args = parse_args()
    api_key = os.getenv("BENCH_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Please set BENCH_LLM_API_KEY (or OPENAI_API_KEY) before running this benchmark."
        )

    pdf_path = Path(args.pdf).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, args.chunk_token_size, args.chunk_overlap)
    if len(chunks) <= 20:
        raise SystemExit(
            f"chunk_count={len(chunks)} is not above 20; adjust chunk parameters."
        )

    variants = [
        {
            "name": "schema_no_type_guidance",
            "include_type_guidance": False,
            "json_mode": "schema",
        },
        {
            "name": "schema_with_type_guidance",
            "include_type_guidance": True,
            "json_mode": "schema",
        },
        {
            "name": "json_object_no_type_guidance",
            "include_type_guidance": False,
            "json_mode": "json_object",
        },
        {
            "name": "json_object_with_type_guidance",
            "include_type_guidance": True,
            "json_mode": "json_object",
        },
    ]
    selected = {item.strip() for item in args.run_variants.split(",") if item.strip()}
    if selected:
        variants = [variant for variant in variants if variant["name"] in selected]

    print(f"PDF: {pdf_path}", flush=True)
    print(
        f"Chunk params: size={args.chunk_token_size}, overlap={args.chunk_overlap}",
        flush=True,
    )
    print(f"Chunk count: {len(chunks)}", flush=True)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.llm_base_url,
        timeout=600,
    )
    try:
        results = []
        for variant in variants:
            print(f"\n=== {variant['name']} ===", flush=True)
            result = await run_variant(
                client,
                variant["name"],
                variant["include_type_guidance"],
                variant["json_mode"],
                chunks,
                language=args.language,
                llm_model=args.llm_model,
                max_tokens=args.max_tokens,
                max_concurrency=args.max_concurrency,
            )
            results.append(result)
            print(
                f"{variant['name']}: raw_volume={result['raw_total_volume']} "
                f"unique_volume={result['unique_total_volume']} "
                f"warnings={result['warning_counts']}",
                flush=True,
            )
    finally:
        await client.close()

    output = {
        "pdf_path": str(pdf_path),
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap": args.chunk_overlap,
        "chunk_count": len(chunks),
        "chunks_are_contiguous": [c["chunk_order_index"] for c in chunks]
        == list(range(len(chunks))),
        "results": results,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
