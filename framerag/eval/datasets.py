"""Dataset loaders for multi-hop QA evaluation.

Requires: pip install datasets
Each loader returns list[dict] with keys: id, question, gold_answers, supporting_docs.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def load_hotpotqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    config: str = "distractor",
) -> list[dict]:
    """Load HotpotQA (Yang et al. 2018).

    Args:
        split: 'train' | 'validation'
        max_samples: limit number of samples (None = all)
        config: 'distractor' (10 context paras) or 'fullwiki'
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("hotpot_qa", config, split=split, trust_remote_code=True)
    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        # supporting_docs: list of (title, sentences) pairs
        titles = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        supporting_docs = []
        for title, sents in zip(titles, sentences_list):
            doc = f"{title}\n" + " ".join(sents)
            supporting_docs.append(doc)

        answer = ex["answer"]
        gold_answers = [answer] if answer else []

        samples.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_answers": gold_answers,
            "supporting_docs": supporting_docs,
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
        })
    logger.info(f"[HotpotQA] Loaded {len(samples)} samples from {split}/{config}")
    return samples


def load_2wikimultihopqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load 2WikiMultiHopQA (Ho et al. 2020).

    HuggingFace dataset: 'voidful/2WikiMultihopQA' or 'KILT/hotpotqa' subset.
    Falls back to 'hfl/2WikiMultiHopQA' if primary fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    for hf_name in ["voidful/2WikiMultihopQA", "hfl/2WikiMultiHopQA"]:
        try:
            ds = load_dataset(hf_name, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load 2WikiMultiHopQA from HuggingFace. Try downloading manually.")

    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # Extract supporting documents from context field (varies by HF version)
        supporting_docs: list[str] = []
        context = ex.get("context", ex.get("passages", []))
        if isinstance(context, list):
            for ctx in context:
                if isinstance(ctx, dict):
                    title = ctx.get("title", "")
                    text = ctx.get("text", ctx.get("paragraph_text", ""))
                    supporting_docs.append(f"{title}\n{text}" if title else text)
                elif isinstance(ctx, str):
                    supporting_docs.append(ctx)
        elif isinstance(context, dict):
            for title, paras in context.items():
                if isinstance(paras, list):
                    supporting_docs.append(f"{title}\n" + " ".join(str(p) for p in paras))

        answer = ex.get("answer", ex.get("answers", [""]))[0] if isinstance(ex.get("answers", ex.get("answer", "")), list) else ex.get("answer", "")
        gold_answers = [answer] if answer else []

        samples.append({
            "id": str(ex.get("id", ex.get("_id", i))),
            "question": ex.get("question", ""),
            "gold_answers": gold_answers,
            "supporting_docs": supporting_docs,
            "type": ex.get("type", ""),
        })
    logger.info(f"[2WikiMultiHopQA] Loaded {len(samples)} samples from {split}")
    return samples


def load_musique(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load MuSiQue (Trivedi et al. 2022).

    HuggingFace dataset: 'drt/musique' or 'musique/musique'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    for hf_name in ["drt/musique", "Zenoverse/musique"]:
        try:
            ds = load_dataset(hf_name, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not load MuSiQue from HuggingFace.")

    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # MuSiQue has 'paragraphs': [{idx, title, paragraph_text, is_supporting}]
        paragraphs = ex.get("paragraphs", [])
        supporting_docs = []
        for para in paragraphs:
            if isinstance(para, dict):
                title = para.get("title", "")
                text = para.get("paragraph_text", para.get("text", ""))
                supporting_docs.append(f"{title}\n{text}" if title else text)

        answer = ex.get("answer", "")
        answer_aliases = ex.get("answer_aliases", [])
        gold_answers = [answer] + answer_aliases if answer else answer_aliases

        samples.append({
            "id": str(ex.get("id", i)),
            "question": ex.get("question", ""),
            "gold_answers": gold_answers,
            "supporting_docs": supporting_docs,
            "answerable": ex.get("answerable", True),
        })
    logger.info(f"[MuSiQue] Loaded {len(samples)} samples from {split}")
    return samples


def load_chronoqa(
    data_path: Optional[str] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load ChronoQA — narrative QA from the E²RAG paper (arXiv 2506.05939).

    ChronoQA tests temporal, causal, and character consistency in narrative QA.
    Dataset: https://huggingface.co/datasets/zy113/ChronoQA

    Args:
        data_path: optional local path to JSON file (overrides HuggingFace loader)
        split: 'train' | 'validation' | 'test'
        max_samples: limit samples (None = all)
    """
    # ── Local file override ────────────────────────────────────────────────────
    if data_path is not None:
        import json as _json
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ChronoQA file not found: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return _parse_chronoqa_records(raw, split=split, max_samples=max_samples,
                                       source=data_path)

    # ── HuggingFace loader ────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    try:
        ds = load_dataset("zy113/ChronoQA", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error(f"[ChronoQA] Failed to load from HuggingFace: {e}")
        logger.error("Install datasets: pip install datasets")
        return []

    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        samples.append(_parse_chronoqa_example(ex, i))

    logger.info(f"[ChronoQA] Loaded {len(samples)} samples (split={split}) from HuggingFace")
    return samples


def _parse_chronoqa_example(ex: dict, fallback_id: int) -> dict:
    """Normalise a single ChronoQA example to the standard loader format."""
    answer = ex.get("answer", ex.get("answers", ""))
    if isinstance(answer, list):
        gold_answers = [a for a in answer if a]
    else:
        gold_answers = [answer] if answer else []

    # ChronoQA stores the source document(s) under various keys
    supporting_docs = (
        ex.get("documents")
        or ex.get("passages")
        or ex.get("context")
        or ex.get("story")
        or []
    )
    if isinstance(supporting_docs, str):
        supporting_docs = [supporting_docs]

    return {
        "id":             str(ex.get("id", fallback_id)),
        "question":       ex.get("question", ""),
        "gold_answers":   gold_answers,
        "supporting_docs": supporting_docs,
        "type":           ex.get("type", ex.get("category", "")),
        "source":         ex.get("source", ex.get("book", "")),
    }


def _parse_chronoqa_records(
    raw: list,
    split: str,
    max_samples: Optional[int],
    source: str,
) -> list[dict]:
    """Parse a local JSON file (list of dicts) into the standard format."""
    samples = []
    for i, ex in enumerate(raw):
        if ex.get("split", split) != split:
            continue
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(_parse_chronoqa_example(ex, i))
    logger.info(f"[ChronoQA] Loaded {len(samples)} samples from {source} (split={split})")
    return samples
