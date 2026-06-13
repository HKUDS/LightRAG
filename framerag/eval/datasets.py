"""Dataset loaders for multi-hop QA evaluation.

Requires: pip install datasets
Each loader returns list[dict] with keys: id, question, gold_answers, supporting_docs.
"""
from __future__ import annotations

import logging
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
    """Load ChronoQA (from arxiv 2506.05939 — Entity-Event KG for RAG).

    ChronoQA tests temporal, causal, and character consistency in narrative QA.

    Args:
        data_path: local path to ChronoQA JSON file (required until HF release)
        split: 'train' | 'validation' | 'test'
        max_samples: limit samples

    Returns same format as other loaders.

    Dataset format expected (JSON array):
    [
      {
        "id": "...",
        "question": "...",
        "answer": "..." or ["..."],
        "documents": ["passage1", "passage2", ...],
        "type": "temporal|causal|character",
        "split": "train|validation|test"
      },
      ...
    ]
    """
    if data_path is None:
        logger.warning(
            "[ChronoQA] No data_path provided. ChronoQA is not yet on HuggingFace. "
            "Download from the paper's repository when released and pass data_path=..."
        )
        return []

    import json, os
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"ChronoQA file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for i, ex in enumerate(raw):
        if ex.get("split", split) != split:
            continue
        if max_samples and len(samples) >= max_samples:
            break

        answer = ex.get("answer", ex.get("answers", ""))
        if isinstance(answer, list):
            gold_answers = answer
        else:
            gold_answers = [answer] if answer else []

        supporting_docs = ex.get("documents", ex.get("passages", ex.get("context", [])))
        if isinstance(supporting_docs, str):
            supporting_docs = [supporting_docs]

        samples.append({
            "id": str(ex.get("id", i)),
            "question": ex.get("question", ""),
            "gold_answers": gold_answers,
            "supporting_docs": supporting_docs,
            "type": ex.get("type", ""),
        })

    logger.info(f"[ChronoQA] Loaded {len(samples)} samples from {data_path} (split={split})")
    return samples
