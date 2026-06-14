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

    # hotpotqa/hotpot_qa replaced the old hotpot_qa dataset name on HuggingFace
    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        # context: {'title': [str, ...], 'sentences': [[str, ...], ...]}
        titles         = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        supporting_docs = [
            f"{title}\n" + " ".join(sents)
            for title, sents in zip(titles, sentences_list)
        ]

        answer = ex["answer"]
        gold_answers = [answer] if answer else []

        samples.append({
            "id":             ex["id"],
            "question":       ex["question"],
            "gold_answers":   gold_answers,
            "supporting_docs": supporting_docs,
            "type":           ex.get("type", ""),
            "level":          ex.get("level", ""),
        })
    logger.info(f"[HotpotQA] Loaded {len(samples)} samples from {split}/{config}")
    return samples


def load_2wikimultihopqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load 2WikiMultiHopQA (Ho et al. 2020).

    HuggingFace dataset: xanhho/2WikiMultiHopQA (confirmed working)
    Fields: _id, type, question, context (JSON str), supporting_facts (JSON str),
            evidences (JSON str), answer
    context JSON format: [["title", ["sent1", "sent2", ...]], ...]
    """
    import json as _json

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    for hf_name in ["xanhho/2WikiMultiHopQA", "voidful/2WikiMultihopQA"]:
        try:
            ds = load_dataset(hf_name, split=split)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(
            "Could not load 2WikiMultiHopQA from HuggingFace. "
            "Try: pip install datasets  and check your internet connection."
        )

    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # context is a JSON string: [["title", ["sent1", "sent2"]], ...]
        supporting_docs: list[str] = []
        raw_ctx = ex.get("context", "[]")
        try:
            ctx_list = _json.loads(raw_ctx) if isinstance(raw_ctx, str) else raw_ctx
            for item in ctx_list:
                if isinstance(item, list) and len(item) == 2:
                    title, sents = item
                    text = " ".join(sents) if isinstance(sents, list) else str(sents)
                    supporting_docs.append(f"{title}\n{text}" if title else text)
                elif isinstance(item, dict):
                    title = item.get("title", "")
                    sents = item.get("sentences", item.get("text", ""))
                    text = " ".join(sents) if isinstance(sents, list) else str(sents)
                    supporting_docs.append(f"{title}\n{text}" if title else text)
        except Exception:
            if isinstance(raw_ctx, str) and raw_ctx:
                supporting_docs = [raw_ctx]

        answer = ex.get("answer", "")
        gold_answers = [answer] if answer else []

        samples.append({
            "id":              str(ex.get("_id", ex.get("id", i))),
            "question":        ex.get("question", ""),
            "gold_answers":    gold_answers,
            "supporting_docs": supporting_docs,
            "type":            ex.get("type", ""),
        })
    logger.info(f"[2WikiMultiHopQA] Loaded {len(samples)} samples from {split}")
    return samples


def load_musique(
    split: str = "validation",
    max_samples: Optional[int] = None,
    data_path: Optional[str] = None,
) -> list[dict]:
    """Load MuSiQue (Trivedi et al. 2022).

    MuSiQue is NOT available on HuggingFace Hub. Download from:
      https://github.com/StonyBrookNLP/musique/releases/download/v1.0/musique_v1.0.zip

    Args:
        split: 'train' | 'validation' | 'test' (maps to file suffix)
        max_samples: limit samples
        data_path: path to local JSONL file (e.g. musique_ans_v1.0_dev.jsonl)
                   If None, tries common file locations automatically.

    JSONL fields (musique_ans_v1.0_*.jsonl):
      id, question, answer, answer_aliases, answerable,
      paragraphs: [{idx, title, paragraph_text, is_supporting}]
    """
    import json as _json

    # ── Local JSONL file ──────────────────────────────────────────────────────
    split_suffix = {"validation": "dev", "train": "train", "test": "test"}.get(split, split)
    candidates = [
        data_path,
        f"musique_ans_v1.0_{split_suffix}.jsonl",
        f"data/musique_ans_v1.0_{split_suffix}.jsonl",
        f"musique/musique_ans_v1.0_{split_suffix}.jsonl",
        f"musique_full_v1.0_{split_suffix}.jsonl",
    ]
    local_file = next((p for p in candidates if p and os.path.exists(p)), None)

    if local_file:
        logger.info(f"[MuSiQue] Loading from local file: {local_file}")
        return _load_musique_jsonl(local_file, max_samples)

    # ── HuggingFace fallback (try known mirrors) ───────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    for hf_name in ["drt/musique", "Zenoverse/musique", "StonyBrookNLP/musique"]:
        try:
            ds = load_dataset(hf_name, split=split)
            logger.info(f"[MuSiQue] Loaded from HuggingFace: {hf_name}")
            return _parse_musique_hf(ds, max_samples)
        except Exception:
            continue

    raise RuntimeError(
        "MuSiQue not found on HuggingFace and no local file detected.\n"
        "Download from: https://github.com/StonyBrookNLP/musique/releases/download/v1.0/musique_v1.0.zip\n"
        "Then pass data_path='musique_ans_v1.0_dev.jsonl' or place the file in the current directory."
    )

def _parse_musique_example(ex: dict, idx: int) -> dict:
    """Parse one MuSiQue example (works for both HF and local JSONL)."""
    paragraphs = ex.get("paragraphs", [])
    supporting_docs = []
    for para in paragraphs:
        if isinstance(para, dict):
            title = para.get("title", "")
            text  = para.get("paragraph_text", para.get("text", ""))
            supporting_docs.append(f"{title}\n{text}" if title else text)

    answer         = ex.get("answer", "")
    answer_aliases = ex.get("answer_aliases") or []
    gold_answers   = ([answer] + answer_aliases) if answer else answer_aliases

    return {
        "id":              str(ex.get("id", idx)),
        "question":        ex.get("question", ""),
        "gold_answers":    gold_answers,
        "supporting_docs": supporting_docs,
        "answerable":      ex.get("answerable", True),
    }


def _parse_musique_hf(ds, max_samples: Optional[int]) -> list[dict]:
    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        samples.append(_parse_musique_example(ex, i))
    logger.info(f"[MuSiQue] Loaded {len(samples)} samples from HuggingFace")
    return samples


def _load_musique_jsonl(path: str, max_samples: Optional[int]) -> list[dict]:
    import json as _json
    samples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            ex = _json.loads(line)
            samples.append(_parse_musique_example(ex, i))
    logger.info(f"[MuSiQue] Loaded {len(samples)} samples from {path}")
    return samples


def load_chronoqa(
    data_path: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load ChronoQA — passage-grounded narrative QA (arXiv 2506.05939).

    Dataset: https://huggingface.co/datasets/zy113/ChronoQA
    - 1,028 QA pairs across 18 public-domain narratives
    - Single split: "train" (all 1028 rows; no train/val/test split)
    - Each row has ONE field "queries" (nested dict) containing all data
    - Each sample has gold evidence excerpts with byte offsets

    Actual schema (nested under "queries"):
      query           str   — the question
      ground_truth    str   — gold answer
      story_id        str   — narrative ID ("1"–"18")
      question_id     int   — index within that story
      category        str   — reasoning facet (8 types)
      passages        list  — [{excerpt, start_byte, end_byte, ...}]
      supporting_passages list[int] — indices into passages (gold evidence)
      primary_assessment  dict | None

    Args:
        data_path: optional local JSONL/JSON file path (overrides HuggingFace)
        split: HuggingFace split name — ChronoQA only has "train"
        max_samples: limit total samples loaded
    """
    # ── Local file override ────────────────────────────────────────────────────
    if data_path is not None:
        import json as _json
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"ChronoQA file not found: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".jsonl"):
                raw = [_json.loads(line) for line in f if line.strip()]
            else:
                raw = _json.load(f)
        return _parse_chronoqa_records(raw, max_samples=max_samples, source=data_path)

    # ── HuggingFace loader ────────────────────────────────────────────────────
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    try:
        ds = load_dataset("zy113/ChronoQA", split=split)
    except Exception as e:
        logger.error(f"[ChronoQA] Failed to load from HuggingFace: {e}")
        return []

    samples = []
    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        # All data is nested under the "queries" field
        ex = row.get("queries", row)
        samples.append(_parse_chronoqa_example(ex, i))

    logger.info(f"[ChronoQA] Loaded {len(samples)} samples from HuggingFace (split={split})")
    return samples


# Story ID → title mapping (from dataset description)
_CHRONOQA_STORY_TITLES = {
    "1":  "A Study in Scarlet",
    "2":  "The Hound of the Baskervilles",
    "3":  "Harry Potter and the Chamber of Secrets",
    "4":  "Harry Potter and the Sorcerer's Stone",
    "5":  "Les Misérables",
    "6":  "The Phantom of the Opera",
    "7":  "The Sign of the Four",
    "8":  "The Wonderful Wizard of Oz",
    "9":  "The Adventures of Sherlock Holmes",
    "10": "Lady Susan",
    "11": "Dangerous Connections",
    "12": "The Picture of Dorian Gray",
    "13": "The Diary of a Nobody",
    "14": "The Sorrows of Young Werther",
    "15": "The Mysterious Affair at Styles",
    "16": "Pride and Prejudice",
    "17": "The Secret Garden",
    "18": "Anne of Green Gables",
}


def _parse_chronoqa_example(ex: dict, fallback_id: int) -> dict:
    """Normalise a single ChronoQA queries dict.

    Fields inside "queries" (from actual HuggingFace inspection):
      query                str
      ground_truth         str
      story_id             str   ("1"–"18")
      question_id          int
      category             str   (reasoning facet)
      passages             list  [{excerpt, start_byte, end_byte,
                                   start_sentence, end_sentence,
                                   first_words, last_words,
                                   verification_status, verification_notes}]
      supporting_passages  list[int] | None  — indices of gold passages
      primary_assessment   dict | None
      verification_notes   list | None
    """
    story_id    = str(ex.get("story_id", fallback_id))
    question_id = ex.get("question_id", fallback_id)
    uid         = f"{story_id}_{question_id}"

    gold_answer  = ex.get("ground_truth", "")
    gold_answers = [gold_answer] if gold_answer else []

    category = ex.get("category", "")

    # All passage dicts; excerpt field has the actual text
    raw_passages: list = ex.get("passages") or []
    all_excerpts: list[str] = [
        p["excerpt"] for p in raw_passages
        if isinstance(p, dict) and p.get("excerpt")
    ]

    # supporting_passages: indices pointing to the gold evidence passages
    support_idxs: list[int] = ex.get("supporting_passages") or []
    gold_excerpts: list[str] = [
        all_excerpts[i] for i in support_idxs
        if isinstance(i, int) and i < len(all_excerpts)
    ] if support_idxs else all_excerpts  # fall back to all if not annotated

    story_title = _CHRONOQA_STORY_TITLES.get(story_id, story_id)

    return {
        "id":              uid,
        "story_id":        story_id,
        "story_title":     story_title,
        "question":        ex.get("query", ""),
        "gold_answers":    gold_answers,
        "supporting_docs": gold_excerpts,   # gold evidence excerpts for RAG corpus
        "all_excerpts":    all_excerpts,    # all passage excerpts in this QA pair
        "category":        category,
        "type":            category,
        "passages_raw":    raw_passages,    # full dicts with byte offsets
    }


def _parse_chronoqa_records(
    raw: list,
    max_samples: Optional[int],
    source: str,
) -> list[dict]:
    """Parse a local JSONL/JSON list into the standard ChronoQA format."""
    samples = []
    for i, row in enumerate(raw):
        if max_samples and len(samples) >= max_samples:
            break
        ex = row.get("queries", row)
        samples.append(_parse_chronoqa_example(ex, i))
    logger.info(f"[ChronoQA] Loaded {len(samples)} samples from {source}")
    return samples
