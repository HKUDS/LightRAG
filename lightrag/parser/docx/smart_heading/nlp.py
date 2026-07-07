"""spaCy access for smart heading discovery (mandatory when enabled).

Strictly lazy: nothing here imports spaCy until a ``smart_heading=true``
document actually needs an NLP judgment, so deployments that never enable
the parameter carry zero dependency and zero resident memory. When enabled
and the runtime or a pinned model is missing, loading HARD-FAILS with
install guidance — there is no rule-only degradation path (a silent
degradation would let the same file parse differently across environments,
breaking the I4 determinism promise).

Thread-safety: model loading is locked, and inference takes a process-wide
lock too — spaCy pipelines are not thread-safe and extract() runs on a
worker-thread pool. Judgments are per-paragraph and short, so the serialized
inference cost is negligible next to document parsing.
"""

from __future__ import annotations

import importlib.util
import threading
from typing import Any

_MODELS = {"zh": "zh_core_web_sm", "en": "en_core_web_sm"}

_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_pipelines: dict[str, Any] = {}

#: NER labels that veto a leading number's numbering identity (§2.2.5).
HOMOPHONE_ENTITY_LABELS = frozenset({"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"})

_INSTALL_HINT = (
    "smart_heading requires spaCy and its pinned language models. Install "
    "with: pip install lightrag-hku[api] && lightrag-download-cache "
    "--spacy --spacy-install (offline: see requirements-offline-smart-heading.txt)"
)


class SmartHeadingNLPError(RuntimeError):
    """spaCy runtime/model unavailable while smart_heading is enabled."""


def missing_spacy_models() -> list[str]:
    """Names of the pinned models that are not installed.

    Lightweight probe (package-metadata lookups only — nothing is imported
    into memory and no model is loaded), safe to call at server startup.
    """
    if importlib.util.find_spec("spacy") is None:
        return sorted(_MODELS.values())
    import spacy.util

    return sorted(m for m in _MODELS.values() if not spacy.util.is_package(m))


def ensure_spacy_models_installed(context: str) -> None:
    """Raise :class:`SmartHeadingNLPError` if any pinned model is missing.

    Used for startup fail-fast when configuration shows the deployment will
    use smart_heading; the parse-time hard error in :func:`_get_pipeline`
    remains the backstop for per-file enablement.
    """
    missing = missing_spacy_models()
    if missing:
        raise SmartHeadingNLPError(
            f"{context}, but spaCy model(s) {', '.join(missing)} are not "
            "installed. " + _INSTALL_HINT
        )


def _get_pipeline(lang: str):
    pipeline = _pipelines.get(lang)
    if pipeline is not None:
        return pipeline
    with _load_lock:
        pipeline = _pipelines.get(lang)
        if pipeline is not None:
            return pipeline
        try:
            import spacy
        except ImportError as exc:
            raise SmartHeadingNLPError(
                f"spaCy is not installed. {_INSTALL_HINT}"
            ) from exc
        model_name = _MODELS[lang]
        try:
            pipeline = spacy.load(model_name)
        except OSError as exc:
            raise SmartHeadingNLPError(
                f"spaCy model {model_name!r} is not installed. {_INSTALL_HINT}"
            ) from exc
        _pipelines[lang] = pipeline
        return pipeline


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def route_language(text: str) -> str:
    """Route to the zh or en pipeline by CJK character share."""
    if not text:
        return "en"
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return "zh" if cjk * 2 >= len(text.replace(" ", "") or " ") else "en"


def analyze(text: str):
    """Run the routed pipeline on ``text`` under the inference lock."""
    pipeline = _get_pipeline(route_language(text))
    with _infer_lock:
        return pipeline(text)


def sentence_count(text: str) -> int:
    """Number of sentences spaCy sees in ``text``."""
    doc = analyze(text)
    return sum(1 for _ in doc.sents)


def leading_entity_label(text: str) -> str | None:
    """Label of an entity anchored at the start of ``text`` (or None).

    Used for numbering-homophone vetoes: a paragraph opening with a
    DATE/MONEY/PERCENT/QUANTITY entity ("2026年3月…", "$100 …") did not
    open with a heading number.
    """
    stripped = text.lstrip()
    offset = len(text) - len(stripped)
    doc = analyze(text)
    for ent in doc.ents:
        if ent.start_char <= offset:
            return ent.label_
        if ent.start_char > offset:
            break
    return None


def token_following_leading_number(text: str) -> str | None:
    """The token right after a leading number ("3.14 版" → "版")."""
    doc = analyze(text)
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return None
    if tokens[0].like_num or tokens[0].text[:1].isdigit():
        return tokens[1].text if len(tokens) > 1 else None
    return None


def ends_with_sentence_period(text: str) -> bool:
    """Whether a trailing English period closes a sentence (vs abbreviation).

    Appends a phantom continuation and asks spaCy to re-segment: when the
    original trailing dot ends a sentence, the phantom word starts a new
    one; an abbreviation dot ("Fig." / "et al.") keeps it inside the same
    sentence.
    """
    stripped = text.rstrip()
    if not stripped.endswith("."):
        return False
    doc = analyze(stripped + " Next")
    for sent in doc.sents:
        if sent.end_char >= len(stripped):
            return sent.start_char >= len(stripped)
    return False
