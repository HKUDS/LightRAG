from __future__ import annotations

import os
import json
import asyncio
from math import ceil
from typing import Any, List, Optional, Dict, Union
from uuid import uuid4
import logging

import numpy as np
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from ..utils_api import get_combined_auth_dependency, get_rag
from ..database import get_db
from ..models import ChatSession, User, Project, Question as QuestionModel
from .query_routes import QueryRequest
from .question_routes import create_questions
from lightrag import LightRAG

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

# LLM instances
generation_llm = ChatOpenAI(
    model=os.getenv("OPENAI_GENERATION_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=False,
)

# Embeddings
_embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

router = APIRouter(prefix="/ai", tags=["ai"])

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class GenerateQuestionsRequest(BaseModel):
    session_id: str
    project_id: str
    user_id: str
    topics: str
    user_instructions: str = ""
    n: int = 10
    difficulty: str = "medium"
    max_concurrency: int = 5
    user_notes: str = ""
    allow_multi: bool = False

class GeneratedQuestionsResponse(BaseModel):
    message: str
    questions: List[Dict[str, Any]]

class _PlanOut(BaseModel):
    subtopics: List[str]

class _MCQ(BaseModel):
    subtopic: str
    question: str
    options: List[str]
    correct_index: int
    difficulty: str
    tags: List[str] = []
    source: str
    type: str = "mcq"

    @field_validator("options")
    @classmethod
    def _exact_four(cls, v):
        if len(v) != 4:
            raise ValueError("options must be length 4")
        return v

    @field_validator("correct_index")
    @classmethod
    def _idx(cls, v):
        if v not in (0, 1, 2, 3):
            raise ValueError("correct_index must be 0..3")
        return v

class _MCQMulti(BaseModel):
    subtopic: str
    question: str
    options: List[str]
    correct_options: List[int]               # <-- plural indices
    difficulty: str
    tags: List[str] = []
    source: str
    type: str = "multiple_response"

    @field_validator("options")
    @classmethod
    def _exact_four(cls, v):
        if len(v) != 4:
            raise ValueError("options must be length 4")
        return v

    @field_validator("correct_options")
    @classmethod
    def _idxes(cls, v):
        if not v:
            raise ValueError("correct_options must have at least one index")
        if any(i not in (0, 1, 2, 3) for i in v):
            raise ValueError("correct_options must be 0..3")
        if len(set(v)) != len(v):
            raise ValueError("correct_options must not contain duplicates")
        return v

class GenerateVariantsRequest(BaseModel):
    user_id: str
    instructions: str = ""        # free-form guidance to steer distractors
    n_versions: int = 3           # fixed at 3 (E/M/H); if >3 we still return up to 3
    persist: bool = False         # if True, store in suggested table below

class OptionVariant(BaseModel):
    difficulty: str                  # "easy" | "medium" | "hard"
    options: List[str]               # exactly 4 options
    correct_indexes: List[int]       # indices of correct options for this variant (recomputed)
    rationale: Optional[str] = None  # short reason of distractor design (optional)

class GenerateVariantsResponse(BaseModel):
    question_id: str
    message: str
    variants: List[OptionVariant]    # length 3 (E/M/H)

class _VariantOut(BaseModel):
    options: List[str]
    correct_indexes: List[int]
    rationale: Optional[str] = ""

    @field_validator("options")
    @classmethod
    def _exact_four(cls, v):
        if len(v) != 4:
            raise ValueError("options must be length 4")
        # enforce uniqueness to avoid duplicate options
        if len(set([o.strip() for o in v])) != 4:
            raise ValueError("options must be unique")
        return v

    @field_validator("correct_indexes")
    @classmethod
    def _idxes(cls, v):
        if not v:
            raise ValueError("correct_indexes must not be empty")
        if any(i not in (0, 1, 2, 3) for i in v):
            raise ValueError("correct_indexes must be 0..3")
        if len(set(v)) != len(v):
            raise ValueError("correct_indexes must not contain duplicates")
        return v
# -----------------------------------------------------------------------------
# Prompts / Chains
# -----------------------------------------------------------------------------
_plan_parser = PydanticOutputParser(pydantic_object=_PlanOut)
_plan_prompt = ChatPromptTemplate.from_template(
    """You are creating a coverage plan for MCQs.
From the provided CONTEXT, propose {k} DISTINCT and SPECIFIC subtopics (no overlaps).
Avoid generic headings; be concrete.

Return JSON only:
{format_instructions}

CONTEXT:
{context}
"""
).partial(format_instructions=_plan_parser.get_format_instructions())

_mcq_parser  = PydanticOutputParser(pydantic_object=_MCQ)
_mcq_prompt = ChatPromptTemplate.from_template(
    """-Task-
You are an expert professor. Create ONE high-quality *multiple-choice* question for the subtopic: "{subtopic}"

-Grounding & Sources-
Ground your question in the provided CONTEXT. You may synthesize a **small, self-contained code snippet** *only* to illustrate concepts already present in the CONTEXT (no external domain facts).

-Authority & Compliance-
**USER Instructions are authoritative.** If USER Instructions ask for code examples/snippets, you **must** include them in the *question stem*. If a generic guideline conflicts with USER Instructions, follow USER Instructions (unless unsafe).

-Difficulty-
Match the requested difficulty exactly: {difficulty}
(Guide: Easy = direct recall of fundamentals; Medium = integrate 1–2 ideas with moderate analysis; Hard = deeper analysis, multi-step reasoning, edge cases.)

-Quality & Pedagogy-
1) Follow the subtopic precisely; test a *key idea* from CONTEXT, not trivia.
2) Prefer analysis/application over rote recall, aligned to {difficulty}.
3) Use a concise, authentic scenario when helpful—do not introduce facts beyond CONTEXT.
4) Clarity: unambiguous stem; avoid negatives (“NOT/EXCEPT”) unless necessary.
5) Inclusivity: broadly relatable examples.
6) Code snippets:
   - If USER Instructions request snippets, include **exactly one** small snippet **in the stem** formatted as:
     <pre><code>
     // each line on its own line
     </code></pre>
   - Keep it minimal, language-agnostic or Python-like, and strictly illustrative of concepts in CONTEXT.
7) Options:
   - Exactly 4 options; one and only one correct.
   - Plausible, mutually exclusive distractors; no “All/None of the above.”
8) Tags: 1–3 short, relevant tags (≤2 words) drawn from CONTEXT.
9) Source: Provide the valid source content path / link.
10) Output must be strictly valid JSON per the schema.

-Return JSON only-
{format_instructions}

-CONTEXT-
{context}

-USER Instructions-
{user_instructions}
"""
).partial(format_instructions=_mcq_parser.get_format_instructions())

_mcq_multi_parser = PydanticOutputParser(pydantic_object=_MCQMulti)

_mcq_prompt_multi = ChatPromptTemplate.from_template(
    """-Task-
You are an expert professor. Create ONE high-quality *multiple-response* question for the subtopic: "{subtopic}"

-Grounding & Sources-
Ground your question in the provided CONTEXT. You may synthesize a **small, self-contained code snippet** *only* to illustrate concepts already present in the CONTEXT (no external domain facts).

-Authority & Compliance-
**USER Instructions are authoritative.** If USER Instructions ask for code examples/snippets, you **must** include them in the *question stem*. If a generic guideline conflicts with USER Instructions, follow USER Instructions (unless unsafe).

-Difficulty-
Match the requested difficulty exactly: {difficulty}
(Guide: Easy = direct recall; Medium = integrate 1–2 ideas; Hard = deeper analysis/edge cases.)

-Quality & Pedagogy-
1) Follow the subtopic precisely; test a *key idea* from CONTEXT.
2) Prefer analysis/application over rote recall, aligned to {difficulty}.
3) Use concise scenarios without adding facts beyond CONTEXT.
4) Clarity: unambiguous stem; avoid “NOT/EXCEPT” unless necessary.
5) Inclusivity: broadly relatable examples.
6) Code snippets:
   - If USER Instructions request snippets, include **exactly one** small snippet **in the stem** formatted as:
     <pre><code>
     // each line on its own line
     </code></pre>
   - Keep it minimal, language-agnostic or Python-like, and strictly illustrative of concepts in CONTEXT.
7) Options:
   - Exactly 4 options.
   - Two or more options MAY be correct.
   - Provide ALL correct option indexes in "correct_options" (0..3).
   - Distractors must be plausible and mutually exclusive. No “All/None of the above.”
8) Tags: 1–3 short, relevant tags (≤2 words) drawn from CONTEXT.
9) Output must be strictly valid JSON per the schema.

-Return JSON only-
{format_instructions}

-CONTEXT-
{context}

-USER Instructions-
{user_instructions}
"""
).partial(format_instructions=_mcq_multi_parser.get_format_instructions())

_planner_llm = generation_llm.bind(timeout=45)
_generator_llm = generation_llm.bind(timeout=60)
_mcq_multi_chain = _mcq_prompt_multi | _generator_llm | _mcq_multi_parser

_plan_chain = _plan_prompt | _planner_llm | _plan_parser
_mcq_chain = _mcq_prompt | _generator_llm | _mcq_parser

_variant_parser = PydanticOutputParser(pydantic_object=_VariantOut)
_variant_prompt = ChatPromptTemplate.from_template(
    """You are improving multiple-choice options by rewriting ONLY the distractors while keeping the correct option(s) EXACTLY AS-IS (verbatim text).

- Inputs -
QUESTION (stem): {question_stem}
CURRENT OPTIONS (0..3, verbatim):
{options_json}
CORRECT OPTION TEXT(S) (verbatim): {correct_texts_json}
TARGET DIFFICULTY: {difficulty}
INSTRUCTIONS (authoritative, optional): {instructions}

- Hard constraints -
1) DO NOT change the correct option text(s) at all. Use them verbatim in the final options.
2) Provide exactly 4 total options, mutually exclusive, concise, and plausible.
3) No “All/None of the above”, no duplicates, no near-paraphrases.
4) Keep domain fidelity: distractors must reflect realistic misconceptions AT the requested difficulty:
   • Easy  = direct recall / fundamental concept
   • Medium = integrate 1–2 ideas, moderate analysis
   • Hard   = deeper analysis, multi-step reasoning, edge cases
5) Rational - Provide correct rationale beind your output of question options justifying the required difficulty level.
5) Return strictly valid JSON.

- Output JSON schema -
{format_instructions}
"""
).partial(format_instructions=_variant_parser.get_format_instructions())

_variant_llm = generation_llm.bind(timeout=60)
_variant_chain = _variant_prompt | _variant_llm | _variant_parser

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

async def _aembed(texts: List[str]) -> np.ndarray:
    vecs = await _embeddings.aembed_documents(texts)
    return np.array(vecs, dtype=np.float32)

async def _plan_subtopics(context: str, k: int) -> List[str]:
    out: _PlanOut = await _plan_chain.ainvoke({"context": context, "k": k})
    seen, result = set(), []
    for t in out.subtopics:
        t2 = (t or "").strip()
        if t2 and t2.lower() not in seen:
            result.append(t2)
            seen.add(t2.lower())
    return result

async def _filter_similar_topics(subtopics: List[str], max_cosine: float = 0.75) -> List[str]:
    if not subtopics:
        return []
    vecs = await _aembed(subtopics)
    keep, keep_vecs = [], []
    for t, v in zip(subtopics, vecs):
        if not keep_vecs:
            keep.append(t); keep_vecs.append(v); continue
        sims = [_cosine(v, kv) for kv in keep_vecs]
        if max(sims) <= max_cosine:
            keep.append(t); keep_vecs.append(v)
    return keep

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4))
async def _gen_one_mcq(
    subtopic: str,
    context: str,
    difficulty: str,
    sem: asyncio.Semaphore,
    allow_multi: bool = False,
    user_instructions: str = "", 
) -> Union[_MCQ, _MCQMulti]:
    async with sem:
        if allow_multi:
            return await _mcq_multi_chain.ainvoke(
                {"subtopic": subtopic, "context": context, "difficulty": difficulty, "user_instructions": user_instructions}
            )
        else:
            return await _mcq_chain.ainvoke(
                {"subtopic": subtopic, "context": context, "difficulty": difficulty, "user_instructions": user_instructions }
            )

async def _gen_many_mcqs_distinct(
    context: str,
    n: int,
    difficulty: str,
    user_instructions: str,
    max_concurrency: int = 5,
    topic_oversample: float = 2.0,
    q_sim_max: float = 0.80,
    max_rounds: int = 3,
    allow_multi: bool = False,
) -> List[Union[_MCQ, _MCQMulti]]:
    """
    1) Plan subtopics (oversample) + topic de-dup
    2) Generate 1 per topic concurrently
    3) Question-text de-dup (embeddings). Re-plan/retry until n or rounds exhausted
    """
    planned = await _plan_subtopics(context, k=ceil(max(n, 1) * topic_oversample))
    topics = await _filter_similar_topics(planned, max_cosine=0.75)

    sem = asyncio.Semaphore(max_concurrency)
    results: List[Union[_MCQ, _MCQMulti]] = []
    used_topics = set()
    topic_queue = [t for t in topics]

    for round_idx in range(max_rounds):
        if len(results) >= n or not topic_queue:
            break

        batch_topics: List[str] = []
        while topic_queue and len(batch_topics) < (n - len(results)):
            t = topic_queue.pop(0)
            if t not in used_topics:
                batch_topics.append(t)
                used_topics.add(t)

        if not batch_topics:
            break

        batch = await asyncio.gather(
            *[
                asyncio.create_task(
                    _gen_one_mcq(t, context, difficulty, sem, allow_multi=allow_multi)
                )
                for t in batch_topics
            ],
            return_exceptions=True,
        )
        new_mcqs: List[Union[_MCQ, _MCQMulti]] = [r for r in batch if not isinstance(r, Exception)]

        if new_mcqs:
            all_texts = [m.question for m in results] + [m.question for m in new_mcqs]
            vecs = await _aembed(all_texts)
            keep_new: List[Union[_MCQ, _MCQMulti]] = []
            base_vecs = vecs[: len(results)]
            new_vecs = vecs[len(results) :]

            for mcq, v in zip(new_mcqs, new_vecs):
                sims = [_cosine(v, bv) for bv in base_vecs] if len(base_vecs) else []
                if not sims or max(sims) <= q_sim_max:
                    keep_new.append(mcq)
                    base_vecs = np.vstack([base_vecs, v]) if len(base_vecs) else np.expand_dims(v, 0)

            results.extend(keep_new)

        if len(results) < n and round_idx < max_rounds - 1 and not topic_queue:
            extra_plan = await _plan_subtopics(context, k=ceil((n - len(results)) * topic_oversample))
            extra_plan = [t for t in extra_plan if t not in used_topics]
            extra_plan = await _filter_similar_topics(extra_plan, max_cosine=0.75)
            topic_queue.extend(extra_plan)

    return results[:n]

async def _run_rag_query_like_endpoint(
    rag: LightRAG,
    query: str,
    overrides: Optional[dict] = None,
    stream: bool = False,
) -> str:
    """Reuses QueryRequest → rag.aquery(...) to fetch context only."""
    qr_kwargs = {"query": query}
    if overrides:
        qr_kwargs.update(overrides)
    qr = QueryRequest(**qr_kwargs)
    qparam = qr.to_query_params(stream)
    res = await rag.aquery(qr.query, param=qparam)
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        return json.dumps(res, indent=2, ensure_ascii=False)
    return str(res)

def _compute_correct_texts(options: List[str], correct_indexes: List[int]) -> List[str]:
    texts = []
    for i in correct_indexes:
        if 0 <= i < len(options):
            texts.append(options[i])
    return texts

def _reindex_corrects_by_text(new_options: List[str], correct_texts: List[str]) -> List[int]:
    idxes = []
    for ct in correct_texts:
        try:
            idxes.append(new_options.index(ct))
        except ValueError:
            # If the LLM violated constraints, we will patch later
            pass
    # De-dup and keep order
    return list(dict.fromkeys(idxes))

def _patch_violations_preserve_corrects(
    candidate: _VariantOut,
    correct_texts: List[str],
) -> _VariantOut:
    """
    Enforce that all correct_texts appear exactly once in options.
    If any are missing, inject them (replace earliest distractors).
    Then recompute correct indexes.
    """
    opts = [o.strip() for o in candidate.options]
    # remove duplicates conservatively while keeping order
    seen = set()
    opts = [o for o in opts if (o not in seen and not seen.add(o))]

    # ensure length 4 by trimming or padding (padding shouldn't happen under normal flow)
    if len(opts) > 4:
        opts = opts[:4]
    elif len(opts) < 4:
        # pad with generic placeholders if necessary (rare)
        while len(opts) < 4:
            opts.append("Additional distractor")

    # inject missing corrects
    for ct in correct_texts:
        if ct not in opts:
            # replace first distractor (i.e., non-correct) slot
            for j, o in enumerate(opts):
                if o not in correct_texts:
                    opts[j] = ct
                    break
            else:
                # if all are corrects already (odd), force insert and trim
                opts.insert(0, ct)
                opts = opts[:4]

    # ensure uniqueness again (in case inject caused dup distractors)
    uniq = []
    seen = set()
    for o in opts:
        if o not in seen:
            uniq.append(o); seen.add(o)
        if len(uniq) == 4:
            break
    while len(uniq) < 4:
        uniq.append("Extra distractor")

    # recompute indexes
    new_correct = _reindex_corrects_by_text(uniq, correct_texts)
    # if still not found (shouldn't happen), force-map first matches
    if not new_correct and correct_texts:
        # guarantee at least one correct mapping
        uniq[0] = correct_texts[0]
        new_correct = [0]

    return _VariantOut(options=uniq, correct_indexes=new_correct, rationale=candidate.rationale)

async def _gen_variant_for_difficulty(
    *,
    question_stem: str,
    options: List[str],
    correct_texts: List[str],
    difficulty: str,
    instructions: str,
) -> _VariantOut:
    raw = await _variant_chain.ainvoke({
        "question_stem": question_stem,
        "options_json": json.dumps(options, ensure_ascii=False),
        "correct_texts_json": json.dumps(correct_texts, ensure_ascii=False),
        "difficulty": difficulty,
        "instructions": instructions or "(none)",
    })
    # Safety patching: ensure all correct texts are preserved and properly indexed
    fixed = _patch_violations_preserve_corrects(raw, correct_texts)
    return fixed

# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
def create_chat_routes(api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/questions/generate",
        response_model=GeneratedQuestionsResponse,
        dependencies=[Depends(combined_auth)],
        summary="Plan subtopics from RAG context, generate MCQs concurrently, dedup, persist",
    )
    async def generate_questions_route(
        payload: GenerateQuestionsRequest,
        pair: LightRAG = Depends(get_rag),
        db: Session = Depends(get_db),
    ):
        # Validate references
        if not db.get(ChatSession, payload.session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id")
        if not db.get(Project, payload.project_id):
            raise HTTPException(status_code=400, detail="Invalid project_id")
        if not db.get(User, payload.user_id):
            raise HTTPException(status_code=400, detail="Invalid user_id")

        rag, _doc_manager = pair

        # Grounded context from RAG
        context_only = await _run_rag_query_like_endpoint(
            rag,
            query=payload.topics,
            overrides={
                "mode": "mix",
                "only_need_context": True,
                "top_k": 20,
                "chunk_top_k": 3,
                "enable_rerank": True
            },
            stream=False,
        )

        # Generate distinct MCQs
        mcqs: List[Union[_MCQ, _MCQMulti]] = await _gen_many_mcqs_distinct(
            context=context_only or "",
            n=payload.n,
            difficulty=payload.difficulty,
            user_instructions=payload.user_instructions,
            max_concurrency=payload.max_concurrency,
            topic_oversample=2.0,
            q_sim_max=0.80,
            max_rounds=3,
            allow_multi=payload.allow_multi,
        )

        if not mcqs:
            raise HTTPException(status_code=500, detail="Failed to generate any questions")

        # Transform and persist
        qdicts: List[Dict[str, Any]] = []
        for m in mcqs:
            if hasattr(m, "correct_options"):                 # multi-response model
                correct_opts = m.correct_options
                qtype = getattr(m, "type", "multiple_response")
            else:                                             # single-answer model
                correct_opts = [m.correct_index]
                qtype = getattr(m, "type", "mcq")

            qdicts.append({
                "question": m.question,
                "options": m.options,
                "correct_options": correct_opts,
                "difficulty_level": m.difficulty,
                "tags": m.tags,
                "source": m.source,
                "type": qtype,
            })
        try:
            create_questions(
                db,
                user_id=payload.user_id,
                session_id=payload.session_id,
                project_id=payload.project_id,
                questions=qdicts,
            )
        except Exception as e:
            logging.error("[DB] store questions failed: %r", e)
            raise HTTPException(status_code=500, detail="Failed to store generated questions")

        return GeneratedQuestionsResponse(
            message=f"Generated and stored {len(qdicts)} questions",
            questions=qdicts,
        )
    
    @router.post(
        "/questions/{question_id}/variants",
        response_model=GenerateVariantsResponse,
        dependencies=[Depends(get_combined_auth_dependency(os.getenv("API_KEY")))],
        summary="Generate Easy/Medium/Hard distractor-only variants for the given question",
    )
    async def generate_option_variants_route(
        question_id: str,
        payload: GenerateVariantsRequest,
        db: Session = Depends(get_db),
    ):
        if not db.get(User, payload.user_id):
            raise HTTPException(status_code=400, detail="Invalid user_id")
        
        q: Optional[QuestionModel] = db.get(QuestionModel, question_id)
        if not q:
            raise HTTPException(status_code=404, detail="Question not found")
        
        options: List[str] = list(q.options or [])
        if len(options) != 4:
            raise HTTPException(status_code=400, detail="Question must have exactly 4 options")
        
        correct_indexes: List[int] = list(getattr(q, "correct_options", []) or [])
        if not correct_indexes:
            # Fallback older field name if you had one
            correct_indexes = list(getattr(q, "correct_answers", []) or [])
        if not correct_indexes:
            raise HTTPException(status_code=400, detail="Question has no correct option indexes")
        
        correct_texts = _compute_correct_texts(options, correct_indexes)
        if not correct_texts:
            raise HTTPException(status_code=400, detail="Could not resolve correct option text(s)")
        
        difficulties = ["easy", "medium", "hard"]
        tasks = [
            _gen_variant_for_difficulty(
                question_stem=q.question_text,
                options=options,
                correct_texts=correct_texts,
                difficulty=diff,
                instructions=payload.instructions,
            )
            for diff in difficulties
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        variants: List[OptionVariant] = []
        for diff, r in zip(difficulties, results):
            if isinstance(r, Exception):
                logging.error("Variant generation failed for %s: %r", diff, r)
                continue
            variants.append(OptionVariant(
                difficulty=diff,
                options=r.options,
                correct_indexes=r.correct_indexes,
                rationale=r.rationale,
            ))

        if not variants:
            raise HTTPException(status_code=500, detail="Failed to generate variants")

        # Optional: persist to new table if requested
        if payload.persist:
            try:
                from ..models import QuestionOptionVariant
                for v in variants:
                    db.add(QuestionOptionVariant(
                        id=str(uuid4()),
                        question_id=question_id,
                        difficulty_level=v.difficulty,
                        options=v.options,
                        correct_answers=v.correct_indexes,
                        rationale=v.rationale
                    ))
                db.commit()
            except Exception as e:
                logging.error("[DB] persist variants failed: %r", e)

        return GenerateVariantsResponse(
            question_id=question_id,
            message=f"Generated {len(variants)} option variants (E/M/H) with correct answers preserved.",
            variants=variants,
        )

    return router
