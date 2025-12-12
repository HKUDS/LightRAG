#!/usr/bin/env python3
"""
Fast prompt A/B testing without full backend restart.

Bypasses the server entirely - queries DB directly, calls LLM directly,
runs RAGAS eval on results. ~30s per variant vs 5+ minutes full eval.

Usage:
    # Test all variants on 3 queries
    python test_prompt_variants.py

    # Test specific number of queries
    python test_prompt_variants.py --num-queries 5

    # Test specific query indices
    python test_prompt_variants.py --indices 0,5,9
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Ensure we can import lightrag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import AsyncOpenAI
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

# ============================================================================
# PROMPT VARIANTS - Edit these to test different prompts
# ============================================================================

BASELINE_PROMPT = """Answer based ONLY on this context:

{context_data}

---

GROUNDING RULES:
- Every claim must be directly supported by the context above
- Before writing each sentence, verify you can point to its source in context
- If information isn't in context, don't include it - even if it seems obvious
- Cover all relevant points from the context that address the question

DO NOT:
- Infer what "typically" happens in similar situations
- Add standard terminology not explicitly in the context
- Expand acronyms unless the expansion appears in context
- Fill in gaps with plausible-sounding details

RESPONSE FORMAT GUIDANCE:
Match your response structure to the question type:
- "What were the lessons learned / key takeaways / main themes..." → Enumerate distinct points (e.g., "Key lessons include: (1)..., (2)..., (3)...")
- "What challenges / issues / gaps..." → List specific challenges with brief explanations
- "What considerations / strategies / factors..." → Enumerate the specific items mentioned in context
- "How does X describe / explain / define..." → Provide a coherent summary of the description
- "What are the interdependencies / relationships..." → Describe the connections and their nature

ANSWER STRUCTURE:
- Lead with the direct answer to the question in your first sentence
- Use the question's key terms in your response (e.g., if asked about "lessons learned", say "lessons learned include...")
- For enumeration questions, provide numbered or bulleted items when context supports multiple points

Write naturally in flowing paragraphs. Do NOT include [1], [2] citations - they're added automatically.

Question: {user_prompt}

Answer (grounded only in context above):"""


# Variant 1: More direct but still grounded
VARIANT_1 = """Answer based ONLY on this context:

{context_data}

---

CRITICAL: Your answer will be evaluated on two dimensions:
1. FAITHFULNESS - Every claim must be directly from the context (no hallucination)
2. RELEVANCE - Your answer must directly address what the question asks

ANSWER RULES:
- First sentence: Directly answer the question using the question's key terms
- Only include facts explicitly stated in the context above
- If asked about specific tools/methods, name them from the context
- If the context doesn't cover something, say "The available information does not specify..."

FORMAT:
- "What were the X?" → "The key X were: (1)..., (2)..., (3)..."
- "How does Y describe Z?" → "According to Y, Z is described as..."
- "What challenges/issues?" → List the specific challenges mentioned

Question: {user_prompt}

Answer:"""


# Variant 2: Minimal, trusts the model more
VARIANT_2 = """Context:
{context_data}

---

Answer the question using ONLY information from the context above.
- Do not add anything not explicitly stated in the context
- If unsure, say "not specified in the available information"

Question: {user_prompt}

Answer:"""


# Variant 3: DSPy MIPRO-optimized (CoT + extraction focus)
VARIANT_MIPRO = """Context:
{context_data}

---

Given the context above and the question below, follow these steps:

1. **Extract** - Identify specific facts from the context relevant to the question
2. **Answer** - Produce a concise answer (2-4 key points) strictly based on extracted facts

RULES:
- Do NOT hallucinate or add information not present in the context
- If context lacks relevant information, state: "The available information does not address this"
- Use the question's key terms in your response
- For list questions, enumerate with (1), (2), (3)

Question: {user_prompt}

Answer:"""


# Variant 4: Hybrid - combines best of balanced + mipro insights
VARIANT_HYBRID = """Answer based ONLY on this context:

{context_data}

---

TASK: Answer the question using ONLY facts from the context above.

STEP 1 - Identify relevant facts:
Before answering, mentally note which specific facts from the context address the question.

STEP 2 - Structure your answer:
- Start with a direct answer using the question's key terms
- For "What were the X?" → List the X with brief explanations
- For "How does Y describe Z?" → Summarize Y's description of Z
- Use (1), (2), (3) for multiple points when appropriate

STEP 3 - Verify grounding:
- Every claim must come from the context
- If context doesn't cover something, say "The available information does not specify..."
- Don't add obvious-seeming details not explicitly stated

Question: {user_prompt}

Answer:"""


# Variant 5: Ultra-direct - focuses on answer structure only
VARIANT_DIRECT = """Context:
{context_data}

---

Answer the question below using ONLY the context above.

ANSWER FORMAT:
- First sentence: Direct answer using the question's key terms
- Then: Supporting details from context (use numbered points for lists)
- Do not include information not in the context

Question: {user_prompt}

Answer:"""


# Variant 6: Question-type aware
VARIANT_QTYPE = """Context:
{context_data}

---

Answer using ONLY the context above. Match your response to the question type:

IF "What were the lessons/challenges/considerations..." → Enumerate: "(1)..., (2)..., (3)..."
IF "How does X describe/explain..." → Summarize what X says about the topic
IF "What are the relationships/interdependencies..." → Describe the connections

RULES:
- Every fact must be from the context above
- Use the question's terminology in your answer
- If information is missing, acknowledge it

Question: {user_prompt}

Answer:"""


PROMPT_VARIANTS = {
    'baseline': BASELINE_PROMPT,
    'balanced': VARIANT_1,
    'minimal': VARIANT_2,
    'mipro': VARIANT_MIPRO,
    'hybrid': VARIANT_HYBRID,
    'direct': VARIANT_DIRECT,
    'qtype': VARIANT_QTYPE,
}


# ============================================================================
# Core Functions
# ============================================================================

@dataclass
class TestResult:
    variant: str
    query: str
    answer: str
    faithfulness: float
    relevance: float
    latency_ms: float


async def call_llm(prompt: str, client: AsyncOpenAI) -> tuple[str, float]:
    """Call LLM and return response + latency"""
    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=2000,
    )

    latency = (time.perf_counter() - start) * 1000
    return response.choices[0].message.content, latency


async def get_context_from_server(query: str, server_url: str = 'http://localhost:9621') -> str:
    """Get context from running LightRAG server (only_need_context mode)"""
    import httpx

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f'{server_url}/query',
            json={
                'query': query,
                'mode': 'mix',
                'only_need_context': True,
            }
        )
        data = response.json()
        return data.get('response', '')


def run_ragas_eval(question: str, answer: str, context: str, ground_truth: str) -> tuple[float, float]:
    """Run RAGAS evaluation and return (faithfulness, relevance)"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Create dataset
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [[context]],
        'ground_truth': [ground_truth],
    }
    dataset = Dataset.from_dict(data)

    # Create evaluation LLM
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    # Extract first element since we're evaluating single samples
    faith = result['faithfulness'][0] if isinstance(result['faithfulness'], list) else result['faithfulness']
    relevance = result['answer_relevancy'][0] if isinstance(result['answer_relevancy'], list) else result['answer_relevancy']

    return float(faith), float(relevance)


async def test_variant(
    variant_name: str,
    prompt_template: str,
    queries: list[dict],
    client: AsyncOpenAI,
    server_url: str,
) -> list[TestResult]:
    """Test a single prompt variant on all queries"""
    results = []

    for i, q in enumerate(queries):
        question = q['question']
        ground_truth = q['ground_truth']

        print(f'  [{variant_name}] Query {i+1}/{len(queries)}: {question[:50]}...')

        # Get context from server
        context = await get_context_from_server(question, server_url)

        if not context or context == 'No relevant context found for the query.':
            print(f'    ⚠️ No context retrieved, skipping')
            continue

        # Build prompt and call LLM
        prompt = prompt_template.format(
            context_data=context,
            user_prompt=question,
        )

        answer, latency = await call_llm(prompt, client)

        # Run RAGAS eval
        try:
            faith, relevance = run_ragas_eval(question, answer, context, ground_truth)
        except Exception as e:
            print(f'    ⚠️ RAGAS eval failed: {e}')
            faith, relevance = 0.0, 0.0

        results.append(TestResult(
            variant=variant_name,
            query=question[:60],
            answer=answer[:200],
            faithfulness=faith,
            relevance=relevance,
            latency_ms=latency,
        ))

        print(f'    Faith: {faith:.3f} | Relevance: {relevance:.3f} | {latency:.0f}ms')

    return results


def print_summary(all_results: dict[str, list[TestResult]]):
    """Print comparison summary"""
    print('\n' + '='*80)
    print('📊 PROMPT VARIANT COMPARISON')
    print('='*80)

    print(f"\n{'Variant':<12} | {'Faithfulness':>12} | {'Relevance':>12} | {'Avg Latency':>12}")
    print('-'*55)

    for variant, results in all_results.items():
        if not results:
            continue
        avg_faith = sum(r.faithfulness for r in results) / len(results)
        avg_rel = sum(r.relevance for r in results) / len(results)
        avg_lat = sum(r.latency_ms for r in results) / len(results)

        print(f'{variant:<12} | {avg_faith:>12.3f} | {avg_rel:>12.3f} | {avg_lat:>10.0f}ms')

    print('='*80)

    # Per-query breakdown
    print('\n📋 Per-Query Results:')
    queries = list(all_results.values())[0] if all_results else []
    for i in range(len(queries)):
        query = queries[i].query if i < len(queries) else 'N/A'
        print(f'\nQ{i+1}: {query}')
        for variant, results in all_results.items():
            if i < len(results):
                r = results[i]
                print(f'  {variant:<10}: Faith={r.faithfulness:.2f} Rel={r.relevance:.2f}')


async def main():
    parser = argparse.ArgumentParser(description='Fast prompt A/B testing')
    parser.add_argument('--num-queries', '-n', type=int, default=3, help='Number of queries to test')
    parser.add_argument('--indices', '-i', type=str, help='Specific query indices (comma-separated)')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:9621', help='LightRAG server URL')
    parser.add_argument('--variants', '-v', type=str, help='Variants to test (comma-separated)')
    args = parser.parse_args()

    # Load test dataset
    dataset_path = Path(__file__).parent / 'pharma_test_dataset.json'
    with open(dataset_path) as f:
        data = json.load(f)

    # Handle both formats: dict with 'test_cases' key or direct list
    if isinstance(data, dict) and 'test_cases' in data:
        dataset = data['test_cases']
    else:
        dataset = data

    # Select queries
    if args.indices:
        indices = [int(i) for i in args.indices.split(',')]
        queries = [dataset[i] for i in indices if i < len(dataset)]
    else:
        queries = dataset[:args.num_queries]

    print(f'🧪 Testing {len(queries)} queries')

    # Select variants
    if args.variants:
        variant_names = args.variants.split(',')
        variants = {k: v for k, v in PROMPT_VARIANTS.items() if k in variant_names}
    else:
        variants = PROMPT_VARIANTS

    print(f'📝 Testing variants: {list(variants.keys())}')

    # Create OpenAI client
    client = AsyncOpenAI(
        api_key=os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1'),
    )

    # Test each variant
    all_results = {}
    for variant_name, template in variants.items():
        print(f'\n🔄 Testing variant: {variant_name}')
        results = await test_variant(variant_name, template, queries, client, args.server)
        all_results[variant_name] = results

    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    asyncio.run(main())
