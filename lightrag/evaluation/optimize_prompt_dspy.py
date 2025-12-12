#!/usr/bin/env python3
"""
DSPy-based Prompt Optimization for LightRAG.

This script automatically optimizes RAG response prompts using:
1. DSPy's MIPROv2 optimizer for instruction tuning
2. RAGAS metrics (Faithfulness + Answer Relevancy) as objectives
3. Your pharma test dataset for training/validation

The optimized prompt can then be copied to lightrag/prompt.py.

Usage:
    # Quick optimization (3 queries, light mode)
    python optimize_prompt_dspy.py --mode light --num-queries 3

    # Full optimization (all queries, medium mode)
    python optimize_prompt_dspy.py --mode medium

    # Export optimized prompt
    python optimize_prompt_dspy.py --export-prompt optimized_prompt.txt
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dspy
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Ensure we can import lightrag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# DSPy Signatures
# ============================================================================

class RAGResponse(dspy.Signature):
    """Generate a grounded answer based on retrieved context.

    The answer must be faithful to the context (no hallucination) and
    directly relevant to the question asked.
    """
    context: str = dspy.InputField(desc="Retrieved knowledge graph context with entities, relations, and source excerpts")
    question: str = dspy.InputField(desc="User's question to answer")
    answer: str = dspy.OutputField(desc="Grounded answer based only on the context provided")


class RAGResponseWithReasoning(dspy.Signature):
    """Generate a grounded answer with explicit reasoning about context usage.

    First identify relevant facts from context, then synthesize the answer.
    """
    context: str = dspy.InputField(desc="Retrieved knowledge graph context")
    question: str = dspy.InputField(desc="User's question")
    relevant_facts: str = dspy.OutputField(desc="Key facts from context relevant to the question")
    answer: str = dspy.OutputField(desc="Answer synthesized from the relevant facts only")


# ============================================================================
# DSPy Modules
# ============================================================================

class SimpleRAG(dspy.Module):
    """Simple RAG response module."""

    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(RAGResponse)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.respond(context=context, question=question)


class ChainOfThoughtRAG(dspy.Module):
    """RAG with chain-of-thought reasoning."""

    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(RAGResponse)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.respond(context=context, question=question)


class ReasoningRAG(dspy.Module):
    """RAG that first extracts relevant facts then answers."""

    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(RAGResponseWithReasoning)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.respond(context=context, question=question)


# ============================================================================
# RAGAS-based Metric
# ============================================================================

def ragas_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Evaluate prediction using RAGAS metrics.

    Returns combined score of faithfulness and answer relevancy.
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    question = example.question
    context = example.context
    ground_truth = getattr(example, 'ground_truth', '')
    answer = pred.answer

    # Skip if answer is empty
    if not answer or not answer.strip():
        return 0.0

    # Create RAGAS dataset
    data = {
        'question': [question],
        'answer': [answer],
        'contexts': [[context]],
        'ground_truth': [ground_truth] if ground_truth else [answer],
    }
    dataset = Dataset.from_dict(data)

    # Run evaluation
    try:
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )

        # Extract scores (they come as lists for single samples)
        faith = result['faithfulness']
        relevance = result['answer_relevancy']

        if isinstance(faith, list):
            faith = faith[0]
        if isinstance(relevance, list):
            relevance = relevance[0]

        # Combined score (weight faithfulness slightly higher)
        score = 0.6 * float(faith) + 0.4 * float(relevance)

        if trace is not None:
            print(f"  RAGAS: faith={faith:.3f} rel={relevance:.3f} -> {score:.3f}")

        return score

    except Exception as e:
        print(f"  RAGAS eval failed: {e}")
        return 0.0


def fast_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """
    Fast heuristic metric for initial bootstrapping.

    Checks basic quality signals without calling RAGAS.
    """
    answer = pred.answer if hasattr(pred, 'answer') else ''
    context = example.context
    question = example.question

    if not answer or not answer.strip():
        return 0.0

    score = 0.0

    # Length check (not too short, not too long)
    words = len(answer.split())
    if 20 <= words <= 500:
        score += 0.2
    elif 10 <= words <= 1000:
        score += 0.1

    # Contains key terms from question
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words) / max(len(q_words), 1)
    score += 0.3 * min(overlap * 2, 1.0)

    # References context (rough check)
    context_words = set(context.lower().split()[:200])  # First 200 words
    context_overlap = len(context_words & a_words) / max(len(a_words), 1)
    score += 0.5 * min(context_overlap * 3, 1.0)

    return min(score, 1.0)


# ============================================================================
# Data Loading
# ============================================================================

async def get_context_from_server(query: str, server_url: str = 'http://localhost:9621') -> str:
    """Get context from running LightRAG server."""
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


def load_dataset(dataset_path: Path, num_queries: int | None = None) -> list[dict]:
    """Load test dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and 'test_cases' in data:
        dataset = data['test_cases']
    else:
        dataset = data

    if num_queries:
        dataset = dataset[:num_queries]

    return dataset


async def prepare_dspy_examples(
    dataset: list[dict],
    server_url: str = 'http://localhost:9621',
) -> list[dspy.Example]:
    """Convert dataset to DSPy examples with context."""
    examples = []

    print(f"Fetching context for {len(dataset)} queries...")

    for i, item in enumerate(dataset):
        question = item['question']
        ground_truth = item.get('ground_truth', '')

        print(f"  [{i+1}/{len(dataset)}] {question[:50]}...")

        context = await get_context_from_server(question, server_url)

        if not context or context == 'No relevant context found for the query.':
            print(f"    Skipping - no context")
            continue

        example = dspy.Example(
            context=context,
            question=question,
            ground_truth=ground_truth,
        ).with_inputs('context', 'question')

        examples.append(example)

    print(f"Prepared {len(examples)} examples")
    return examples


# ============================================================================
# Optimization
# ============================================================================

def optimize_with_mipro(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    mode: Literal['light', 'medium', 'heavy'] = 'light',
) -> dspy.Module:
    """Optimize module using MIPROv2."""
    from dspy.teleprompt import MIPROv2

    print(f"\nRunning MIPROv2 optimization (mode={mode})...")

    optimizer = MIPROv2(
        metric=metric,
        auto=mode,
        num_threads=2,  # Conservative for API rate limits
    )

    optimized = optimizer.compile(
        module,
        trainset=trainset,
        requires_permission_to_run=False,
    )

    return optimized


def optimize_with_bootstrap(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric,
    max_demos: int = 4,
) -> dspy.Module:
    """Optimize module using BootstrapFewShot."""
    from dspy.teleprompt import BootstrapFewShot

    print(f"\nRunning BootstrapFewShot optimization...")

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_demos,
        max_labeled_demos=max_demos,
        max_rounds=2,
    )

    optimized = optimizer.compile(module, trainset=trainset)

    return optimized


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_module(
    module: dspy.Module,
    testset: list[dspy.Example],
    metric,
    name: str = "Module",
) -> float:
    """Evaluate a module on test set."""
    from dspy.evaluate import Evaluate

    print(f"\nEvaluating {name}...")

    evaluator = Evaluate(
        devset=testset,
        num_threads=2,
        display_progress=True,
        display_table=5,
    )

    result = evaluator(module, metric=metric)

    # Extract score from result (could be float or EvaluationResult object)
    if hasattr(result, 'score'):
        score = float(result.score)
    elif isinstance(result, (int, float)):
        score = float(result)
    else:
        # Try to extract from string representation
        score = float(str(result).split('%')[0].split()[-1]) / 100 if '%' in str(result) else 0.0

    print(f"{name} score: {score:.3f}")

    return score


# ============================================================================
# Prompt Export
# ============================================================================

def extract_optimized_prompt(module: dspy.Module) -> str:
    """Extract the optimized prompt template from a DSPy module."""
    # Get the predictor
    predictor = None
    for name, child in module.named_predictors():
        predictor = child
        break

    if predictor is None:
        return "Could not extract prompt - no predictor found"

    # Build prompt representation
    prompt_parts = []

    # Add signature docstring if available
    if hasattr(predictor, 'signature'):
        sig = predictor.signature
        if sig.__doc__:
            prompt_parts.append(f"# Task Description\n{sig.__doc__}\n")

        # Add input/output field descriptions
        prompt_parts.append("# Input Fields")
        for field_name, field in sig.input_fields.items():
            desc = getattr(field, 'desc', '') or ''
            prompt_parts.append(f"- {field_name}: {desc}")

        prompt_parts.append("\n# Output Fields")
        for field_name, field in sig.output_fields.items():
            desc = getattr(field, 'desc', '') or ''
            prompt_parts.append(f"- {field_name}: {desc}")

    # Add any demos/examples
    if hasattr(predictor, 'demos') and predictor.demos:
        prompt_parts.append("\n# Few-Shot Examples")
        for i, demo in enumerate(predictor.demos):
            prompt_parts.append(f"\n## Example {i+1}")
            for key, value in demo.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                prompt_parts.append(f"{key}: {value}")

    # Add extended signature if available (from MIPRO)
    if hasattr(predictor, 'extended_signature'):
        prompt_parts.append("\n# Optimized Instructions")
        prompt_parts.append(str(predictor.extended_signature))

    return "\n".join(prompt_parts)


def format_as_lightrag_prompt(optimized_prompt: str) -> str:
    """Format the optimized prompt for use in lightrag/prompt.py."""
    template = '''"""
RAG Response Prompt - Optimized by DSPy
========================================

Copy this to lightrag/prompt.py as PROMPTS['rag_response']
"""

PROMPTS['rag_response'] = """Answer based ONLY on this context:

{{context_data}}

---

{optimized_instructions}

Question: {{user_prompt}}

Answer (grounded only in context above):"""
'''

    return template.format(optimized_instructions=optimized_prompt)


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description='DSPy Prompt Optimization for LightRAG')
    parser.add_argument('--num-queries', '-n', type=int, help='Number of queries to use')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:9621', help='LightRAG server URL')
    parser.add_argument('--mode', '-m', choices=['light', 'medium', 'heavy'], default='light',
                        help='Optimization intensity')
    parser.add_argument('--optimizer', '-o', choices=['mipro', 'bootstrap', 'both'], default='bootstrap',
                        help='Optimizer to use')
    parser.add_argument('--module', choices=['simple', 'cot', 'reasoning'], default='cot',
                        help='DSPy module architecture')
    parser.add_argument('--fast-metric', action='store_true', help='Use fast heuristic metric instead of RAGAS')
    parser.add_argument('--export-prompt', type=str, help='Export optimized prompt to file')
    parser.add_argument('--save-module', type=str, help='Save optimized module to JSON')
    args = parser.parse_args()

    # Configure DSPy
    lm = dspy.LM(
        model=os.getenv('LLM_MODEL', 'openai/gpt-4o-mini'),
        api_key=os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY'),
        api_base=os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1'),
        temperature=0.1,
        max_tokens=32000,  # OpenRouter models support high limits
    )
    dspy.configure(lm=lm)

    print("=" * 70)
    print("DSPy Prompt Optimization for LightRAG")
    print("=" * 70)
    print(f"LLM: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Module: {args.module}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    # Load dataset
    dataset_path = Path(__file__).parent / 'pharma_test_dataset.json'
    dataset = load_dataset(dataset_path, args.num_queries)

    # Prepare examples
    examples = await prepare_dspy_examples(dataset, args.server)

    if len(examples) < 2:
        print("Error: Need at least 2 examples for optimization")
        return

    # Split into train/test
    split_idx = max(1, len(examples) - 2)
    trainset = examples[:split_idx]
    testset = examples[split_idx:]

    print(f"\nTrain set: {len(trainset)} examples")
    print(f"Test set: {len(testset)} examples")

    # Select module
    if args.module == 'simple':
        module = SimpleRAG()
    elif args.module == 'cot':
        module = ChainOfThoughtRAG()
    else:
        module = ReasoningRAG()

    # Select metric
    metric = fast_metric if args.fast_metric else ragas_metric

    # Evaluate baseline
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline_score = evaluate_module(module, testset, metric, "Baseline")

    # Optimize
    print("\n" + "=" * 70)
    print("OPTIMIZATION")
    print("=" * 70)

    optimized_module = None

    if args.optimizer in ['bootstrap', 'both']:
        optimized_module = optimize_with_bootstrap(module, trainset, metric)
        bootstrap_score = evaluate_module(optimized_module, testset, metric, "Bootstrap")

    if args.optimizer in ['mipro', 'both']:
        base_for_mipro = optimized_module if optimized_module else module
        optimized_module = optimize_with_mipro(base_for_mipro, trainset, metric, args.mode)
        mipro_score = evaluate_module(optimized_module, testset, metric, "MIPRO")

    # Extract and display optimized prompt
    print("\n" + "=" * 70)
    print("OPTIMIZED PROMPT")
    print("=" * 70)

    optimized_prompt = extract_optimized_prompt(optimized_module)
    print(optimized_prompt)

    # Export if requested
    if args.export_prompt:
        formatted = format_as_lightrag_prompt(optimized_prompt)
        with open(args.export_prompt, 'w') as f:
            f.write(formatted)
        print(f"\nExported to: {args.export_prompt}")

    if args.save_module:
        optimized_module.save(args.save_module)
        print(f"Saved module to: {args.save_module}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline score: {baseline_score:.3f}")
    if args.optimizer in ['bootstrap', 'both']:
        print(f"Bootstrap score: {bootstrap_score:.3f}")
    if args.optimizer in ['mipro', 'both']:
        print(f"MIPRO score: {mipro_score:.3f}")


if __name__ == '__main__':
    asyncio.run(main())
