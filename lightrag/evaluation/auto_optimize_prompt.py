#!/usr/bin/env python3
"""
Automated DSPy Prompt Optimization Loop

Runs continuous optimization until target RAGAS score is achieved:
1. Generate prompt variants (DSPy + LLM mutation)
2. Test each variant against eval dataset using RAGAS
3. Compare to baseline - keep winners
4. Inject winning prompt into lightrag/prompt.py
5. Repeat until target or max iterations

Usage:
    # Full optimization run (12 queries, up to 10 iterations)
    python auto_optimize_prompt.py --target-ragas 0.95

    # Quick test (3 queries, 3 iterations)
    python auto_optimize_prompt.py --quick

    # Optimize specific prompt
    python auto_optimize_prompt.py --prompt-key rag_response

    # Dry run (no prompt.py modification)
    python auto_optimize_prompt.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

# Load environment
load_dotenv()

# Ensure we can import lightrag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from openai import AsyncOpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class OptimizationConfig:
    """Configuration for the optimization loop."""

    target_ragas: float = 0.95
    target_faithfulness: float = 0.95
    max_iterations: int = 10
    queries_per_eval: int | None = None  # None = use all
    quick_eval_queries: int = 3
    prompt_key: str = 'naive_rag_response'
    server_url: str = 'http://localhost:9621'
    results_dir: Path = field(default_factory=lambda: Path('optimization_results'))
    dry_run: bool = False
    quick_mode: bool = False
    mode: str = 'naive'  # Query mode: naive, mix, etc.


@dataclass
class PromptVariant:
    """A prompt variant to test."""

    name: str
    prompt: str
    source: str = 'unknown'  # How it was generated: 'baseline', 'llm_mutation', 'dspy', etc.


@dataclass
class QueryResult:
    """Result for a single query evaluation."""

    question: str
    answer: str
    faithfulness: float
    relevance: float
    latency_ms: float


@dataclass
class EvalResult:
    """Evaluation result for a prompt variant."""

    variant: PromptVariant
    query_results: list[QueryResult]
    faithfulness: float
    relevance: float
    ragas_score: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            'variant_name': self.variant.name,
            'source': self.variant.source,
            'faithfulness': self.faithfulness,
            'relevance': self.relevance,
            'ragas_score': self.ragas_score,
            'total_latency_ms': self.total_latency_ms,
            'query_count': len(self.query_results),
        }


@dataclass
class IterationResult:
    """Result of a single optimization iteration."""

    iteration: int
    baseline_score: float
    variants_tested: list[EvalResult]
    best_variant: EvalResult | None
    improved: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# LLM Mutation Prompts
# ============================================================================

MUTATION_PROMPT = """You are a prompt engineer specializing in RAG (Retrieval-Augmented Generation) systems.

Given this RAG response prompt and its RAGAS evaluation scores, improve it to achieve higher scores.

## Current Prompt:
```
{current_prompt}
```

## Current Scores:
- Faithfulness: {faithfulness:.3f} (target: {target_faith:.2f})
- Answer Relevance: {relevance:.3f} (target: {target_rel:.2f})
- Combined RAGAS: {ragas_score:.3f}

## Score Interpretations:
- **Faithfulness** measures if the answer contains ONLY facts from the context (no hallucination)
- **Answer Relevance** measures if the answer directly addresses the question asked

## Improvement Guidelines:
1. If faithfulness < {target_faith}: Add stricter grounding language, explicit warnings against adding external knowledge
2. If relevance < {target_rel}: Improve answer structure guidance, ensure question terms are used in response
3. Keep the prompt concise - overly long prompts don't help
4. Preserve the {{content_data}}, {{user_prompt}}, {{response_type}}, and {{coverage_guidance}} placeholders exactly

## Format Requirements:
- The prompt MUST contain these exact placeholders: {{content_data}}, {{user_prompt}}, {{response_type}}, {{coverage_guidance}}
- Keep the general structure but improve the instructions

Output ONLY the improved prompt text. No explanations, no markdown code blocks, just the raw prompt:
"""

AGGRESSIVE_MUTATION_PROMPT = """You are an expert prompt engineer. The current RAG prompt is underperforming.

Current Prompt:
```
{current_prompt}
```

Scores: Faith={faithfulness:.3f}, Rel={relevance:.3f}, Target={target_faith:.2f}

The prompt is {gap_description}. Create a SIGNIFICANTLY DIFFERENT variant that:
{mutation_focus}

CRITICAL: Keep these exact placeholders: {{content_data}}, {{user_prompt}}, {{response_type}}, {{coverage_guidance}}

Output ONLY the new prompt (no explanations):
"""


# ============================================================================
# Prompt Optimizer Class
# ============================================================================


class PromptOptimizer:
    """Automated prompt optimization loop."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.history: list[IterationResult] = []
        self.best_prompt: str | None = None
        self.best_score: float = 0.0
        self.current_scores: EvalResult | None = None

        # Setup OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1'),
        )
        self.model = os.getenv('LLM_MODEL', 'gpt-4o-mini')

        # Load test dataset
        self.test_queries = self._load_test_queries()

        # Setup results directory
        self.run_dir = self._setup_run_directory()

    def _load_test_queries(self) -> list[dict]:
        """Load test queries from pharma dataset."""
        dataset_path = Path(__file__).parent / 'pharma_test_dataset.json'
        with open(dataset_path) as f:
            data = json.load(f)

        queries = data.get('test_cases', data)

        # Apply query limit
        if self.config.quick_mode:
            return queries[: self.config.quick_eval_queries]
        elif self.config.queries_per_eval:
            return queries[: self.config.queries_per_eval]
        return queries

    def _setup_run_directory(self) -> Path:
        """Create a directory for this optimization run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.config.results_dir / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create 'latest' symlink
        latest_link = self.config.results_dir / 'latest'
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)

        return run_dir

    def get_current_prompt(self) -> str:
        """Read the current prompt from lightrag/prompt.py."""
        prompt_file = Path(__file__).parent.parent / 'prompt.py'
        content = prompt_file.read_text()

        # Extract the target prompt using regex
        pattern = rf"PROMPTS\['{self.config.prompt_key}'\]\s*=\s*\"\"\"(.*?)\"\"\""
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1)

        raise ValueError(f"Could not find PROMPTS['{self.config.prompt_key}'] in prompt.py")

    def inject_prompt(self, new_prompt: str) -> bool:
        """Update lightrag/prompt.py with the winning prompt."""
        if self.config.dry_run:
            logger.info('🔒 DRY RUN: Would inject prompt but skipping')
            return False

        prompt_file = Path(__file__).parent.parent / 'prompt.py'

        # Read current file
        content = prompt_file.read_text()

        # Create backup
        backup = prompt_file.with_suffix('.py.bak')
        shutil.copy(prompt_file, backup)
        logger.info(f'📦 Backup created: {backup}')

        # Replace the prompt
        pattern = rf"(PROMPTS\['{self.config.prompt_key}'\]\s*=\s*)\"\"\".*?\"\"\""
        replacement = f'\\1"""{new_prompt}"""'

        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        if new_content == content:
            logger.warning('⚠️ No changes made - pattern not found')
            return False

        # Write updated file
        prompt_file.write_text(new_content)
        logger.info(f'💉 Injected optimized prompt into {prompt_file}')
        return True

    async def get_context(self, query: str) -> str:
        """Get context from LightRAG server."""
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f'{self.config.server_url}/query',
                    json={
                        'query': query,
                        'mode': self.config.mode,
                        'only_need_context': True,
                    },
                )
                data = response.json()
                return data.get('response', '')
            except Exception as e:
                logger.error(f'Failed to get context: {e}')
                return ''

    async def call_llm(self, prompt_template: str, context: str, question: str) -> tuple[str, float]:
        """Call LLM with a prompt and return (answer, latency_ms)."""
        start = time.perf_counter()

        # Format the prompt - handle both {content_data} and {context_data} placeholders
        try:
            formatted = prompt_template.format(
                content_data=context,  # Original LightRAG placeholder
                context_data=context,  # Alternative placeholder (for mutations)
                user_prompt=question,
                response_type='Multiple Paragraphs',
                coverage_guidance='',  # Empty for testing
            )
        except KeyError as e:
            logger.error(f'Prompt template missing placeholder: {e}')
            return '', 0.0

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': formatted}],
                temperature=0.1,
                max_tokens=2000,
            )
            latency = (time.perf_counter() - start) * 1000
            return response.choices[0].message.content or '', latency
        except Exception as e:
            logger.error(f'LLM call failed: {e}')
            return '', 0.0

    async def call_llm_for_mutation(self, mutation_prompt: str) -> str:
        """Call LLM to generate a mutated prompt."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': mutation_prompt}],
                temperature=0.7,  # Higher temperature for creativity
                max_tokens=2000,
            )
            return response.choices[0].message.content or ''
        except Exception as e:
            logger.error(f'Mutation LLM call failed: {e}')
            return ''

    def run_ragas_eval(
        self, question: str, answer: str, context: str, ground_truth: str
    ) -> tuple[float, float]:
        """Run RAGAS evaluation and return (faithfulness, relevance)."""
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        if not answer or not answer.strip():
            return 0.0, 0.0

        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [[context]],
            'ground_truth': [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        try:
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
            embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=llm,
                embeddings=embeddings,
            )

            faith = result['faithfulness']
            rel = result['answer_relevancy']

            if isinstance(faith, list):
                faith = faith[0]
            if isinstance(rel, list):
                rel = rel[0]

            return float(faith), float(rel)
        except Exception as e:
            logger.error(f'RAGAS eval failed: {e}')
            return 0.0, 0.0

    async def evaluate_variant(self, variant: PromptVariant) -> EvalResult:
        """Evaluate a single prompt variant."""
        query_results = []
        total_latency = 0.0

        for i, q in enumerate(self.test_queries):
            question = q['question']
            ground_truth = q.get('ground_truth', '')

            logger.info(f'    [{i + 1}/{len(self.test_queries)}] {question[:50]}...')

            # Get context from server
            context = await self.get_context(question)

            if not context or context == 'No relevant context found for the query.':
                logger.warning(f'      ⚠️ No context, skipping')
                continue

            # Call LLM with the variant prompt
            answer, latency = await self.call_llm(variant.prompt, context, question)
            total_latency += latency

            if not answer:
                logger.warning(f'      ⚠️ Empty answer, skipping')
                continue

            # Run RAGAS eval
            faith, rel = self.run_ragas_eval(question, answer, context, ground_truth)

            query_results.append(
                QueryResult(
                    question=question[:60],
                    answer=answer[:200],
                    faithfulness=faith,
                    relevance=rel,
                    latency_ms=latency,
                )
            )

            logger.info(f'      Faith: {faith:.3f} | Rel: {rel:.3f} | {latency:.0f}ms')

        if not query_results:
            return EvalResult(
                variant=variant,
                query_results=[],
                faithfulness=0.0,
                relevance=0.0,
                ragas_score=0.0,
                total_latency_ms=total_latency,
            )

        avg_faith = mean([r.faithfulness for r in query_results])
        avg_rel = mean([r.relevance for r in query_results])
        # Weighted RAGAS score (faith slightly higher)
        ragas_score = 0.6 * avg_faith + 0.4 * avg_rel

        return EvalResult(
            variant=variant,
            query_results=query_results,
            faithfulness=avg_faith,
            relevance=avg_rel,
            ragas_score=ragas_score,
            total_latency_ms=total_latency,
        )

    async def generate_mutation(self, base_prompt: str, aggressive: bool = False) -> PromptVariant:
        """Generate a mutated prompt using LLM."""
        if aggressive and self.current_scores:
            # Determine the gap and focus
            faith_gap = self.config.target_faithfulness - self.current_scores.faithfulness
            rel_gap = 0.90 - self.current_scores.relevance

            if faith_gap > rel_gap:
                gap_description = 'hallucinating (low faithfulness)'
                mutation_focus = """
1. Add EXPLICIT warnings: "Do NOT add information not in the context"
2. Add verification step: "Before each claim, check it appears in context"
3. Add consequence framing: "Claims without context support = failure"
"""
            else:
                gap_description = 'missing the point (low relevance)'
                mutation_focus = """
1. Emphasize using question's exact terminology in the answer
2. Add structure guidance for different question types
3. Require first sentence to directly answer the question
"""

            mutation_request = AGGRESSIVE_MUTATION_PROMPT.format(
                current_prompt=base_prompt,
                faithfulness=self.current_scores.faithfulness,
                relevance=self.current_scores.relevance,
                target_faith=self.config.target_faithfulness,
                gap_description=gap_description,
                mutation_focus=mutation_focus,
            )
            name = 'aggressive_mutation'
        else:
            mutation_request = MUTATION_PROMPT.format(
                current_prompt=base_prompt,
                faithfulness=self.current_scores.faithfulness if self.current_scores else 0.0,
                relevance=self.current_scores.relevance if self.current_scores else 0.0,
                ragas_score=self.current_scores.ragas_score if self.current_scores else 0.0,
                target_faith=self.config.target_faithfulness,
                target_rel=0.90,
            )
            name = 'llm_mutation'

        mutated = await self.call_llm_for_mutation(mutation_request)

        # Validate the mutated prompt has required placeholders
        required = ['{context_data}', '{user_prompt}', '{response_type}', '{coverage_guidance}']
        for placeholder in required:
            if placeholder not in mutated:
                logger.warning(f'⚠️ Mutation missing {placeholder}, adding it')
                # Try to fix common issues
                if '{context_data}' not in mutated and '---Context---' in mutated:
                    mutated = mutated.replace('---Context---', '---Context---\n{context_data}')

        return PromptVariant(name=name, prompt=mutated, source='llm_mutation')

    async def generate_variants(self) -> list[PromptVariant]:
        """Generate prompt variants for testing."""
        variants = []

        base_prompt = self.best_prompt or self.get_current_prompt()

        # Always include the current best/baseline
        variants.append(PromptVariant(name='baseline', prompt=base_prompt, source='baseline'))

        # Generate LLM mutation
        logger.info('  🧬 Generating LLM mutation...')
        mutation = await self.generate_mutation(base_prompt)
        variants.append(mutation)

        # Generate aggressive mutation if we have scores
        if self.current_scores and self.current_scores.ragas_score < self.config.target_ragas:
            logger.info('  🧬 Generating aggressive mutation...')
            aggressive = await self.generate_mutation(base_prompt, aggressive=True)
            variants.append(aggressive)

        return variants

    def select_winner(self, results: list[EvalResult]) -> EvalResult | None:
        """Select the best variant from results."""
        if not results:
            return None

        # Sort by RAGAS score, then faithfulness as tiebreaker
        sorted_results = sorted(results, key=lambda r: (r.ragas_score, r.faithfulness), reverse=True)

        return sorted_results[0]

    def save_iteration_results(self, iteration: int, results: list[EvalResult], best: EvalResult | None):
        """Save results for an iteration to disk."""
        iteration_dir = self.run_dir / f'iteration_{iteration:03d}'
        iteration_dir.mkdir(exist_ok=True)

        # Save summary
        summary = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'best_score': best.ragas_score if best else 0.0,
            'variants': [r.to_dict() for r in results],
        }

        with open(iteration_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save the best prompt
        if best:
            with open(iteration_dir / 'best_prompt.txt', 'w') as f:
                f.write(best.variant.prompt)

        logger.info(f'  📁 Results saved to {iteration_dir}')

    def check_target_reached(self, result: EvalResult) -> bool:
        """Check if optimization target has been reached."""
        return (
            result.ragas_score >= self.config.target_ragas
            and result.faithfulness >= self.config.target_faithfulness
        )

    async def run(self) -> EvalResult | None:
        """Run the optimization loop."""
        logger.info('=' * 70)
        logger.info('🚀 AUTOMATED PROMPT OPTIMIZATION')
        logger.info('=' * 70)
        logger.info(f'Target RAGAS: {self.config.target_ragas}')
        logger.info(f'Target Faithfulness: {self.config.target_faithfulness}')
        logger.info(f'Max Iterations: {self.config.max_iterations}')
        logger.info(f'Queries per eval: {len(self.test_queries)}')
        logger.info(f'Mode: {self.config.mode}')
        logger.info(f'Dry run: {self.config.dry_run}')
        logger.info('=' * 70)

        for iteration in range(self.config.max_iterations):
            logger.info(f'\n📍 ITERATION {iteration + 1}/{self.config.max_iterations}')
            logger.info('-' * 50)

            # Generate variants
            logger.info('🔄 Generating variants...')
            variants = await self.generate_variants()
            logger.info(f'  Generated {len(variants)} variants')

            # Evaluate each variant
            results = []
            for variant in variants:
                logger.info(f'\n📝 Testing "{variant.name}" ({variant.source})')
                result = await self.evaluate_variant(variant)
                results.append(result)
                logger.info(
                    f'  📊 Result: Faith={result.faithfulness:.3f} '
                    f'Rel={result.relevance:.3f} RAGAS={result.ragas_score:.3f}'
                )

            # Select winner
            winner = self.select_winner(results)

            if winner:
                improved = winner.ragas_score > self.best_score

                if improved:
                    logger.info(f'\n✨ NEW BEST: "{winner.variant.name}" ({winner.ragas_score:.3f})')
                    self.best_prompt = winner.variant.prompt
                    self.best_score = winner.ragas_score
                    self.current_scores = winner

                    # Inject the winning prompt
                    if winner.variant.source != 'baseline':
                        self.inject_prompt(winner.variant.prompt)

                    # Check if target reached
                    if self.check_target_reached(winner):
                        logger.info('\n🎯 TARGET REACHED!')
                        logger.info(f'  Faithfulness: {winner.faithfulness:.3f} ≥ {self.config.target_faithfulness}')
                        logger.info(f'  RAGAS: {winner.ragas_score:.3f} ≥ {self.config.target_ragas}')

                        self.save_iteration_results(iteration, results, winner)
                        self._save_final_summary(winner)
                        return winner
                else:
                    logger.info(f'\n📉 No improvement (best: {self.best_score:.3f})')
                    self.current_scores = winner  # Update for next mutation

            # Save iteration results
            self.save_iteration_results(iteration, results, winner)

            # Record in history
            self.history.append(
                IterationResult(
                    iteration=iteration,
                    baseline_score=self.best_score,
                    variants_tested=results,
                    best_variant=winner,
                    improved=winner.ragas_score > self.best_score if winner else False,
                )
            )

        # Max iterations reached
        logger.info('\n⏰ Max iterations reached')
        if self.best_prompt:
            logger.info(f'Best score achieved: {self.best_score:.3f}')
            self._save_final_summary(self.current_scores)

        return self.current_scores

    def _save_final_summary(self, final: EvalResult | None):
        """Save final optimization summary."""
        summary = {
            'final_score': final.ragas_score if final else 0.0,
            'final_faithfulness': final.faithfulness if final else 0.0,
            'final_relevance': final.relevance if final else 0.0,
            'total_iterations': len(self.history),
            'target_ragas': self.config.target_ragas,
            'target_faithfulness': self.config.target_faithfulness,
            'target_reached': self.check_target_reached(final) if final else False,
            'timestamp': datetime.now().isoformat(),
        }

        with open(self.run_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        if final:
            with open(self.run_dir / 'final_prompt.txt', 'w') as f:
                f.write(final.variant.prompt)

        logger.info(f'\n📁 Final results: {self.run_dir}')


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description='Automated DSPy Prompt Optimization')
    parser.add_argument('--target-ragas', '-t', type=float, default=0.95, help='Target RAGAS score')
    parser.add_argument(
        '--target-faithfulness', '-f', type=float, default=0.95, help='Target faithfulness score'
    )
    parser.add_argument('--max-iterations', '-i', type=int, default=10, help='Max optimization iterations')
    parser.add_argument('--server', '-s', type=str, default='http://localhost:9621', help='LightRAG server URL')
    parser.add_argument(
        '--prompt-key',
        '-p',
        type=str,
        default=None,
        help='Prompt key to optimize (auto-selected from mode if not specified)',
    )
    parser.add_argument('--mode', '-m', type=str, default='naive', help='Query mode (naive, mix, local, global)')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode (3 queries)')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Do not modify prompt.py')
    parser.add_argument('--num-queries', '-n', type=int, help='Number of queries to use')
    parser.add_argument('--all-modes', '-a', action='store_true', help='Optimize both naive and mix prompts')
    args = parser.parse_args()

    # Auto-select prompt key based on mode
    # naive mode → naive_rag_response
    # mix/local/global mode → rag_response
    prompt_key = args.prompt_key
    if prompt_key is None:
        if args.mode == 'naive':
            prompt_key = 'naive_rag_response'
        else:
            prompt_key = 'rag_response'

    # Determine which mode/prompt combinations to optimize
    if args.all_modes:
        # Optimize both naive and mix prompts
        modes_to_optimize = [
            ('naive', 'naive_rag_response'),
            ('mix', 'rag_response'),
        ]
    else:
        modes_to_optimize = [(args.mode, prompt_key)]

    results = {}
    for mode, key in modes_to_optimize:
        print(f'\n{"=" * 70}')
        print(f'🎯 Optimizing {key} (mode: {mode})')
        print('=' * 70)

        config = OptimizationConfig(
            target_ragas=args.target_ragas,
            target_faithfulness=args.target_faithfulness,
            max_iterations=args.max_iterations,
            server_url=args.server,
            prompt_key=key,
            mode=mode,
            quick_mode=args.quick,
            dry_run=args.dry_run,
            queries_per_eval=args.num_queries,
        )

        optimizer = PromptOptimizer(config)
        result = await optimizer.run()
        results[(mode, key)] = result

        if result:
            print(f'\n📊 Results for {key}:')
            print(f'   Faithfulness: {result.faithfulness:.3f}')
            print(f'   Relevance: {result.relevance:.3f}')
            print(f'   RAGAS Score: {result.ragas_score:.3f}')

    # Print final summary
    print('\n' + '=' * 70)
    print('📊 FINAL SUMMARY')
    print('=' * 70)
    for (mode, key), result in results.items():
        if result:
            print(f'{key} ({mode}): RAGAS={result.ragas_score:.3f} (F={result.faithfulness:.3f}, R={result.relevance:.3f})')
        else:
            print(f'{key} ({mode}): No improvement found')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
