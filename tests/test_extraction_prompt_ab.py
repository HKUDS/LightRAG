"""
A/B Test for Entity Extraction Prompts

Compares original vs optimized extraction prompts using real LLM calls.
Run with: pytest tests/test_extraction_prompt_ab.py -v --run-integration
Or directly: python tests/test_extraction_prompt_ab.py
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import tiktoken

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.prompt import PROMPTS
from lightrag.prompt_optimized import PROMPTS_OPTIMIZED

# =============================================================================
# Sample Texts for Testing
# =============================================================================

SAMPLE_TEXTS = {
    "covid_medical": {
        "name": "COVID-19 Medical",
        "text": """
COVID-19, caused by the SARS-CoV-2 virus, emerged in Wuhan, China in late 2019.
The disease spreads primarily through respiratory droplets and can cause symptoms
ranging from mild fever and cough to severe pneumonia and acute respiratory distress
syndrome (ARDS). The World Health Organization declared it a pandemic on March 11, 2020.

Risk factors for severe disease include advanced age, obesity, and pre-existing
conditions such as diabetes and cardiovascular disease. Vaccines developed by Pfizer,
Moderna, and AstraZeneca have shown high efficacy in preventing severe illness.
""",
    },
    "financial_market": {
        "name": "Financial Markets",
        "text": """
Stock markets faced a sharp downturn today as tech giants saw significant declines,
with the global tech index dropping by 3.4% in midday trading. Analysts attribute
the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after
reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted
a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%,
reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices
continued their rally, climbing to $87.60 per barrel, supported by supply constraints.

The Federal Reserve's upcoming policy announcement is expected to influence investor
confidence and overall market stability.
""",
    },
    "legal_regulatory": {
        "name": "Legal/Regulatory",
        "text": """
The merger between Acme Corp and Beta Industries requires approval from the Federal
Trade Commission. Legal counsel advised that the deal may face antitrust scrutiny
due to market concentration concerns in the semiconductor industry.

The European Commission has also opened an investigation into the proposed acquisition,
citing potential impacts on competition in the EU market. Both companies have agreed
to divest certain assets to address regulatory concerns.

Industry analysts expect the approval process to take 12-18 months, with final
clearance dependent on remedies proposed by the merging parties.
""",
    },
    "narrative_fiction": {
        "name": "Narrative Fiction",
        "text": """
While Alex clenched his jaw, the buzz of frustration dull against the backdrop of
Taylor's authoritarian certainty. It was this competitive undercurrent that kept him
alert, the sense that his and Jordan's shared commitment to discovery was an unspoken
rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment,
observed the device with something akin to reverence. "If this tech can be understood..."
Taylor said, their voice quieter, "It could change the game for us. For all of us."

Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's,
a wordless clash of wills softening into an uneasy truce.
""",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ExtractionResult:
    """Result from a single extraction run."""

    entities: list[dict]
    relations: list[dict]
    raw_output: str
    input_tokens: int
    format_errors: int

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def orphan_count(self) -> int:
        """Entities with no relationships."""
        entity_names = {e["name"].lower() for e in self.entities}
        connected = set()
        for r in self.relations:
            connected.add(r["source"].lower())
            connected.add(r["target"].lower())
        return len(entity_names - connected)

    @property
    def orphan_ratio(self) -> float:
        if self.entity_count == 0:
            return 0.0
        return self.orphan_count / self.entity_count


@dataclass
class ComparisonResult:
    """Comparison between two extraction results."""

    sample_name: str
    original: ExtractionResult
    optimized: ExtractionResult

    def entity_diff_pct(self) -> float:
        if self.original.entity_count == 0:
            return 0.0
        return (
            (self.optimized.entity_count - self.original.entity_count)
            / self.original.entity_count
            * 100
        )

    def relation_diff_pct(self) -> float:
        if self.original.relation_count == 0:
            return 0.0
        return (
            (self.optimized.relation_count - self.original.relation_count)
            / self.original.relation_count
            * 100
        )

    def token_diff_pct(self) -> float:
        if self.original.input_tokens == 0:
            return 0.0
        return (
            (self.optimized.input_tokens - self.original.input_tokens)
            / self.original.input_tokens
            * 100
        )


# =============================================================================
# Helper Functions
# =============================================================================


def format_prompt(
    prompts: dict, text: str, entity_types: str = "person, organization, location, concept, product, event, category, method"
) -> tuple[str, str]:
    """Format system and user prompts with the given text."""
    tuple_delimiter = prompts["DEFAULT_TUPLE_DELIMITER"]
    completion_delimiter = prompts["DEFAULT_COMPLETION_DELIMITER"]

    # Format examples
    examples = "\n".join(prompts["entity_extraction_examples"])
    examples = examples.format(
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
    )

    context = {
        "tuple_delimiter": tuple_delimiter,
        "completion_delimiter": completion_delimiter,
        "entity_types": entity_types,
        "language": "English",
        "examples": examples,
        "input_text": text,
    }

    system_prompt = prompts["entity_extraction_system_prompt"].format(**context)
    user_prompt = prompts["entity_extraction_user_prompt"].format(**context)

    return system_prompt, user_prompt


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


async def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API with the given prompts."""
    import openai

    client = openai.AsyncOpenAI()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content


def parse_extraction(output: str, tuple_delimiter: str = "<|#|>") -> tuple[list[dict], list[dict], int]:
    """Parse extraction output into entities and relations."""
    entities = []
    relations = []
    format_errors = 0

    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("<|COMPLETE|>"):
            continue

        parts = line.split(tuple_delimiter)

        if len(parts) >= 4 and parts[0].lower() == "entity":
            entities.append(
                {
                    "name": parts[1].strip(),
                    "type": parts[2].strip(),
                    "description": parts[3].strip() if len(parts) > 3 else "",
                }
            )
        elif len(parts) >= 5 and parts[0].lower() == "relation":
            relations.append(
                {
                    "source": parts[1].strip(),
                    "target": parts[2].strip(),
                    "keywords": parts[3].strip(),
                    "description": parts[4].strip() if len(parts) > 4 else "",
                }
            )
        elif line and not line.startswith("**") and tuple_delimiter in line:
            # Line looks like it should be parsed but failed
            format_errors += 1

    return entities, relations, format_errors


async def run_extraction(prompts: dict, text: str) -> ExtractionResult:
    """Run extraction with the given prompts on the text."""
    system_prompt, user_prompt = format_prompt(prompts, text)
    input_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)

    output = await call_llm(system_prompt, user_prompt)
    entities, relations, format_errors = parse_extraction(output)

    return ExtractionResult(
        entities=entities,
        relations=relations,
        raw_output=output,
        input_tokens=input_tokens,
        format_errors=format_errors,
    )


def print_comparison_table(results: list[ComparisonResult]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("ENTITY EXTRACTION PROMPT A/B COMPARISON")
    print("=" * 80)

    total_orig_entities = 0
    total_opt_entities = 0
    total_orig_relations = 0
    total_opt_relations = 0
    total_orig_tokens = 0
    total_opt_tokens = 0

    for r in results:
        print(f"\n--- {r.sample_name} ---")
        print(f"{'Metric':<20} {'Original':>12} {'Optimized':>12} {'Diff':>12}")
        print("-" * 56)

        print(f"{'Entities':<20} {r.original.entity_count:>12} {r.optimized.entity_count:>12} {r.entity_diff_pct():>+11.0f}%")
        print(f"{'Relations':<20} {r.original.relation_count:>12} {r.optimized.relation_count:>12} {r.relation_diff_pct():>+11.0f}%")
        print(f"{'Orphan Ratio':<20} {r.original.orphan_ratio:>11.0%} {r.optimized.orphan_ratio:>11.0%} {'':>12}")
        print(f"{'Format Errors':<20} {r.original.format_errors:>12} {r.optimized.format_errors:>12}")
        print(f"{'Input Tokens':<20} {r.original.input_tokens:>12,} {r.optimized.input_tokens:>12,} {r.token_diff_pct():>+11.0f}%")

        total_orig_entities += r.original.entity_count
        total_opt_entities += r.optimized.entity_count
        total_orig_relations += r.original.relation_count
        total_opt_relations += r.optimized.relation_count
        total_orig_tokens += r.original.input_tokens
        total_opt_tokens += r.optimized.input_tokens

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} {'Original':>12} {'Optimized':>12} {'Diff':>12}")
    print("-" * 56)

    ent_diff = (total_opt_entities - total_orig_entities) / total_orig_entities * 100 if total_orig_entities else 0
    rel_diff = (total_opt_relations - total_orig_relations) / total_orig_relations * 100 if total_orig_relations else 0
    tok_diff = (total_opt_tokens - total_orig_tokens) / total_orig_tokens * 100 if total_orig_tokens else 0

    print(f"{'Total Entities':<20} {total_orig_entities:>12} {total_opt_entities:>12} {ent_diff:>+11.0f}%")
    print(f"{'Total Relations':<20} {total_orig_relations:>12} {total_opt_relations:>12} {rel_diff:>+11.0f}%")
    print(f"{'Total Input Tokens':<20} {total_orig_tokens:>12,} {total_opt_tokens:>12,} {tok_diff:>+11.0f}%")

    # Recommendation
    print("\n" + "-" * 56)
    if tok_diff < -30 and ent_diff >= -10:
        print("RECOMMENDATION: Use OPTIMIZED prompt (significant token savings, comparable extraction)")
    elif ent_diff > 20 and tok_diff < 0:
        print("RECOMMENDATION: Use OPTIMIZED prompt (better extraction AND token savings)")
    elif ent_diff < -20:
        print("RECOMMENDATION: Keep ORIGINAL prompt (optimized extracts significantly fewer entities)")
    else:
        print("RECOMMENDATION: Both prompts are comparable - consider token cost vs extraction breadth")

    print("=" * 80 + "\n")


# =============================================================================
# Pytest Tests
# =============================================================================


class TestExtractionPromptAB:
    """A/B testing for entity extraction prompts."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_compare_all_samples(self) -> None:
        """Compare prompts across all sample texts."""
        results = []

        for key, sample in SAMPLE_TEXTS.items():
            print(f"\nProcessing: {sample['name']}...")

            original = await run_extraction(PROMPTS, sample["text"])
            optimized = await run_extraction(PROMPTS_OPTIMIZED, sample["text"])

            results.append(
                ComparisonResult(
                    sample_name=sample["name"],
                    original=original,
                    optimized=optimized,
                )
            )

        print_comparison_table(results)

        # Basic assertions
        for r in results:
            assert r.original.format_errors == 0, f"Original had format errors on {r.sample_name}"
            assert r.optimized.format_errors == 0, f"Optimized had format errors on {r.sample_name}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_single_sample(self) -> None:
        """Quick test with just one sample."""
        sample = SAMPLE_TEXTS["covid_medical"]

        original = await run_extraction(PROMPTS, sample["text"])
        optimized = await run_extraction(PROMPTS_OPTIMIZED, sample["text"])

        result = ComparisonResult(
            sample_name=sample["name"],
            original=original,
            optimized=optimized,
        )

        print_comparison_table([result])

        assert original.entity_count > 0, "Original should extract entities"
        assert optimized.entity_count > 0, "Optimized should extract entities"
        assert optimized.format_errors == 0, "Optimized should have no format errors"


# =============================================================================
# CLI Runner
# =============================================================================


async def main() -> None:
    """Run A/B comparison from command line."""
    print("Starting Entity Extraction Prompt A/B Test...")
    print("This will make real API calls to OpenAI.\n")

    results = []

    for key, sample in SAMPLE_TEXTS.items():
        print(f"Processing: {sample['name']}...")

        original = await run_extraction(PROMPTS, sample["text"])
        optimized = await run_extraction(PROMPTS_OPTIMIZED, sample["text"])

        results.append(
            ComparisonResult(
                sample_name=sample["name"],
                original=original,
                optimized=optimized,
            )
        )

    print_comparison_table(results)


if __name__ == "__main__":
    asyncio.run(main())
