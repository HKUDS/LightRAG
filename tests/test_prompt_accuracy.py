"""
Accuracy tests for optimized prompts.
Validates that optimized prompts produce correct, parseable outputs.

Run with: uv run --extra test python tests/test_prompt_accuracy.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.prompt import PROMPTS


# =============================================================================
# Test Data
# =============================================================================

KEYWORD_TEST_QUERIES = [
    {
        "query": "What are the main causes of climate change and how do they affect polar ice caps?",
        "expected_high": ["climate change", "causes", "effects"],
        "expected_low": ["polar ice caps", "greenhouse"],
    },
    {
        "query": "How did Apple's iPhone sales compare to Samsung Galaxy in Q3 2024?",
        "expected_high": ["sales comparison", "smartphone"],
        "expected_low": ["Apple", "iPhone", "Samsung", "Galaxy", "Q3 2024"],
    },
    {
        "query": "hello",  # Trivial - should return empty
        "expected_high": [],
        "expected_low": [],
    },
]

ORPHAN_TEST_CASES = [
    {
        "orphan": {"name": "Pfizer", "type": "organization", "desc": "Pharmaceutical company that developed COVID-19 vaccine"},
        "candidate": {"name": "Moderna", "type": "organization", "desc": "Biotechnology company that developed mRNA COVID-19 vaccine"},
        "should_connect": True,
        "reason": "Both are COVID-19 vaccine developers",
    },
    {
        "orphan": {"name": "Mount Everest", "type": "location", "desc": "Highest mountain in the world, located in the Himalayas"},
        "candidate": {"name": "Python Programming", "type": "concept", "desc": "Popular programming language used for data science"},
        "should_connect": False,
        "reason": "No logical connection between mountain and programming language",
    },
]

SUMMARIZATION_TEST_CASES = [
    {
        "name": "Albert Einstein",
        "type": "Entity",
        "descriptions": [
            '{"description": "Albert Einstein was a German-born theoretical physicist."}',
            '{"description": "Einstein developed the theory of relativity and won the Nobel Prize in Physics in 1921."}',
            '{"description": "He is widely regarded as one of the most influential scientists of the 20th century."}',
        ],
        "must_contain": ["physicist", "relativity", "Nobel Prize", "influential"],
    },
]

RAG_TEST_CASES = [
    {
        "query": "What is the capital of France?",
        "context": "Paris is the capital and largest city of France. It has a population of over 2 million people.",
        "must_contain": ["Paris"],
        "must_not_contain": ["[1]", "[2]", "References"],
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


async def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API with a single prompt."""
    import openai

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    raw_output: str = ""


# =============================================================================
# Test Functions
# =============================================================================


async def test_keywords_extraction() -> list[TestResult]:
    """Test keywords extraction prompt."""
    results = []

    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    for case in KEYWORD_TEST_QUERIES:
        prompt = PROMPTS["keywords_extraction"].format(
            examples=examples,
            query=case["query"]
        )

        output = await call_llm(prompt)

        # Try to parse JSON
        try:
            # Clean potential markdown
            clean = output.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]

            parsed = json.loads(clean)

            has_high = "high_level_keywords" in parsed
            has_low = "low_level_keywords" in parsed
            is_list_high = isinstance(parsed.get("high_level_keywords"), list)
            is_list_low = isinstance(parsed.get("low_level_keywords"), list)

            if has_high and has_low and is_list_high and is_list_low:
                # Check if trivial query returns empty
                if case["expected_high"] == [] and case["expected_low"] == []:
                    passed = len(parsed["high_level_keywords"]) == 0 and len(parsed["low_level_keywords"]) == 0
                    details = "Empty lists returned for trivial query" if passed else f"Non-empty for trivial: {parsed}"
                else:
                    # Check that some expected keywords are present (case-insensitive)
                    high_lower = [k.lower() for k in parsed["high_level_keywords"]]
                    low_lower = [k.lower() for k in parsed["low_level_keywords"]]
                    all_keywords = " ".join(high_lower + low_lower)

                    found_high = sum(1 for exp in case["expected_high"] if exp.lower() in all_keywords)
                    found_low = sum(1 for exp in case["expected_low"] if exp.lower() in all_keywords)

                    passed = found_high > 0 or found_low > 0
                    details = f"Found {found_high}/{len(case['expected_high'])} high, {found_low}/{len(case['expected_low'])} low"
            else:
                passed = False
                details = f"Missing keys or wrong types: has_high={has_high}, has_low={has_low}"

        except json.JSONDecodeError as e:
            passed = False
            details = f"JSON parse error: {e}"

        results.append(TestResult(
            name=f"Keywords: {case['query'][:40]}...",
            passed=passed,
            details=details,
            raw_output=output[:200]
        ))

    return results


async def test_orphan_validation() -> list[TestResult]:
    """Test orphan connection validation prompt."""
    results = []

    for case in ORPHAN_TEST_CASES:
        prompt = PROMPTS["orphan_connection_validation"].format(
            orphan_name=case["orphan"]["name"],
            orphan_type=case["orphan"]["type"],
            orphan_description=case["orphan"]["desc"],
            candidate_name=case["candidate"]["name"],
            candidate_type=case["candidate"]["type"],
            candidate_description=case["candidate"]["desc"],
            similarity_score=0.85,
        )

        output = await call_llm(prompt)

        try:
            # Clean potential markdown
            clean = output.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]

            parsed = json.loads(clean)

            has_should_connect = "should_connect" in parsed
            has_confidence = "confidence" in parsed
            has_reasoning = "reasoning" in parsed

            if has_should_connect and has_confidence and has_reasoning:
                correct_decision = parsed["should_connect"] == case["should_connect"]
                valid_confidence = 0.0 <= parsed["confidence"] <= 1.0

                passed = correct_decision and valid_confidence
                details = f"Decision: {parsed['should_connect']} (expected {case['should_connect']}), confidence: {parsed['confidence']:.2f}"
            else:
                passed = False
                details = f"Missing keys: should_connect={has_should_connect}, confidence={has_confidence}, reasoning={has_reasoning}"

        except json.JSONDecodeError as e:
            passed = False
            details = f"JSON parse error: {e}"

        results.append(TestResult(
            name=f"Orphan: {case['orphan']['name']} ↔ {case['candidate']['name']}",
            passed=passed,
            details=details,
            raw_output=output[:200]
        ))

    return results


async def test_entity_summarization() -> list[TestResult]:
    """Test entity summarization prompt."""
    results = []

    for case in SUMMARIZATION_TEST_CASES:
        prompt = PROMPTS["summarize_entity_descriptions"].format(
            description_name=case["name"],
            description_type=case["type"],
            description_list="\n".join(case["descriptions"]),
            summary_length=200,
            language="English",
        )

        output = await call_llm(prompt)

        # Check if required terms are present
        output_lower = output.lower()
        found = [term for term in case["must_contain"] if term.lower() in output_lower]
        missing = [term for term in case["must_contain"] if term.lower() not in output_lower]

        # Check it's not empty and mentions the entity
        has_content = len(output.strip()) > 50
        mentions_entity = case["name"].lower() in output_lower

        passed = len(found) >= len(case["must_contain"]) // 2 and has_content and mentions_entity
        details = f"Found {len(found)}/{len(case['must_contain'])} terms, mentions entity: {mentions_entity}"
        if missing:
            details += f", missing: {missing}"

        results.append(TestResult(
            name=f"Summarize: {case['name']}",
            passed=passed,
            details=details,
            raw_output=output[:200]
        ))

    return results


async def test_naive_rag_response() -> list[TestResult]:
    """Test naive RAG response prompt."""
    results = []

    for case in RAG_TEST_CASES:
        prompt = PROMPTS["naive_rag_response"].format(
            response_type="concise paragraph",
            user_prompt=case["query"],
            content_data=case["context"],
        )

        output = await call_llm(prompt)

        # Check must_contain
        output_lower = output.lower()
        found = [term for term in case["must_contain"] if term.lower() in output_lower]

        # Check must_not_contain (citation markers)
        violations = [term for term in case["must_not_contain"] if term in output]

        passed = len(found) == len(case["must_contain"]) and len(violations) == 0
        details = f"Found {len(found)}/{len(case['must_contain'])} required terms"
        if violations:
            details += f", VIOLATIONS: {violations}"

        results.append(TestResult(
            name=f"RAG: {case['query'][:40]}",
            passed=passed,
            details=details,
            raw_output=output[:200]
        ))

    return results


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run all accuracy tests."""
    print("\n" + "=" * 70)
    print("  PROMPT ACCURACY TESTS")
    print("=" * 70)

    all_results = []

    # Run tests in parallel
    print("\nRunning tests...")

    keywords_results, orphan_results, summarize_results, rag_results = await asyncio.gather(
        test_keywords_extraction(),
        test_orphan_validation(),
        test_entity_summarization(),
        test_naive_rag_response(),
    )

    all_results.extend(keywords_results)
    all_results.extend(orphan_results)
    all_results.extend(summarize_results)
    all_results.extend(rag_results)

    # Print results
    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)

    passed = 0
    failed = 0

    for result in all_results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status}: {result.name}")
        print(f"       {result.details}")
        if not result.passed:
            print(f"       Output: {result.raw_output}...")

        if result.passed:
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed}/{passed + failed} tests passed")
    print("=" * 70)

    if failed > 0:
        print("\n⚠️  Some tests failed - review prompt changes")
        sys.exit(1)
    else:
        print("\n✓ All prompts producing correct outputs!")


if __name__ == "__main__":
    asyncio.run(main())
