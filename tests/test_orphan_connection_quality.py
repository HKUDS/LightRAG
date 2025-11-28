"""
Orphan Connection Quality Tests

Tests to validate that orphan connections improve (not poison) retrieval quality.

Test Categories:
1. Precision tests - Do orphan connections add relevant context?
2. Recall tests - Do orphan connections help find information that was missed?
3. Noise tests - Do orphan connections introduce irrelevant information?
4. A/B comparison - Same queries with/without connections
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryTestCase:
    """A test case for evaluating retrieval quality."""
    query: str
    expected_entities: list[str]  # Entities that SHOULD be retrieved
    unexpected_entities: list[str]  # Entities that should NOT be retrieved
    description: str
    category: str  # "precision", "recall", "noise"


# Test cases designed to evaluate orphan connection quality
TEST_CASES = [
    # PRECISION TESTS - Do we retrieve the RIGHT things?
    QueryTestCase(
        query="What types of neural networks are used in deep learning?",
        expected_entities=["Neural Networks", "Convolutional Neural Network",
                          "Recurrent Neural Network", "Transformer"],
        unexpected_entities=["Quantum Computing", "Climate Change", "FDA"],
        description="Should retrieve NN types via orphan connections (CNN->NN, RNN->NN)",
        category="precision"
    ),
    QueryTestCase(
        query="What quantum computing hardware approaches exist?",
        expected_entities=["Qubit", "Trapped Ions", "Superconducting Qubits",
                          "Photonic Qubits", "Topological Qubits", "IonQ"],
        unexpected_entities=["Neural Networks", "Machine Learning", "Climate Change"],
        description="Should retrieve qubit types via orphan connections",
        category="precision"
    ),

    # RECALL TESTS - Do we find things we would have MISSED without connections?
    QueryTestCase(
        query="What companies are working on quantum computing?",
        expected_entities=["IonQ", "Microsoft", "Google", "IBM"],
        unexpected_entities=[],
        description="Should find IonQ (connected via Trapped Ions) and Microsoft (via Topological Qubits)",
        category="recall"
    ),
    QueryTestCase(
        query="What are greenhouse gases?",
        expected_entities=["Carbon Dioxide (CO2)", "Methane (CH4)", "Nitrous Oxide (N2O)",
                          "Fluorinated Gases"],
        unexpected_entities=["Machine Learning", "Quantum Computing"],
        description="Should retrieve all GHGs via orphan connections forming a cluster",
        category="recall"
    ),

    # NOISE TESTS - Do we retrieve IRRELEVANT things?
    QueryTestCase(
        query="What is reinforcement learning?",
        expected_entities=["Reinforcement Learning", "Machine Learning"],
        unexpected_entities=["Climate Change", "FDA", "Vehicle Emissions Standards"],
        description="Should NOT pull in unrelated domains despite graph connectivity",
        category="noise"
    ),
    QueryTestCase(
        query="How does computer vision work?",
        expected_entities=["Computer Vision", "Image Segmentation", "Object Tracking",
                          "Feature Extraction", "Edge Detection"],
        unexpected_entities=["Quantum Computing", "Climate Modeling", "Drug Discovery"],
        description="Should retrieve CV techniques, not unrelated domains",
        category="noise"
    ),

    # EDGE CASE - Orphan connections shouldn't create nonsense pathways
    QueryTestCase(
        query="What is Amazon?",
        expected_entities=["Amazon"],
        unexpected_entities=[],  # We connected Amazon -> Microsoft, is this causing issues?
        description="Amazon query - check if connection to Microsoft causes retrieval issues",
        category="noise"
    ),
]


async def run_query(rag, query: str, mode: str = "local") -> dict:
    """Run a query and return retrieved entities."""
    # This would need to be adapted based on how LightRAG returns context
    result = await rag.aquery(query, param={"mode": mode})
    return result


async def evaluate_test_case(rag, test_case: QueryTestCase) -> dict:
    """Evaluate a single test case."""
    result = await run_query(rag, test_case.query)

    # Extract retrieved entities from result
    # (Implementation depends on LightRAG response format)
    retrieved_entities = []  # Parse from result

    # Calculate metrics
    expected_found = [e for e in test_case.expected_entities if e in retrieved_entities]
    unexpected_found = [e for e in test_case.unexpected_entities if e in retrieved_entities]

    precision = len(expected_found) / len(retrieved_entities) if retrieved_entities else 0
    recall = len(expected_found) / len(test_case.expected_entities) if test_case.expected_entities else 1
    noise_rate = len(unexpected_found) / len(retrieved_entities) if retrieved_entities else 0

    return {
        "test_case": test_case.description,
        "category": test_case.category,
        "query": test_case.query,
        "expected_found": expected_found,
        "expected_missed": [e for e in test_case.expected_entities if e not in retrieved_entities],
        "unexpected_found": unexpected_found,
        "precision": precision,
        "recall": recall,
        "noise_rate": noise_rate,
        "pass": len(unexpected_found) == 0 and recall > 0.5
    }


async def run_ab_comparison(rag_with_connections, rag_without_connections, query: str) -> dict:
    """
    Compare retrieval results with and without orphan connections.

    This requires two separate LightRAG instances:
    - One with orphan connections applied
    - One without (baseline)
    """
    result_with = await run_query(rag_with_connections, query)
    result_without = await run_query(rag_without_connections, query)

    return {
        "query": query,
        "with_connections": result_with,
        "without_connections": result_without,
        "improved": None,  # Human evaluation needed
    }


def generate_test_report(results: list[dict]) -> str:
    """Generate a test report from evaluation results."""
    report = ["# Orphan Connection Quality Test Report\n"]

    # Summary by category
    for category in ["precision", "recall", "noise"]:
        cat_results = [r for r in results if r["category"] == category]
        if cat_results:
            passed = sum(1 for r in cat_results if r["pass"])
            report.append(f"\n## {category.upper()} Tests: {passed}/{len(cat_results)} passed\n")
            for r in cat_results:
                status = "✅" if r["pass"] else "❌"
                report.append(f"- {status} {r['test_case']}")
                if r.get("unexpected_found"):
                    report.append(f"  - ⚠️ Noise detected: {r['unexpected_found']}")

    # Overall metrics
    all_precision = [r["precision"] for r in results if r["precision"] is not None]
    all_recall = [r["recall"] for r in results if r["recall"] is not None]
    all_noise = [r["noise_rate"] for r in results if r["noise_rate"] is not None]

    report.append(f"\n## Overall Metrics")
    report.append(f"- Average Precision: {sum(all_precision)/len(all_precision):.2f}")
    report.append(f"- Average Recall: {sum(all_recall)/len(all_recall):.2f}")
    report.append(f"- Average Noise Rate: {sum(all_noise)/len(all_noise):.2f}")

    return "\n".join(report)


# Manual evaluation checklist
EVALUATION_CHECKLIST = """
## Manual Evaluation Checklist

For each orphan connection, evaluate:

1. **Semantic Validity** (Is the connection logically correct?)
   - [ ] The entities are genuinely related
   - [ ] The relationship type makes sense
   - [ ] A human expert would agree with this connection

2. **Retrieval Impact** (Does this help or hurt queries?)
   - [ ] Queries about entity A now appropriately include entity B
   - [ ] Queries about entity B now appropriately include entity A
   - [ ] No unrelated queries are polluted by this connection

3. **Specificity** (Is the connection too broad?)
   - [ ] The connection is specific enough to be useful
   - [ ] Not just "both are technology" or "both are nouns"
   - [ ] The relationship description is meaningful

4. **Directionality** (Does the relationship make sense both ways?)
   - [ ] Query for A -> retrieves B makes sense
   - [ ] Query for B -> retrieves A makes sense

## Red Flags to Watch For:
- Connections between entirely different domains (e.g., Climate -> Quantum)
- Very low similarity scores with high confidence (LLM hallucination?)
- Hub entities getting too many connections (becoming noise magnets)
- Circular clusters forming (A->B->C->A with no external connections)
"""


if __name__ == "__main__":
    print("Orphan Connection Quality Test Framework")
    print("=" * 50)
    print(f"Total test cases: {len(TEST_CASES)}")
    print(f"- Precision tests: {len([t for t in TEST_CASES if t.category == 'precision'])}")
    print(f"- Recall tests: {len([t for t in TEST_CASES if t.category == 'recall'])}")
    print(f"- Noise tests: {len([t for t in TEST_CASES if t.category == 'noise'])}")
    print("\nRun with a LightRAG instance to execute tests.")
    print(EVALUATION_CHECKLIST)
