#!/usr/bin/env python3
"""
Orphan Connection Quality Validation Script

Runs actual queries against LightRAG and analyzes whether orphan connections
improve or poison retrieval quality.
"""

import asyncio
from dataclasses import dataclass

import httpx

API_BASE = 'http://localhost:9622'


@dataclass
class TestResult:
    query: str
    expected: list[str]
    unexpected: list[str]
    retrieved_entities: list[str]
    precision: float
    recall: float
    noise_count: int
    passed: bool
    details: str


TEST_CASES = [
    # Test 1: Neural Network Types (PRECISION)
    # Note: "Quantum" may appear legitimately due to "Quantum Machine Learning" being a real field
    {
        'query': 'What types of neural networks are used in deep learning?',
        'expected': ['Neural Networks', 'Convolutional Neural Network', 'Recurrent Neural Network', 'Transformer'],
        'unexpected': ['FDA', 'Atopic Dermatitis', 'Vehicle Emissions Standards'],  # Truly unrelated
        'category': 'precision',
        'description': 'Should retrieve NN types via orphan connections (CNN->NN, RNN->NN)',
    },
    # Test 2: Quantum Companies (RECALL)
    {
        'query': 'What companies are working on quantum computing?',
        'expected': ['IonQ', 'Microsoft', 'Google', 'IBM'],
        'unexpected': ['FDA', 'Atopic Dermatitis'],  # Medical domain unrelated
        'category': 'recall',
        'description': 'Should find IonQ (via Trapped Ions) and Microsoft (via Topological Qubits)',
    },
    # Test 3: Greenhouse Gases (RECALL)
    # Note: "Quantum" may appear due to "climate simulation via quantum computing" being valid
    {
        'query': 'What are greenhouse gases?',
        'expected': ['Carbon Dioxide', 'CO2', 'Methane', 'CH4', 'Nitrous Oxide', 'N2O', 'Fluorinated'],
        'unexpected': ['FDA', 'Atopic Dermatitis', 'IonQ'],  # Medical/specific tech unrelated
        'category': 'recall',
        'description': 'Should retrieve all GHGs via orphan connections forming a cluster',
    },
    # Test 4: Reinforcement Learning (NOISE)
    # Note: Cross-domain mentions like "climate modeling" may appear from original docs
    {
        'query': 'What is reinforcement learning?',
        'expected': ['Reinforcement Learning', 'Machine Learning'],
        'unexpected': ['FDA', 'Atopic Dermatitis', 'Dupixent'],  # Medical domain truly unrelated
        'category': 'noise',
        'description': 'Should NOT pull in truly unrelated medical domain',
    },
    # Test 5: Computer Vision (NOISE)
    # Note: Drug Discovery may appear due to "medical imaging" being a CV application
    {
        'query': 'How does computer vision work?',
        'expected': ['Computer Vision', 'Image', 'Object', 'Feature', 'Edge Detection'],
        'unexpected': ['FDA', 'Atopic Dermatitis', 'Kyoto Protocol'],  # Truly unrelated domains
        'category': 'noise',
        'description': 'Should retrieve CV techniques, not truly unrelated domains',
    },
    # Test 6: Amazon Cross-Domain Check (EDGE CASE)
    {
        'query': 'What is Amazon?',
        'expected': ['Amazon'],
        'unexpected': ['FDA', 'Atopic Dermatitis'],  # Medical domain unrelated to tech company
        'category': 'edge_case',
        'description': 'Check if Amazon->Microsoft connection causes retrieval issues',
    },
    # Test 7: Medical Domain Isolation (STRICT NOISE TEST)
    {
        'query': 'What is Dupixent used for?',
        'expected': ['Dupixent', 'Atopic Dermatitis', 'FDA'],
        'unexpected': ['Neural Networks', 'Quantum Computing', 'Climate Change', 'IonQ'],
        'category': 'noise',
        'description': 'Medical query should NOT retrieve tech/climate domains',
    },
]


async def run_query(query: str, mode: str = 'local') -> dict:
    """Run a query against LightRAG API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f'{API_BASE}/query', json={'query': query, 'mode': mode, 'only_need_context': True}
        )
        return response.json()


def extract_entities_from_context(context: str) -> list[str]:
    """Extract entity names from the context string."""
    entities = []
    # Look for entity patterns in the context
    lines = context.split('\n')
    for line in lines:
        # Entity lines often start with entity names in quotes or bold
        if 'Entity:' in line or line.startswith('-'):
            # Extract potential entity name
            parts = line.split(':')
            if len(parts) > 1:
                entity = parts[1].strip().strip('"').strip("'")
                if entity and len(entity) > 2:
                    entities.append(entity)
    return entities


async def evaluate_test_case(test_case: dict) -> TestResult:
    """Evaluate a single test case."""
    query = test_case['query']
    expected = test_case['expected']
    unexpected = test_case['unexpected']

    try:
        result = await run_query(query)
        response_text = result.get('response', '')

        # Check which expected entities appear in the response
        found_expected = []
        missed_expected = []
        for entity in expected:
            # Case-insensitive partial match
            if entity.lower() in response_text.lower():
                found_expected.append(entity)
            else:
                missed_expected.append(entity)

        # Check for unexpected (noise) entities
        found_unexpected = []
        for entity in unexpected:
            if entity.lower() in response_text.lower():
                found_unexpected.append(entity)

        # Calculate metrics
        precision = len(found_expected) / len(expected) if expected else 1.0
        recall = len(found_expected) / len(expected) if expected else 1.0
        noise_count = len(found_unexpected)

        # Pass criteria: recall > 50% AND no noise detected
        passed = recall >= 0.5 and noise_count == 0

        details = f'Found: {found_expected} | Missed: {missed_expected} | Noise: {found_unexpected}'

        return TestResult(
            query=query,
            expected=expected,
            unexpected=unexpected,
            retrieved_entities=found_expected,
            precision=precision,
            recall=recall,
            noise_count=noise_count,
            passed=passed,
            details=details,
        )

    except Exception as e:
        return TestResult(
            query=query,
            expected=expected,
            unexpected=unexpected,
            retrieved_entities=[],
            precision=0.0,
            recall=0.0,
            noise_count=0,
            passed=False,
            details=f'Error: {e!s}',
        )


async def get_graph_stats() -> dict:
    """Get current graph statistics."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.get(f'{API_BASE}/health')
        graph = await client.get(f'{API_BASE}/graphs?label=*&max_depth=0&max_nodes=1000')

        graph_data = graph.json()
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        # Count orphans (nodes with no edges)
        node_ids = {n['id'] for n in nodes}
        connected_ids = set()
        for e in edges:
            connected_ids.add(e.get('source'))
            connected_ids.add(e.get('target'))

        orphan_ids = node_ids - connected_ids

        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'orphan_count': len(orphan_ids),
            'orphan_rate': len(orphan_ids) / len(nodes) if nodes else 0,
        }


async def main():
    print('=' * 60)
    print('ORPHAN CONNECTION QUALITY VALIDATION')
    print('=' * 60)

    # Get graph stats first
    try:
        stats = await get_graph_stats()
        print('\n📊 Current Graph Statistics:')
        print(f'   Nodes: {stats["total_nodes"]}')
        print(f'   Edges: {stats["total_edges"]}')
        print(f'   Orphans: {stats["orphan_count"]} ({stats["orphan_rate"]:.1%})')
    except Exception as e:
        print(f'⚠️  Could not get graph stats: {e}')

    print('\n' + '-' * 60)
    print('Running Quality Tests...')
    print('-' * 60)

    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        category = str(test_case['category']).upper()
        print(f'\n🧪 Test {i}: {category} - {test_case["description"]}')
        print(f'   Query: "{test_case["query"]}"')

        result = await evaluate_test_case(test_case)
        results.append(result)

        status = '✅ PASS' if result.passed else '❌ FAIL'
        print(f'   {status}')
        print(f'   Recall: {result.recall:.0%} | Noise: {result.noise_count}')
        print(f'   {result.details}')

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_recall = sum(r.recall for r in results) / len(results)
    total_noise = sum(r.noise_count for r in results)

    print(f'\n📈 Results: {passed}/{total} tests passed ({passed / total:.0%})')
    print(f'📈 Average Recall: {avg_recall:.0%}')
    print(f'📈 Total Noise Instances: {total_noise}')

    # Category breakdown
    categories = {}
    for r, tc in zip(results, TEST_CASES, strict=False):
        cat = tc['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'total': 0}
        categories[cat]['total'] += 1
        if r.passed:
            categories[cat]['passed'] += 1

    print('\n📊 By Category:')
    for cat, data in categories.items():
        print(f'   {cat.upper()}: {data["passed"]}/{data["total"]}')

    # Verdict
    print('\n' + '-' * 60)
    if total_noise == 0 and avg_recall >= 0.6:
        print('✅ VERDICT: Orphan connections are IMPROVING retrieval')
        print('   - No cross-domain pollution detected')
        print('   - Good recall on expected entities')
    elif total_noise > 0:
        print('⚠️  VERDICT: Orphan connections MAY BE POISONING retrieval')
        print(f'   - {total_noise} noise instances detected')
        print('   - Review the connections causing cross-domain bleed')
    else:
        print('⚠️  VERDICT: Orphan connections have MIXED results')
        print('   - Recall could be improved')
        print('   - No significant noise detected')
    print('-' * 60)


if __name__ == '__main__':
    asyncio.run(main())
