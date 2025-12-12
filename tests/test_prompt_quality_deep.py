"""
Deep Quality Analysis for Optimized Prompts.

Tests prompts on challenging, diverse inputs to identify weaknesses.
Run with: uv run --extra test python tests/test_prompt_quality_deep.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag.prompt import PROMPTS

# =============================================================================
# Test Data: Entity Extraction (5 Domains)
# =============================================================================

ENTITY_TEST_TEXTS = {
    'technical': {
        'name': 'Technical/API Documentation',
        'text': """
The FastAPI framework provides automatic OpenAPI documentation via Swagger UI at /docs.
To enable OAuth2 authentication, use the OAuth2PasswordBearer class from fastapi.security.
The dependency injection system allows you to declare dependencies using Depends().
Response models are validated using Pydantic's BaseModel class. For async database
operations, SQLAlchemy 2.0 with asyncpg driver is recommended. Rate limiting can be
implemented using slowapi middleware with Redis as the backend store.
""",
        'expected_entities': ['FastAPI', 'OpenAPI', 'Swagger UI', 'OAuth2', 'Pydantic', 'SQLAlchemy', 'Redis'],
        'expected_relations': ['FastAPI provides OpenAPI', 'OAuth2 for authentication', 'Pydantic validates responses'],
    },
    'legal': {
        'name': 'Legal/Contract',
        'text': """
WHEREAS, Acme Corporation ("Licensor") owns certain intellectual property rights in
the Software known as "DataSync Pro"; and WHEREAS, Beta Technologies Inc. ("Licensee")
desires to obtain a non-exclusive license to use said Software; NOW THEREFORE, in
consideration of the mutual covenants herein, the parties agree: Licensee shall pay
Licensor a royalty of 5% of gross revenues derived from Software usage. This Agreement
shall be governed by the laws of the State of Delaware. Any disputes shall be resolved
through binding arbitration administered by JAMS in San Francisco, California.
""",
        'expected_entities': [
            'Acme Corporation',
            'Beta Technologies Inc',
            'DataSync Pro',
            'Delaware',
            'JAMS',
            'San Francisco',
        ],
        'expected_relations': ['Licensor owns Software', 'Licensee pays royalty', 'Delaware governs agreement'],
    },
    'scientific': {
        'name': 'Scientific/Research Abstract',
        'text': """
We investigated the effects of CRISPR-Cas9 gene editing on tumor suppressor p53 expression
in HeLa cell lines. Using Western blot analysis and qPCR, we observed a 73% reduction in
p53 protein levels after 48 hours of transfection. Control groups treated with scrambled
sgRNA showed no significant change (p > 0.05). Our findings suggest that targeted p53
knockout can be achieved with high efficiency, supporting further research into cancer
immunotherapy applications. Funding was provided by NIH grant R01-CA123456.
""",
        'expected_entities': ['CRISPR-Cas9', 'p53', 'HeLa', 'Western blot', 'qPCR', 'NIH'],
        'expected_relations': ['CRISPR-Cas9 edits p53', 'Western blot measures protein', 'NIH funds research'],
    },
    'news': {
        'name': 'News/Current Events',
        'text': """
BREAKING: Tesla CEO Elon Musk announced today that the company will invest $5 billion
in a new Gigafactory in Austin, Texas, creating an estimated 10,000 jobs by 2025.
The announcement came during Tesla's Q3 earnings call, where the company reported
record revenue of $25.2 billion. Analysts at Goldman Sachs raised their price target
to $300, citing strong demand for the Model Y in European markets. Shares rose 8%
in after-hours trading on the NASDAQ exchange.
""",
        'expected_entities': [
            'Tesla',
            'Elon Musk',
            'Austin',
            'Texas',
            'Gigafactory',
            'Goldman Sachs',
            'Model Y',
            'NASDAQ',
        ],
        'expected_relations': ['Musk leads Tesla', 'Tesla invests in Gigafactory', 'Goldman Sachs analyzes Tesla'],
    },
    'conversational': {
        'name': 'Conversational/Interview',
        'text': """
Interviewer: Dr. Chen, your research on quantum computing has gained significant attention.
Can you explain the breakthrough?

Dr. Chen: Certainly. Our team at MIT developed a new error correction method using topological
qubits. Unlike traditional approaches by IBM or Google, we achieved 99.9% fidelity at room
temperature. My colleague Dr. Sarah Martinez deserves much of the credit - her algorithm
made this possible.

Interviewer: What are the practical applications?

Dr. Chen: Drug discovery is the immediate target. Pfizer has already licensed our technology
for protein folding simulations.
""",
        'expected_entities': ['Dr. Chen', 'MIT', 'IBM', 'Google', 'Dr. Sarah Martinez', 'Pfizer'],
        'expected_relations': ['Dr. Chen works at MIT', 'Martinez developed algorithm', 'Pfizer licenses technology'],
    },
    # New domains for expanded coverage
    'medical': {
        'name': 'Medical/Healthcare',
        'text': """
The patient presented with Type 2 diabetes mellitus complicated by diabetic retinopathy.
Treatment included metformin 500mg twice daily and monthly intravitreal injections of
Avastin (bevacizumab). Dr. Sarah Chen at Johns Hopkins recommended adding Jardiance
(empagliflozin) given the patient's elevated HbA1c of 8.2%. Insurance coverage through
Aetna required prior authorization.
""",
        'expected_entities': [
            'Type 2 diabetes',
            'diabetic retinopathy',
            'metformin',
            'Avastin',
            'Dr. Sarah Chen',
            'Johns Hopkins',
            'Jardiance',
            'Aetna',
        ],
        'expected_relations': ['metformin treats diabetes', 'Avastin treats retinopathy', 'Aetna covers insurance'],
    },
    'financial': {
        'name': 'Financial/Trading',
        'text': """
The S&P 500 index futures dropped 2.3% following hawkish Fed minutes. BlackRock's
iShares ETF (SPY) saw record outflows of $4.2 billion. Morgan Stanley upgraded
NVIDIA to Overweight with a $500 price target, citing AI datacenter demand.
Bitcoin fell below $40,000 as Coinbase reported a 30% decline in trading volume.
""",
        'expected_entities': ['S&P 500', 'BlackRock', 'SPY', 'Morgan Stanley', 'NVIDIA', 'Bitcoin', 'Coinbase'],
        'expected_relations': ['BlackRock manages SPY', 'Morgan Stanley analyzes NVIDIA', 'Coinbase trades Bitcoin'],
    },
    'social_media': {
        'name': 'Social Media/Informal',
        'text': """
OMG just saw @elonmusk tweet about #Dogecoin again! Price pumped 15% in an hour!
Meanwhile TikTok is banning crypto content and YouTube demonetized half the crypto
channels. My Discord server is going crazy. Even r/wallstreetbets is talking about it.
""",
        'expected_entities': ['Elon Musk', 'Dogecoin', 'TikTok', 'YouTube', 'Discord', 'r/wallstreetbets'],
        'expected_relations': ['Musk tweets about Dogecoin', 'TikTok bans crypto', 'YouTube demonetizes channels'],
    },
}


# =============================================================================
# Test Data: Keywords Extraction (Query Types)
# =============================================================================

KEYWORD_TEST_QUERIES = {
    'factual': {
        'query': 'What is the capital of France?',
        'expected_high': ['geography', 'capital city'],
        'expected_low': ['France', 'capital'],
    },
    'analytical': {
        'query': 'Why does inflation cause interest rates to rise?',
        'expected_high': ['economics', 'monetary policy', 'cause-effect'],
        'expected_low': ['inflation', 'interest rates'],
    },
    'comparison': {
        'query': 'How does Python compare to JavaScript for web development?',
        'expected_high': ['programming languages', 'comparison', 'web development'],
        'expected_low': ['Python', 'JavaScript'],
    },
    'procedural': {
        'query': 'How to deploy a Docker container to AWS ECS?',
        'expected_high': ['deployment', 'cloud computing', 'containerization'],
        'expected_low': ['Docker', 'AWS', 'ECS', 'container'],
    },
    'multi_topic': {
        'query': 'What is machine learning and how does it relate to artificial intelligence?',
        'expected_high': ['machine learning', 'artificial intelligence', 'technology'],
        'expected_low': ['machine learning', 'artificial intelligence'],
    },
    # New query types for expanded coverage
    'negation': {
        'query': 'What programming languages are NOT object-oriented?',
        'expected_high': ['programming languages', 'paradigms'],
        'expected_low': ['object-oriented', 'programming'],
    },
    'implicit': {
        'query': 'Tell me about climate',
        'expected_high': ['climate', 'environment'],
        'expected_low': ['climate'],
    },
    'multi_hop': {
        'query': 'Who is the CEO of the company that makes the iPhone?',
        'expected_high': ['business', 'leadership', 'technology'],
        'expected_low': ['CEO', 'iPhone'],
    },
    'ambiguous': {
        'query': 'What is Java used for?',
        'expected_high': ['technology', 'programming'],
        'expected_low': ['Java'],
    },
}


# =============================================================================
# Test Data: Orphan Validation (Edge Cases)
# =============================================================================

ORPHAN_TEST_CASES = [
    # Same domain, directly related
    {
        'orphan': {'name': 'Python', 'type': 'concept', 'desc': 'Programming language known for simplicity'},
        'candidate': {'name': 'Django', 'type': 'concept', 'desc': 'Web framework written in Python'},
        'expected': True,
        'difficulty': 'easy',
        'category': 'same_domain_direct',
    },
    # Same domain, tangentially related
    {
        'orphan': {'name': 'Bitcoin', 'type': 'concept', 'desc': 'Decentralized cryptocurrency'},
        'candidate': {'name': 'Visa', 'type': 'organization', 'desc': 'Payment processing company'},
        'expected': True,  # Both are payment/financial
        'difficulty': 'medium',
        'category': 'same_domain_tangential',
    },
    # Same domain, unrelated
    {
        'orphan': {
            'name': 'Photosynthesis',
            'type': 'concept',
            'desc': 'Process plants use to convert sunlight to energy',
        },
        'candidate': {'name': 'Mitosis', 'type': 'concept', 'desc': 'Cell division process'},
        'expected': False,  # Both biology but not directly related
        'difficulty': 'medium',
        'category': 'same_domain_unrelated',
    },
    # Different domains, surprisingly related
    {
        'orphan': {'name': 'Netflix', 'type': 'organization', 'desc': 'Streaming entertainment company'},
        'candidate': {'name': 'AWS', 'type': 'organization', 'desc': "Amazon's cloud computing platform"},
        'expected': True,  # Netflix runs on AWS
        'difficulty': 'hard',
        'category': 'cross_domain_related',
    },
    # False positive trap (high similarity, no logic)
    {
        'orphan': {'name': 'Java', 'type': 'concept', 'desc': 'Programming language developed by Sun Microsystems'},
        'candidate': {'name': 'Java', 'type': 'location', 'desc': 'Indonesian island known for coffee production'},
        'expected': False,  # Same name, completely different things
        'difficulty': 'hard',
        'category': 'false_positive_trap',
    },
    # Temporal relationship
    {
        'orphan': {'name': 'Windows 11', 'type': 'product', 'desc': 'Microsoft operating system released in 2021'},
        'candidate': {'name': 'Windows 10', 'type': 'product', 'desc': 'Microsoft operating system released in 2015'},
        'expected': True,  # Successor relationship
        'difficulty': 'easy',
        'category': 'temporal',
    },
    # Causal relationship
    {
        'orphan': {'name': 'COVID-19 Pandemic', 'type': 'event', 'desc': 'Global health crisis starting in 2020'},
        'candidate': {
            'name': 'Remote Work',
            'type': 'concept',
            'desc': 'Working from home or outside traditional office',
        },
        'expected': True,  # Pandemic caused remote work surge
        'difficulty': 'medium',
        'category': 'causal',
    },
    # Hierarchical
    {
        'orphan': {'name': 'Toyota Camry', 'type': 'product', 'desc': 'Mid-size sedan manufactured by Toyota'},
        'candidate': {'name': 'Toyota', 'type': 'organization', 'desc': 'Japanese automotive manufacturer'},
        'expected': True,  # Part-of relationship
        'difficulty': 'easy',
        'category': 'hierarchical',
    },
    # Competitor relationship
    {
        'orphan': {'name': 'Uber', 'type': 'organization', 'desc': 'Ride-sharing company'},
        'candidate': {'name': 'Lyft', 'type': 'organization', 'desc': 'Ride-sharing company'},
        'expected': True,  # Direct competitors
        'difficulty': 'easy',
        'category': 'competitor',
    },
    # No relationship
    {
        'orphan': {'name': 'Beethoven', 'type': 'person', 'desc': 'Classical music composer from 18th century'},
        'candidate': {'name': 'Kubernetes', 'type': 'concept', 'desc': 'Container orchestration platform'},
        'expected': False,  # Completely unrelated
        'difficulty': 'easy',
        'category': 'no_relationship',
    },
    # New edge cases for expanded coverage
    # Subsidiary relationship (ownership)
    {
        'orphan': {'name': 'YouTube', 'type': 'organization', 'desc': 'Video sharing platform'},
        'candidate': {'name': 'Google', 'type': 'organization', 'desc': 'Technology company and search engine'},
        'expected': True,  # Google owns YouTube
        'difficulty': 'medium',
        'category': 'subsidiary',
    },
    # Scientific alias (same thing, different names)
    {
        'orphan': {'name': 'SARS-CoV-2', 'type': 'concept', 'desc': 'Coronavirus that causes COVID-19'},
        'candidate': {
            'name': 'Coronavirus',
            'type': 'concept',
            'desc': 'Family of viruses affecting respiratory system',
        },
        'expected': True,  # SARS-CoV-2 is a type of coronavirus
        'difficulty': 'medium',
        'category': 'scientific_alias',
    },
    # Pseudonym (same person, different names)
    {
        'orphan': {
            'name': 'Mark Twain',
            'type': 'person',
            'desc': 'American author of Tom Sawyer and Huckleberry Finn',
        },
        'candidate': {'name': 'Samuel Clemens', 'type': 'person', 'desc': 'American writer born in Missouri in 1835'},
        'expected': True,  # Same person (pen name)
        'difficulty': 'hard',
        'category': 'pseudonym',
    },
    # Weak false positive (similar names/categories but unrelated)
    {
        'orphan': {'name': 'Mount Everest', 'type': 'location', 'desc': 'Highest mountain in the Himalayas'},
        'candidate': {
            'name': 'Mount Rushmore',
            'type': 'location',
            'desc': 'Memorial carved into mountain in South Dakota',
        },
        'expected': False,  # Both "Mount" but no logical connection
        'difficulty': 'medium',
        'category': 'weak_false_positive',
    },
    # Spurious correlation (statistically correlated but no causal link)
    {
        'orphan': {'name': 'Ice Cream Sales', 'type': 'concept', 'desc': 'Consumer purchases of frozen dessert'},
        'candidate': {'name': 'Drowning Deaths', 'type': 'concept', 'desc': 'Fatalities from submersion in water'},
        'expected': False,  # Both increase in summer but no causal relationship
        'difficulty': 'hard',
        'category': 'spurious_correlation',
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


async def call_llm(prompt: str, model: str = 'gpt-4o-mini') -> str:
    """Call OpenAI API."""
    import openai

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def format_entity_prompt(text: str) -> str:
    """Format entity extraction prompt."""
    examples = '\n'.join(PROMPTS['entity_extraction_examples'])
    tuple_del = PROMPTS['DEFAULT_TUPLE_DELIMITER']
    comp_del = PROMPTS['DEFAULT_COMPLETION_DELIMITER']

    examples = examples.format(
        tuple_delimiter=tuple_del,
        completion_delimiter=comp_del,
    )

    return (
        PROMPTS['entity_extraction_system_prompt'].format(
            tuple_delimiter=tuple_del,
            completion_delimiter=comp_del,
            entity_types='person, organization, location, concept, product, event, category, method',
            language='English',
            examples=examples,
            input_text=text,
        )
        + '\n'
        + PROMPTS['entity_extraction_user_prompt'].format(
            completion_delimiter=comp_del,
            language='English',
        )
    )


def parse_entities(output: str) -> tuple[list[dict], list[dict]]:
    """Parse entity extraction output."""
    entities = []
    relations = []
    tuple_del = '<|#|>'

    for line in output.strip().split('\n'):
        line = line.strip()
        if not line or '<|COMPLETE|>' in line:
            continue

        parts = line.split(tuple_del)
        if len(parts) >= 4 and parts[0].lower() == 'entity':
            entities.append(
                {
                    'name': parts[1].strip(),
                    'type': parts[2].strip(),
                    'desc': parts[3].strip() if len(parts) > 3 else '',
                }
            )
        elif len(parts) >= 5 and parts[0].lower() == 'relation':
            relations.append(
                {
                    'source': parts[1].strip(),
                    'target': parts[2].strip(),
                    'keywords': parts[3].strip(),
                }
            )

    return entities, relations


@dataclass
class EntityResult:
    domain: str
    entities: list[dict]
    relations: list[dict]
    expected_entities: list[str]
    issues: list[str] = field(default_factory=list)
    precision_notes: str = ''
    recall_notes: str = ''


@dataclass
class KeywordResult:
    query_type: str
    query: str
    high_keywords: list[str]
    low_keywords: list[str]
    issues: list[str] = field(default_factory=list)


@dataclass
class OrphanResult:
    category: str
    orphan: str
    candidate: str
    expected: bool
    actual: bool
    confidence: float
    correct: bool
    reasoning: str = ''


# =============================================================================
# Test Functions
# =============================================================================


async def test_entity_extraction_deep() -> list[EntityResult]:
    """Deep test entity extraction on 5 domains."""
    results = []

    for _domain, data in ENTITY_TEST_TEXTS.items():
        print(f'  Testing {data["name"]}...')

        prompt = format_entity_prompt(data['text'])
        output = await call_llm(prompt)
        entities, relations = parse_entities(output)

        result = EntityResult(
            domain=data['name'],
            entities=entities,
            relations=relations,
            expected_entities=data['expected_entities'],
        )

        # Analyze precision (are extracted entities important?)
        entity_names = [e['name'].lower() for e in entities]
        found = [exp for exp in data['expected_entities'] if any(exp.lower() in n for n in entity_names)]
        missed = [exp for exp in data['expected_entities'] if not any(exp.lower() in n for n in entity_names)]

        if missed:
            result.issues.append(f'RECALL: Missed expected entities: {missed}')
            result.recall_notes = f'Found {len(found)}/{len(data["expected_entities"])}'
        else:
            result.recall_notes = f'Found all {len(found)} expected'

        # Check for generic/unhelpful entities
        generic_entities = [e for e in entities if len(e['desc']) < 20]
        if generic_entities:
            result.issues.append(f'QUALITY: {len(generic_entities)} entities have very short descriptions')

        # Check relationship density
        if len(entities) > 0:
            ratio = len(relations) / len(entities)
            if ratio < 0.5:
                result.issues.append(f'CONNECTIVITY: Low relation/entity ratio ({ratio:.2f})')

        results.append(result)

    return results


async def test_keywords_extraction_deep() -> list[KeywordResult]:
    """Deep test keywords extraction on varied query types."""
    results = []
    examples = '\n'.join(PROMPTS['keywords_extraction_examples'])

    for query_type, data in KEYWORD_TEST_QUERIES.items():
        print(f'  Testing {query_type} query...')

        prompt = PROMPTS['keywords_extraction'].format(
            examples=examples,
            query=data['query'],
        )

        output = await call_llm(prompt)

        try:
            clean = output.strip()
            if clean.startswith('```'):
                clean = clean.split('```')[1].replace('json', '').strip()
            parsed = json.loads(clean)

            result = KeywordResult(
                query_type=query_type,
                query=data['query'],
                high_keywords=parsed.get('high_level_keywords', []),
                low_keywords=parsed.get('low_level_keywords', []),
            )

            # Check if key concepts are captured
            all_kw = ' '.join(result.high_keywords + result.low_keywords).lower()

            for exp in data['expected_low']:
                if exp.lower() not in all_kw:
                    result.issues.append(f"MISS: Expected '{exp}' not in keywords")

            # Check for reasonable count
            if len(result.high_keywords) == 0:
                result.issues.append('EMPTY: No high-level keywords')
            if len(result.low_keywords) == 0 and query_type != 'factual':
                result.issues.append('EMPTY: No low-level keywords')

        except json.JSONDecodeError:
            result = KeywordResult(
                query_type=query_type,
                query=data['query'],
                high_keywords=[],
                low_keywords=[],
                issues=['PARSE ERROR: Invalid JSON output'],
            )

        results.append(result)

    return results


async def test_orphan_validation_deep() -> list[OrphanResult]:
    """Deep test orphan validation on edge cases."""
    results = []

    for case in ORPHAN_TEST_CASES:
        print(f'  Testing {case["category"]}: {case["orphan"]["name"]} ↔ {case["candidate"]["name"]}...')

        prompt = PROMPTS['orphan_connection_validation'].format(
            orphan_name=case['orphan']['name'],
            orphan_type=case['orphan']['type'],
            orphan_description=case['orphan']['desc'],
            candidate_name=case['candidate']['name'],
            candidate_type=case['candidate']['type'],
            candidate_description=case['candidate']['desc'],
            similarity_score=0.75,
        )

        output = await call_llm(prompt)

        try:
            clean = output.strip()
            if clean.startswith('```'):
                clean = clean.split('```')[1].replace('json', '').strip()
            parsed = json.loads(clean)

            actual = parsed.get('should_connect', False)
            # Handle discrete labels (HIGH/MEDIUM/LOW/NONE) or numeric
            raw_confidence = parsed.get('confidence', 0.0)
            if isinstance(raw_confidence, str):
                confidence_map = {'HIGH': 0.95, 'MEDIUM': 0.85, 'LOW': 0.60, 'NONE': 0.20}
                confidence = confidence_map.get(raw_confidence.upper(), 0.75)
            else:
                confidence = raw_confidence
            reasoning = parsed.get('reasoning', '')

            results.append(
                OrphanResult(
                    category=case['category'],
                    orphan=case['orphan']['name'],
                    candidate=case['candidate']['name'],
                    expected=case['expected'],
                    actual=actual,
                    confidence=confidence,
                    correct=(actual == case['expected']),
                    reasoning=reasoning,
                )
            )

        except json.JSONDecodeError:
            results.append(
                OrphanResult(
                    category=case['category'],
                    orphan=case['orphan']['name'],
                    candidate=case['candidate']['name'],
                    expected=case['expected'],
                    actual=False,
                    confidence=0.0,
                    correct=False,
                    reasoning='PARSE ERROR',
                )
            )

    return results


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    """Run deep quality analysis."""
    print('\n' + '=' * 70)
    print('  DEEP PROMPT QUALITY ANALYSIS')
    print('=' * 70)

    # Entity Extraction
    print(f'\n📊 ENTITY EXTRACTION ({len(ENTITY_TEST_TEXTS)} domains)')
    print('-' * 50)
    entity_results = await test_entity_extraction_deep()

    for r in entity_results:
        status = '✓' if not r.issues else '⚠'
        print(f'\n{status} {r.domain}')
        print(f'   Entities: {len(r.entities)} | Relations: {len(r.relations)}')
        print(f'   Recall: {r.recall_notes}')
        if r.issues:
            for issue in r.issues:
                print(f'   ❌ {issue}')

        # Show extracted entities
        print(f'   Extracted: {[e["name"] for e in r.entities[:8]]}{"..." if len(r.entities) > 8 else ""}')

    # Keywords Extraction
    print(f'\n\n📊 KEYWORDS EXTRACTION ({len(KEYWORD_TEST_QUERIES)} query types)')
    print('-' * 50)
    keyword_results = await test_keywords_extraction_deep()

    for r in keyword_results:
        status = '✓' if not r.issues else '⚠'
        print(f'\n{status} {r.query_type}: "{r.query[:50]}..."')
        print(f'   High: {r.high_keywords}')
        print(f'   Low: {r.low_keywords}')
        if r.issues:
            for issue in r.issues:
                print(f'   ❌ {issue}')

    # Orphan Validation
    print(f'\n\n📊 ORPHAN VALIDATION ({len(ORPHAN_TEST_CASES)} edge cases)')
    print('-' * 50)
    orphan_results = await test_orphan_validation_deep()

    correct = sum(1 for r in orphan_results if r.correct)
    print(f'\nAccuracy: {correct}/{len(orphan_results)} ({100 * correct / len(orphan_results):.0f}%)')

    for r in orphan_results:
        status = '✓' if r.correct else '✗'
        decision = 'CONNECT' if r.actual else 'REJECT'
        expected = 'CONNECT' if r.expected else 'REJECT'
        print(f'\n{status} [{r.category}] {r.orphan} ↔ {r.candidate}')
        print(f'   Decision: {decision} (conf={r.confidence:.2f}) | Expected: {expected}')
        if not r.correct:
            print(f'   ❌ WRONG: {r.reasoning[:80]}...')

    # Summary
    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)

    entity_issues = sum(len(r.issues) for r in entity_results)
    keyword_issues = sum(len(r.issues) for r in keyword_results)
    orphan_accuracy = 100 * correct / len(orphan_results)

    print(f'\nEntity Extraction: {entity_issues} issues across {len(entity_results)} domains')
    print(f'Keywords Extraction: {keyword_issues} issues across {len(keyword_results)} query types')
    print(f'Orphan Validation: {orphan_accuracy:.0f}% accuracy ({correct}/{len(orphan_results)})')

    # Recommendations
    print('\n📋 RECOMMENDATIONS:')
    if entity_issues > 3:
        print('   • Entity extraction needs improvement (recall or description quality)')
    if keyword_issues > 2:
        print('   • Keywords extraction needs clearer guidance')
    if orphan_accuracy < 80:
        print('   • Orphan validation needs better examples or criteria')
    if entity_issues <= 3 and keyword_issues <= 2 and orphan_accuracy >= 80:
        print('   • All prompts performing well! Minor tuning may help but not critical.')


if __name__ == '__main__':
    asyncio.run(main())
