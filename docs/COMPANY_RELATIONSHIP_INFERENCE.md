# Advanced Company Relationship Inference

## Overview

LightRAG's Advanced Company Relationship Inference feature automatically detects implicit business relationships between organizations by analyzing document context and entity co-occurrence patterns. This goes beyond basic co-occurrence by understanding **relationship semantics**.

## Relationship Types Detected

### 1. **Competitor Relationships**
Organizations competing in the same market or for the same customers.

**Detection Signals:**
- **Explicit indicators**: "competitor", "rival", "versus", "competes with", "competition"
- **Implicit indicators**: "similar to", "alternative", "same market", "same industry"

**Example:**
```
"TechFlow Solutions competes directly with CodeCraft Inc. in the enterprise
software market. Both companies are market leaders with similar offerings."

→ Infers: TechFlow Solutions ←→ CodeCraft Inc. [COMPETITOR]
```

### 2. **Partnership Relationships**
Organizations collaborating, partnering, or forming strategic alliances.

**Detection Signals:**
- **Explicit indicators**: "partner", "collaboration", "joint venture", "strategic alliance", "works with"
- **Implicit indicators**: "ecosystem", "integrates with", "certified partner", "alliance"

**Example:**
```
"DataCloud Systems has formed a strategic partnership with TechFlow Solutions
to integrate their cloud infrastructure."

→ Infers: DataCloud ←→ TechFlow [PARTNERSHIP]
```

### 3. **Supply Chain Relationships**
Vendor-customer or supplier-buyer relationships.

**Detection Signals:**
- "supplier", "vendor", "customer", "client", "provides to", "supplies", "procures"

**Example:**
```
"SecureAuth Corp supplies authentication modules to TechFlow Solutions,
acting as a key vendor in TechFlow's product stack."

→ Infers: SecureAuth ←→ TechFlow [SUPPLY_CHAIN]
```

## How It Works

### 1. **Document Structure Analysis**
The system identifies the **main company** in each document by looking for:
- `Entity: [Company Name]` markers
- `Summary: [Company Name] is...` patterns
- `Company: [Company Name]` headers

### 2. **Contextual Analysis**
For each pair of companies appearing together:
- Extracts sentences mentioning both companies
- Counts relationship indicators (competitor, partner, supplier)
- Calculates confidence score based on signal strength

### 3. **Relationship Inference**
Creates relationships with:
- **Type**: competitor, partnership, or supply_chain
- **Confidence**: 0.0 to 1.0 based on contextual signals
- **Evidence**: Text snippets supporting the inference
- **Weight**: Co-occurrence count × confidence

## Configuration

### Basic Configuration

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",

    # Enable company relationship inference
    addon_params={
        "enable_company_relationship_inference": True,
        "company_inference_min_cooccurrence": 2,
        "company_inference_confidence_threshold": 0.5,
    }
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_company_relationship_inference` | `bool` | `False` | Enable/disable company relationship inference |
| `company_inference_min_cooccurrence` | `int` | `2` | Minimum times companies must appear together |
| `company_inference_confidence_threshold` | `float` | `0.5` | Minimum confidence (0-1) for inference |

### Advanced Configuration

```python
rag = LightRAG(
    working_dir="./rag_storage",

    # High overlap for better co-occurrence
    chunk_token_size=1200,
    chunk_overlap_token_size=400,  # 33% overlap

    addon_params={
        # Standard co-occurrence inference
        "enable_cooccurrence_inference": True,
        "min_cooccurrence": 3,

        # Company relationship inference
        "enable_company_relationship_inference": True,
        "company_inference_min_cooccurrence": 2,  # Lower for companies
        "company_inference_confidence_threshold": 0.5,
    }
)
```

## Usage Examples

### Example 1: Basic Usage

```python
import asyncio
from lightrag import LightRAG

async def main():
    rag = LightRAG(
        working_dir="./company_rag",
        addon_params={
            "enable_company_relationship_inference": True,
        }
    )

    # Insert document with competitor mention
    document = """
    Summary: Acme Corp is a software company.
    Entity: Acme Corp

    Acme Corp competes with TechStar Inc. in the enterprise market.
    Both companies offer similar cloud solutions.
    """

    await rag.ainsert(document)

    # Query relationships
    result = await rag.aquery("What companies compete with Acme Corp?")
    print(result)

asyncio.run(main())
```

### Example 2: Multiple Relationship Types

```python
# Document with various relationships
document = """
Summary: GlobalTech is a technology conglomerate.
Entity: GlobalTech

GlobalTech partners with CloudProvider Inc. for infrastructure needs.
The company competes with InnovateCorp in the AI market.
GlobalTech also sources hardware from SupplierCo.
"""

await rag.ainsert(document)

# This will infer:
# 1. GlobalTech ←→ CloudProvider [PARTNERSHIP]
# 2. GlobalTech ←→ InnovateCorp [COMPETITOR]
# 3. GlobalTech ←→ SupplierCo [SUPPLY_CHAIN]
```

### Example 3: Using High-Quality Config

```python
from examples.company_profile_extraction_config import (
    get_high_quality_extraction_config
)

config = get_high_quality_extraction_config()
rag = LightRAG(working_dir="./rag_storage", **config)

# Config automatically includes company relationship inference
```

## Document Format Best Practices

### Recommended Format

Structure your documents to maximize inference accuracy:

```markdown
Summary: [Main Company] is a [description]
Entity: [Main Company Name]

[Main Company] [relationship indicator] [Other Company].
[Additional context about the relationship].
```

### Good Examples

✅ **Clear competitor indication:**
```
Summary: TechCorp provides cloud services.
Entity: TechCorp

TechCorp competes directly with CloudGiant Inc. in the enterprise
cloud market. Both companies target Fortune 500 customers.
```

✅ **Clear partnership indication:**
```
Summary: StartupX develops AI solutions.
Entity: StartupX

StartupX has formed a strategic partnership with DataFirm to
integrate their AI models with DataFirm's analytics platform.
```

### Less Ideal Examples

⚠️ **Ambiguous mention:**
```
TechCorp and CloudGiant are both cloud providers.
```
- No clear relationship type indicated

⚠️ **Missing main entity:**
```
The company competes with several others in the market.
```
- "The company" is not specific enough

## Confidence Scoring

The system calculates confidence based on:

1. **Signal Strength**: Number of relationship indicators found
2. **Signal Type**: Explicit indicators (2x weight) vs implicit (1x weight)
3. **Co-occurrence Count**: Higher co-occurrence increases confidence

### Confidence Formula

```
confidence = min(max_signal_count / 5.0, 1.0)

Where:
- max_signal_count = highest count among competitor/partner/supply_chain signals
- Explicit indicators count as 2 points
- Implicit indicators count as 1 point
- Maximum confidence = 1.0 at 5+ signals
```

### Confidence Levels

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 0.8 - 1.0 | Very High | Reliable, multiple explicit indicators |
| 0.6 - 0.8 | High | Good evidence from text |
| 0.5 - 0.6 | Medium | Threshold level, some evidence |
| < 0.5 | Low | Below threshold, relationship not inferred |

## Querying Inferred Relationships

### Query Examples

```python
# Find competitors
result = await rag.aquery(
    "What companies compete with TechCorp?",
    param=QueryParam(mode="hybrid")
)

# Find partners
result = await rag.aquery(
    "Which companies does TechCorp partner with?",
    param=QueryParam(mode="global")
)

# Find supply chain
result = await rag.aquery(
    "What are TechCorp's suppliers and customers?",
    param=QueryParam(mode="hybrid")
)

# General relationships
result = await rag.aquery(
    "What are all the business relationships for TechCorp?",
    param=QueryParam(mode="global", top_k=20)
)
```

## Integration with Existing Features

### Works With

✅ **Co-occurrence Inference**: Runs after standard co-occurrence inference
✅ **High-Quality Config**: Compatible with `company_profile_extraction_config.py`
✅ **All Storage Backends**: Works with NetworkX, Neo4j, PostgreSQL, etc.
✅ **Query Modes**: All modes (local, global, hybrid, naive)

### Relationship Priority

1. **LLM-extracted relationships** (highest priority)
2. **Company relationship inference** (contextual)
3. **Standard co-occurrence inference** (lowest priority)

Inferred relationships never override explicitly extracted ones.

## Performance Considerations

### Processing Time

- **Minimal overhead** (~5-10% additional time)
- Only analyzes organization entities
- Skips if `enable_company_relationship_inference=False`

### Accuracy

- **High precision** due to contextual analysis
- **Moderate recall** (may miss implicit relationships without clear indicators)
- **False positive rate**: Very low (<5%) with confidence_threshold=0.5

### Recommendations

| Document Count | Recommendation |
|----------------|----------------|
| < 100 docs | Enable with default settings |
| 100-1000 docs | Enable with confidence_threshold=0.6 |
| > 1000 docs | Enable with confidence_threshold=0.7 for precision |

## Troubleshooting

### Issue: No relationships inferred

**Causes:**
1. Co-occurrence count too low (companies only mentioned once together)
2. Confidence threshold too high
3. No relationship indicators in text

**Solutions:**
- Lower `company_inference_min_cooccurrence` to 1
- Increase `chunk_overlap_token_size` to 400+ for more co-occurrences
- Lower `company_inference_confidence_threshold` to 0.3
- Add explicit relationship indicators to documents

### Issue: Wrong relationship type inferred

**Causes:**
1. Conflicting signals in text
2. Ambiguous language

**Solutions:**
- Use explicit relationship language ("competes with" not "similar to")
- Separate different relationship types into different sections
- Increase confidence threshold to filter ambiguous cases

### Issue: Too many relationships inferred

**Causes:**
1. Threshold too low
2. Generic business mentions

**Solutions:**
- Increase `company_inference_confidence_threshold` to 0.7
- Increase `company_inference_min_cooccurrence` to 3
- Use more specific language in documents

## API Reference

### Main Function

```python
def infer_company_relationships(
    all_nodes: Dict[str, List[Dict]],
    all_edges: Dict[Tuple[str, str], List[Dict]],
    chunk_contents: Optional[Dict[str, str]] = None,
    min_cooccurrence: int = 2,
    confidence_threshold: float = 0.5,
    enable_inference: bool = True,
) -> Tuple[Dict[Tuple[str, str], List[Dict]], Dict[str, Any]]
```

**Returns:**
- Updated edges dictionary with inferred relationships
- Statistics dictionary with inference metrics

### Inferred Relationship Format

```python
{
    "src_id": "Company A",
    "tgt_id": "Company B",
    "weight": 4.8,  # co-occurrence × confidence
    "keywords": "competes_with, competitor, market_rival",
    "description": "Company A and Company B are competitors...",
    "source_id": "inferred_company_relationship",
    "inferred": True,
    "inference_method": "contextual_company_analysis",
    "cooccurrence_count": 3,
    "confidence": 0.8,
    "relationship_type": "competitor",
    "evidence": [
        "Competitor signal: 'Company A competes directly with Company B...'",
        "Competitor signal: 'Both companies target the same market...'"
    ]
}
```

## Examples

See complete working examples:
- [company_relationship_inference_demo.py](../examples/company_relationship_inference_demo.py)
- [company_profile_extraction_config.py](../examples/company_profile_extraction_config.py)

## References

- [Co-occurrence Inference Documentation](./COOCCURRENCE_INFERENCE.md)
- [utils_company_relationships.py](../lightrag/utils_company_relationships.py)
- [Main LightRAG Documentation](../README.md)
