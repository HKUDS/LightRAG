# Company Relationship Inference - Quick Start Guide

## What Does This Feature Do?

Automatically detects **competitor**, **partnership**, and **supply chain** relationships between companies in your knowledge graph by analyzing document context.

## 5-Minute Setup

### 1. Enable the Feature

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    addon_params={
        "enable_company_relationship_inference": True,
    }
)
```

### 2. Structure Your Documents

Format your documents with clear entity markers:

```markdown
Summary: Acme Corp is a technology company.
Entity: Acme Corp

Acme Corp competes with TechStar Inc. in the cloud market.
The company also partners with DataFirm for analytics.
```

### 3. Insert and Query

```python
# Insert documents
await rag.ainsert(document)

# Query relationships
result = await rag.aquery("What companies compete with Acme Corp?")
```

## Document Format Template

```markdown
Summary: [Your Main Company] is a [description]
Entity: [Your Main Company]

[Main Company] competes with [Competitor Company] in [market/industry].
Both companies offer similar [products/services].

[Main Company] partners with [Partner Company] to [collaboration details].

[Main Company] sources [product/service] from [Supplier Company].
```

## Relationship Detection Keywords

### For Competitors 🥊
Use these words to indicate competition:
- "competes with", "competitor", "rival"
- "versus", "alternative to", "similar to"
- "market leader", "same market", "same industry"

### For Partnerships 🤝
Use these words to indicate partnerships:
- "partners with", "partnership", "collaboration"
- "strategic alliance", "works with", "joint venture"
- "integrates with", "certified partner"

### For Supply Chain 📦
Use these words to indicate vendor/customer relationships:
- "supplier", "vendor", "customer", "client"
- "provides to", "supplies", "sources from", "procures"

## Configuration Options

```python
rag = LightRAG(
    working_dir="./rag_storage",

    # Increase overlap for better entity co-occurrence
    chunk_overlap_token_size=400,  # 33% overlap

    addon_params={
        # Enable the feature
        "enable_company_relationship_inference": True,

        # Min times companies must appear together (default: 2)
        "company_inference_min_cooccurrence": 2,

        # Confidence threshold 0-1 (default: 0.5)
        "company_inference_confidence_threshold": 0.5,
    }
)
```

## Tuning Parameters

### If Getting Too Many Relationships:
```python
"company_inference_confidence_threshold": 0.7,  # Increase threshold
"company_inference_min_cooccurrence": 3,        # Require more co-occurrences
```

### If Getting Too Few Relationships:
```python
"company_inference_confidence_threshold": 0.3,  # Decrease threshold
"company_inference_min_cooccurrence": 1,        # Allow single co-occurrence
"chunk_overlap_token_size": 500,               # Increase overlap
```

## Example Query Patterns

```python
# Find competitors
await rag.aquery("What companies compete with [Company]?")

# Find partners
await rag.aquery("Who are [Company]'s partners?")

# Find suppliers
await rag.aquery("Who supplies to [Company]?")

# All relationships
await rag.aquery("What are all business relationships for [Company]?")
```

## Complete Example

```python
import asyncio
from lightrag import LightRAG

async def main():
    # Initialize with company inference
    rag = LightRAG(
        working_dir="./company_demo",
        addon_params={
            "enable_company_relationship_inference": True,
        }
    )

    # Insert document
    doc = """
    Summary: TechCorp is a cloud services provider.
    Entity: TechCorp

    TechCorp competes with CloudGiant Inc. for enterprise customers.
    The company partners with DataAnalytics Co. for data solutions.
    TechCorp sources hardware from HardwareSupplier Ltd.
    """

    await rag.ainsert(doc)

    # Query
    result = await rag.aquery("What are TechCorp's business relationships?")
    print(result)

    # Expected inferred relationships:
    # 1. TechCorp ←→ CloudGiant [COMPETITOR]
    # 2. TechCorp ←→ DataAnalytics [PARTNERSHIP]
    # 3. TechCorp ←→ HardwareSupplier [SUPPLY_CHAIN]

asyncio.run(main())
```

## Best Practices

✅ **DO:**
- Use explicit relationship keywords ("competes with", "partners with")
- Clearly mark the main company with "Entity:" or "Summary:"
- Mention companies together in the same sentences
- Use specific company names (not "the company", "competitors")

❌ **DON'T:**
- Use vague language ("related to", "connected with")
- Mix relationship types in the same sentence
- Use only implicit indicators without explicit ones
- Forget to specify the main entity in documents

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No relationships detected | Add explicit keywords like "competes with", "partners with" |
| Wrong relationship type | Use more specific language, separate different relationship types |
| Too many false positives | Increase `confidence_threshold` to 0.7 |
| Missing relationships | Increase `chunk_overlap_token_size` to 400+ |

## Next Steps

- See full documentation: [COMPANY_RELATIONSHIP_INFERENCE.md](../docs/COMPANY_RELATIONSHIP_INFERENCE.md)
- Run the demo: `python examples/company_relationship_inference_demo.py`
- Use high-quality config: `from examples.company_profile_extraction_config import get_high_quality_extraction_config`

## Need Help?

- Check the full demo: [company_relationship_inference_demo.py](./company_relationship_inference_demo.py)
- Review example config: [company_profile_extraction_config.py](./company_profile_extraction_config.py)
- See the code: [utils_company_relationships.py](../lightrag/utils_company_relationships.py)
