# Knowledge Graph Visualization Guide

## Overview

LightRAG provides visualization tools that distinguish between **explicitly extracted** and **inferred** relationships in the knowledge graph. This makes it easy to understand the quality and source of relationships at a glance.

## Visual Distinction

### Edge Types

| Edge Type | Visual Style | Description |
|-----------|-------------|-------------|
| **Explicit** | Thick, solid lines | Relationships directly extracted from text by LLM |
| **Inferred** | Thin, dashed, faded lines | Relationships inferred from co-occurrence or context |

### Inferred Relationship Colors

Inferred company relationships use color coding by type:

| Relationship Type | Color | Icon | Example |
|-------------------|-------|------|---------|
| **Competitor** | 🔴 Red (faded) | 🥊 | TechCorp ←→ CompetitorInc |
| **Partnership** | 🟢 Green (faded) | 🤝 | TechCorp ←→ PartnerCo |
| **Supply Chain** | 🔵 Blue (faded) | 📦 | TechCorp ←→ SupplierLtd |
| **Generic** | ⚪ Gray (faded) | - | Generic inferred relationships |

## HTML Visualization (Pyvis)

### Usage

```python
python examples/graph_visual_with_html.py
```

### Features

1. **Interactive Graph**: Pan, zoom, and drag nodes
2. **Dashed Lines**: Inferred relationships shown with dashed pattern
3. **Color Coding**: Relationships colored by type
4. **Hover Tooltips**: Show relationship details including:
   - Description
   - Confidence score (for inferred relationships)
   - Inference method
   - Whether explicitly extracted or inferred

### Example Output

```
Knowledge Graph Visualization Generated
======================================================================

Legend:
  • Solid lines (thick) = Explicitly extracted relationships
  • Dashed lines (thin, faded) = Inferred relationships
    - 🥊 Red = Competitor relationships
    - 🤝 Green = Partnership relationships
    - 📦 Blue = Supply chain relationships
    - Gray = Generic inferred relationships

Hover over edges to see details including confidence scores.
======================================================================
```

### Visual Example

```
[TechCorp] ═════════════ [Person A]     (Thick solid line - explicit)

[TechCorp] ─ ─ ─ ─ ─ ─  [CompetitorInc] (Thin dashed red line - inferred competitor)

[TechCorp] ─ ─ ─ ─ ─ ─  [PartnerCo]     (Thin dashed green line - inferred partner)
```

## 3D Visualization (OpenGL)

### Usage

```python
from lightrag.tools.lightrag_visualizer import main
main()
```

### Features

1. **3D Navigation**:
   - WASD: Move camera
   - Q/E: Up/down
   - Right-click drag: Look around
   - Mouse wheel: Adjust speed

2. **Edge Rendering**:
   - Explicit edges: Thick, full-color lines
   - Inferred edges: Thin, faded lines (40% opacity)

3. **Node Details Panel**:
   - Click nodes to see details
   - Connection table shows:
     - Node name (clickable)
     - Relationship type with color coding
     - Inference status with confidence
     - Hover for additional info

### Node Details Table

When you click a node, you'll see a connections table:

| Node | Description | Keywords | Type | Inferred |
|------|-------------|----------|------|----------|
| CompetitorInc | Competitor company | competes_with | 🥊 Competitor | Yes (0.8) |
| PartnerCo | Partner company | partners_with | 🤝 Partner | Yes (0.7) |
| SupplierLtd | Supplier | vendor_customer | 📦 Supply Chain | Yes (0.6) |
| Person A | Employee | works_for | - | No |

### Visual Appearance

- **Explicit edges**: Bright, thick lines matching node colors
- **Inferred edges**:
  - Red (faded): Competitor relationships
  - Green (faded): Partnership relationships
  - Blue (faded): Supply chain relationships
  - 40% opacity for "faded" effect
  - 80% of normal edge width

## Neo4j Visualization

### Setup

```python
# See examples/graph_visual_with_neo4j.py
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    graph_storage="Neo4JStorage",
)
```

### Query in Neo4j Browser

```cypher
// Find all inferred relationships
MATCH (a)-[r]->(b)
WHERE r.inferred = 'true'
RETURN a, r, b

// Find competitor relationships
MATCH (a)-[r]->(b)
WHERE r.relationship_type = 'competitor'
RETURN a, r, b

// Find high-confidence inferred relationships
MATCH (a)-[r]->(b)
WHERE r.inferred = 'true' AND toFloat(r.confidence) > 0.7
RETURN a, r, b
```

### Styling in Neo4j Browser

Add to your Neo4j Browser styles:

```css
/* Inferred relationships - dashed */
relationship[inferred = "true"] {
  shaft-width: 1px;
  caption: {relationship_type};
}

/* Competitor - red */
relationship[relationship_type = "competitor"] {
  color: #FF453A;
  shaft-width: 2px;
}

/* Partnership - green */
relationship[relationship_type = "partnership"] {
  color: #32C759;
  shaft-width: 2px;
}

/* Supply chain - blue */
relationship[relationship_type = "supply_chain"] {
  color: #5AC8FA;
  shaft-width: 2px;
}
```

## Customizing Visualization

### HTML Visualization

Modify [examples/graph_visual_with_html.py](../examples/graph_visual_with_html.py):

```python
# Change inferred edge style
if is_inferred:
    if relationship_type == "competitor":
        edge["color"] = "rgba(255, 0, 0, 0.6)"  # More opaque red
        edge["dashes"] = [10, 5]  # Different dash pattern
    edge["width"] = 2  # Thicker inferred edges
```

### 3D Visualization

Modify [lightrag/tools/lightrag_visualizer/graph_visualizer.py](../lightrag/tools/lightrag_visualizer/graph_visualizer.py):

```python
# In update_buffers() method around line 690
if is_inferred:
    if relationship_type == "competitor":
        edge_color = glm.vec3(1.0, 0.0, 0.0)  # Brighter red
    edge_color *= 0.6  # Different opacity (was 0.4)
```

## Interpreting the Visualization

### Reading the Graph

1. **Dense solid lines**: Core explicitly extracted knowledge
2. **Sparse dashed lines**: Inferred relationships filling gaps
3. **Color clusters**: Different types of business relationships

### Quality Indicators

- **Thick solid lines**: High confidence (explicitly stated)
- **Thin faded lines**: Medium confidence (inferred from context)
- **Hover/click for confidence**: 0.7+ = reliable, 0.5-0.7 = moderate, <0.5 = low

### Common Patterns

```
[Company A] ═══════ [Person 1]
            ═══════ [Person 2]
            ─ ─ ─ ─ [Company B] (competitor)
            ─ ─ ─ ─ [Company C] (partner)
```

**Interpretation**: Company A has explicit employee relationships, plus inferred competitive and partnership relationships.

## Troubleshooting

### Issue: All edges look the same

**Cause**: Graph doesn't have `inferred` attribute

**Solution**: Re-extract graph with inference enabled:
```python
rag = LightRAG(
    working_dir="./rag_storage",
    addon_params={
        "enable_company_relationship_inference": True,
    }
)
```

### Issue: Can't see inferred edges

**Causes**:
1. No inferred relationships in graph
2. Edge width too small
3. Transparency too high

**Solutions**:
- Check if inference is enabled
- Increase edge width in settings
- Reduce transparency (increase opacity multiplier)

### Issue: Too many colors/confusing

**Solution**: Simplify color scheme by relationship type only:
```python
# In graph_visual_with_html.py
if is_inferred:
    edge["color"] = "rgba(200, 200, 200, 0.3)"  # All inferred = gray
```

## Best Practices

### 1. Start with HTML Visualization
- Fast to generate
- Good for initial exploration
- Easy to share (single HTML file)

### 2. Use 3D for Large Graphs
- Better for 100+ nodes
- Navigate in 3D space
- Less cluttered

### 3. Use Neo4j for Analysis
- Query-based exploration
- Advanced filtering
- Production dashboards

### 4. Adjust Edge Width Based on Graph Size
- Small graphs (<50 nodes): Use thicker edges
- Medium graphs (50-200 nodes): Default settings
- Large graphs (>200 nodes): Use thinner edges

### 5. Filter by Confidence
For cleaner visualization, filter low-confidence inferred edges:
```python
# In visualization script
if is_inferred and float(confidence) < 0.6:
    continue  # Skip low-confidence edges
```

## Examples

See complete examples:
- [HTML Visualization](../examples/graph_visual_with_html.py)
- [3D Visualization](../lightrag/tools/lightrag_visualizer/graph_visualizer.py)
- [Neo4j Visualization](../examples/graph_visual_with_neo4j.py)

## Related Documentation

- [Company Relationship Inference](./COMPANY_RELATIONSHIP_INFERENCE.md)
- [Quick Start Guide](../examples/COMPANY_INFERENCE_QUICKSTART.md)
- [Main Documentation](../README.md)
