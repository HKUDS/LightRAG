# 3D Knowledge Graph Visualization

This document describes the 3D knowledge graph visualization feature for LightRAG.

## Overview

The 3D visualization projects the knowledge graph into a 3D space using UMAP dimensionality reduction on entity embeddings, clusters entities using HDBSCAN, and renders them in an interactive Three.js-based viewer with a sci-fi galaxy aesthetic (inspired by RealDeepResearch).

## Architecture

### Backend

**Route**: `lightrag/api/routers/graph3d_routes.py`

Endpoints:
- `GET /graph3d/status` - Check if layout has been computed
- `GET /graph3d/layout.bin` - Binary layout file (positions + clusters + IDs)
- `GET /graph3d/layout_meta.json` - JSON metadata (entity info + edges)
- `GET /graph3d/viewer` - Standalone HTML viewer page

The route is registered in `lightrag_server.py` alongside other graph routes.

### Layout Computation

**Script**: `lightrag/tools/compute_3d_layout.py`

This offline script:
1. Loads entity embeddings from `vdb_entities.json` (3840-dim float16, zlib-compressed)
2. Runs UMAP to project into 3D space (n_components=3, metric='cosine')
3. Clusters entities using HDBSCAN (or KMeans fallback)
4. Loads graph structure from `graph_chunk_entity_relation.graphml`
5. Maps graph edges to VDB entity indices
6. Writes binary layout file + JSON metadata

**Usage**:
```bash
# Activate the graph3d venv
source .venv-graph3d/bin/activate

# Compute layout (takes 30-60 min for 197k entities)
python -m lightrag.tools.compute_3d_layout \
  --working-dir data/rag_storage \
  --n-neighbors 30 \
  --min-cluster-size 50
```

**Output files** (in `data/rag_storage/`):
- `graph3d_layout.bin` - Binary layout (~50MB for 197k entities)
- `graph3d_layout_meta.json` - Metadata with entity info and edges

### Frontend

**Viewer**: `lightrag/api/static/graph3d_viewer.html`

Standalone HTML page using:
- Three.js 0.160 (via CDN)
- WebGL rendering with InstancedMesh for performance
- UnrealBloomPass for glow effects
- OrbitControls for camera interaction
- Binary data format for fast loading

Features:
- Interactive 3D navigation (rotate, zoom, pan)
- Hover tooltips showing entity names
- Click to view entity details in side panel
- Toggle edges and labels
- Reset camera view
- Cluster-based color coding

## Accessing the Viewer

Once the layout is computed and the server is running:

```bash
# Start the server
lightrag-server --host 0.0.0.0 --port 9621

# Open the viewer in browser
open http://localhost:9621/graph3d/viewer
```

## Per-Instance Viewer Title

When multiple LightRAG instances serve different corpora, give each viewer its
own title by placing a `graph3d_config.json` in the instance's working dir:

```json
{
  "title_en": "Research Papers Knowledge Graph",
  "title_zh": "论文知识图谱3D视图",
  "title_full_en": "Research Papers Knowledge Graph · 3D Galaxy",
  "title_full_zh": "论文知识图谱 · 3D星系视图"
}
```

The `/graph3d/viewer` route injects this into the page as `VIEWER_CONFIG`; the
viewer applies the values on top of the i18n table (`app.title` header,
`app.title.full` browser tab). All keys are optional — omitted keys keep the
built-in defaults, and a missing file means the stock "GK Brain" title.

## Binary Format

The binary layout file (`graph3d_layout.bin`) uses a compact format for fast frontend loading:

```
Float32Array: positions (N * 3 floats)
Uint16Array: clusters (N uint16s)
Uint32Array: id_offsets (N+1 uint32s)
Uint16Array: id_data (variable length, UTF-16LE entity names)
```

This format allows zero-copy loading in the browser via `TypedArray` views.

## Performance Considerations

- **UMAP computation**: 30-60 minutes for 197k entities on 64-core machine
- **Memory**: ~3GB RAM for 197k × 3840 float32 matrix
- **Binary file size**: ~50MB for 197k entities
- **Frontend rendering**: 60 FPS with WebGL, even for 200k+ points

## Vector Encoding

LightRAG stores entity embeddings as:
1. float16 numpy array (3840-dim)
2. zlib-compressed
3. base64-encoded
4. Stored in `vdb_entities.json`

The `compute_3d_layout.py` script handles decoding transparently.

## Future Enhancements

- [ ] Add label rendering with CSS2DRenderer
- [ ] Implement entity search in viewer
- [ ] Add filtering by cluster or entity type
- [ ] Support for relation filtering
- [ ] Integrate into WebUI as a tab (requires React rebuild)
- [ ] Add animation for entity relationships
- [ ] Support for dynamic layout updates when graph changes

## Dependencies

The layout computation requires:
- `umap-learn` - Dimensionality reduction
- `numpy` - Array operations
- `networkx` - Graph loading
- `scikit-learn` - KMeans clustering (fallback)
- `hdbscan` - Density-based clustering (optional)

Install with:
```bash
uv venv .venv-graph3d --python 3.12
uv pip install umap-learn numpy networkx --python .venv-graph3d/bin/python
```
