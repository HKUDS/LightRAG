"""Compute 3D layout for knowledge graph visualization.

Reads entity embeddings from LightRAG's VDB, runs UMAP to project into 3D,
clusters with HDBSCAN, and writes a binary layout file plus JSON metadata.
The binary format matches RealDeepResearch's Galaxy viewer for fast frontend loading.

Usage:
    python -m lightrag.tools.compute_3d_layout [--working-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import networkx as nx

# Default working directory (LightRAG's data/rag_storage)
DEFAULT_WORKING_DIR = Path("data/rag_storage")


def decode_vector(vector_b64: str) -> np.ndarray:
    """Decode a zlib-compressed, base64-encoded float16 vector."""
    raw = base64.b64decode(vector_b64)
    decompressed = zlib.decompress(raw)
    return np.frombuffer(decompressed, dtype=np.float16).astype(np.float32)


def load_entity_vectors(vdb_path: Path) -> tuple[list[str], np.ndarray, list[dict]]:
    """Load all entity vectors from VDB JSON file.

    Returns:
        entity_names: List of entity names
        matrix: (N, dim) float32 array of embeddings
        metadata: List of metadata dicts per entity
    """
    print(f"Loading VDB from {vdb_path}...")
    with open(vdb_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embedding_dim = data.get("embedding_dim", 3840)
    entries = data.get("data", [])
    print(f"Found {len(entries)} entities, embedding_dim={embedding_dim}")

    entity_names = []
    vectors = []
    metadata = []

    for i, entry in enumerate(entries):
        if i % 10000 == 0 and i > 0:
            print(f"  Decoded {i}/{len(entries)} vectors...")
        entity_names.append(entry["entity_name"])
        vectors.append(decode_vector(entry["vector"]))
        metadata.append({
            "content": entry.get("content", ""),
            "entity_name": entry["entity_name"],
            "source_id": entry.get("source_id", ""),
            "file_path": entry.get("file_path", ""),
        })

    matrix = np.stack(vectors, axis=0)
    print(f"Loaded matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
    return entity_names, matrix, metadata


def compute_umap_layout(
    matrix: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Run UMAP to project high-dim embeddings to 3D.

    Returns:
        positions: (N, 3) float32 array of 3D positions
    """
    import umap

    print(f"Computing UMAP layout: n_components={n_components}, "
          f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    print(f"Input matrix shape: {matrix.shape}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        verbose=True,
    )
    positions = reducer.fit_transform(matrix)
    print(f"UMAP output shape: {positions.shape}")
    return positions.astype(np.float32)


def compute_mds_layout(
    matrix: np.ndarray,
    n_components: int = 3,
    n_landmarks: int = 3000,
    max_iter: int = 300,
    random_state: int = 42,
    target_radius: float = 5.0,
    dist_gamma: float = 1.0,
) -> np.ndarray:
    """Landmark MDS: metric MDS (SMACOF) on landmarks, then place every point
    by distance triangulation (Gower's interpolation formula).

    Unlike UMAP/t-SNE, MDS explicitly minimizes stress — the squared error
    between high-dim distances and 3D distances — so the layout preserves
    RELATIVE distances at the cost of local cluster crispness.

    Full metric MDS is O(N^2) in memory and time and infeasible for N=190k;
    landmarks cap the expensive part at n_landmarks^2.

    Args:
        matrix: (N, dim) float32 array of unit-normalized embeddings.
        n_components: target dimensionality (3 for the viewer).
        n_landmarks: number of landmarks for the SMACOF step.
        max_iter: SMACOF iterations.
        random_state: landmark selection + MDS seed.
        target_radius: 99th-percentile radius of the output layout. MDS
            coordinates are arbitrary-scale (stress is scale-invariant), so
            the raw output (~0.35 units for cosine inputs) is far too small
            for the viewer's fixed POINT_SIZE and saturates the bloom pass.
            Normalize to the same scale UMAP produces (~10 units extent).
        dist_gamma: exponent applied to cosine distances before MDS
            (d ** gamma). gamma < 1 expands small distances relative to
            large ones, breaking the homogeneous "ball" open visually.
            Because the transform is monotonic, rank-based fidelity
            (Spearman correlation) is mathematically unchanged.

    Returns:
        positions: (N, n_components) float32 array.
    """
    from sklearn.manifold import MDS

    N = len(matrix)
    n_landmarks = min(n_landmarks, N)
    rng = np.random.default_rng(random_state)
    lm_idx = np.sort(rng.choice(N, n_landmarks, replace=False))
    print(f"Landmark MDS: {n_landmarks} landmarks out of {N} points")

    # Landmark-landmark cosine distance matrix (embeddings are unit-normalized)
    L = matrix[lm_idx].astype(np.float64)
    D_lm = 1.0 - np.clip(L @ L.T, -1, 1)
    np.fill_diagonal(D_lm, 0.0)
    if dist_gamma != 1.0:
        D_lm = np.clip(D_lm, 0, None) ** dist_gamma
        print(f"Applied distance exponent gamma={dist_gamma} "
              f"(monotonic: rank fidelity unchanged)")

    print(f"Running metric MDS (SMACOF, max_iter={max_iter}) on landmarks...")
    mds = MDS(
        n_components=n_components,
        metric=True,
        dissimilarity="precomputed",
        n_init=1,
        max_iter=max_iter,
        normalized_stress="auto",
        random_state=random_state,
    )
    Y = mds.fit_transform(D_lm)  # (n_landmarks, n_components)
    print(f"Landmark MDS done, stress={mds.stress_:.1f}")

    # Gower interpolation: place each point from its squared distances to the
    # landmarks. b = -0.5 * (delta^2 - column means of landmark D^2); the
    # coordinates are the least-squares placement against the (centered)
    # landmark configuration.
    Y = Y - Y.mean(axis=0)
    D_lm_sq = D_lm ** 2
    col_means = D_lm_sq.mean(axis=0)  # (n_landmarks,)
    Y_pinv = np.linalg.pinv(Y)  # (n_components, n_landmarks)

    print("Triangulating all points against landmarks...")
    positions = np.empty((N, n_components), dtype=np.float32)
    chunk = 20000
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        sims = matrix[start:end].astype(np.float64) @ L.T  # (chunk, n_landmarks)
        d = np.clip(1.0 - np.clip(sims, -1, 1), 0, None)
        if dist_gamma != 1.0:
            d = d ** dist_gamma
        d_sq = d ** 2
        b = -0.5 * (d_sq - col_means)
        positions[start:end] = (b @ Y_pinv.T).astype(np.float32)
        if (start // chunk) % 5 == 0:
            print(f"  Placed {end}/{N} points...")

    print(f"MDS output shape: {positions.shape}")

    # Normalize scale: center and rescale so the 99th-percentile radius
    # matches target_radius (see docstring).
    center = positions.mean(axis=0)
    positions -= center
    r99 = np.percentile(np.linalg.norm(positions, axis=1), 99)
    if r99 > 0:
        scale = np.float32(target_radius / r99)
        positions *= scale
        print(f"Normalized layout scale: factor={float(scale):.2f}, "
              f"extent={(positions.max(axis=0) - positions.min(axis=0)).round(2)}")
    return positions


def compute_sfdp_layout(
    graphml_path: Path,
    entity_names: list[str],
    matrix: np.ndarray,
    target_radius: float = 5.0,
    random_state: int = 42,
) -> np.ndarray:
    """Force-directed 3D layout via graphviz sfdp (multilevel spring model).

    Embedding-based layouts (UMAP/MDS) ignore the graph and resolve to
    balls/blobs on homogeneous corpora. sfdp uses the graph TOPOLOGY —
    but on hub-dominated graphs (median degree 1, mega-hubs with 3k+
    edges, one giant component) it produces a single dense core with long
    tendrils, NOT separated lobes: ~90% of nodes pack into a tiny core
    while peripheral chains stretch ~25x further. For the neuron/
    brain-sector look use compute_hierarchical_layout instead.

    Entities without graph edges are placed next to their nearest
    connected neighbor in embedding space (plus small jitter), so they
    drift around the region they semantically belong to.

    Returns:
        positions: (N, 3) float32 array, centered and scale-normalized.
    """
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)
    name_to_idx = {name: i for i, name in enumerate(entity_names)}
    N = len(entity_names)

    edges = set()
    for u, v in G.edges():
        if u in name_to_idx and v in name_to_idx:
            a, b = name_to_idx[u], name_to_idx[v]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    edges = sorted(edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {len(edges)} usable edges")

    # Only connected nodes go into the DOT file; sfdp would place isolates
    # arbitrarily. Track which nodes appear.
    connected_mask = np.zeros(N, dtype=bool)
    for a, b in edges:
        connected_mask[a] = True
        connected_mask[b] = True
    n_connected = int(connected_mask.sum())
    print(f"Connected nodes: {n_connected}/{N} "
          f"({N - n_connected} isolates placed post-hoc)")

    print("Running sfdp (multilevel force-directed, dim=3)...")
    node_pos = _run_sfdp_3d(edges)

    positions = np.zeros((N, 3), dtype=np.float32)
    placed = np.zeros(N, dtype=bool)
    for node_id, xyz in node_pos.items():
        positions[node_id] = xyz
        placed[node_id] = True
    print(f"sfdp placed {int(placed.sum())} nodes")

    # Place unplaced/isolated nodes near their nearest connected neighbor
    # in embedding space. Embeddings are unit-normalized, so nearest by
    # cosine == argmax of the dot product — use chunked matmul (BLAS).
    # A cKDTree in 3840-dim degenerates to brute force and never finishes.
    need = ~placed
    if need.any():
        rng = np.random.default_rng(random_state)
        have = np.where(placed)[0]
        need_idx = np.where(need)[0]
        L = matrix[have].astype(np.float64).T  # (dim, n_have)
        nn = np.empty(len(need_idx), dtype=np.int64)
        chunk = 10000
        for start in range(0, len(need_idx), chunk):
            sel = need_idx[start:start + chunk]
            sims = matrix[sel].astype(np.float64) @ L  # (chunk, n_have)
            nn[start:start + chunk] = sims.argmax(axis=1)
        positions[need] = (
            positions[have[nn]]
            + rng.normal(0, 0.05, size=(len(need_idx), 3)).astype(np.float32)
        )
        print(f"Placed {len(need_idx)} isolates near embedding neighbors")

    # Normalize scale like the MDS path (see compute_mds_layout).
    center = positions.mean(axis=0)
    positions -= center
    r99 = np.percentile(np.linalg.norm(positions, axis=1), 99)
    if r99 > 0:
        scale = np.float32(target_radius / r99)
        positions *= scale
        print(f"Normalized layout scale: factor={float(scale):.4f}, "
              f"extent={(positions.max(axis=0) - positions.min(axis=0)).round(2)}")
    return positions


def _run_sfdp_3d(
    edges: list[tuple[int, int]],
) -> dict[int, tuple[float, float, float]]:
    """Run graphviz sfdp (dimen=3) on an edge list, return {node_id: xyz}.

    Shared by compute_sfdp_layout and compute_hierarchical_layout.
    graphviz needs BOTH -Gdim (output dims) and -Gdimen (layout dims), and
    only -Tdot carries true 3D positions (-Tplain's 4th field is node
    width, not z).
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".gv", prefix="graph3d_sfdp_", delete=False
    ) as f:
        dot_path = f.name
        f.write("graph G {\n")
        for a, b in edges:
            f.write(f"{a} -- {b};\n")
        f.write("}\n")
    print(f"Wrote DOT input: {dot_path} ({len(edges)} edges)")

    result = subprocess.run(
        ["sfdp", "-Gdim=3", "-Gdimen=3", "-Goverlap=false", "-Tdot", dot_path],
        capture_output=True, text=True, check=True,
    )
    pos: dict[int, tuple[float, float, float]] = {}
    # Node records look like:  123\t[height=0.5,\n\t\tpos="x,y,z", ...];
    # Edge records contain "--" and are excluded by the leading-id match.
    for m in re.finditer(r'^\s*(\d+)\s*\[(.*?)\];', result.stdout, re.S | re.M):
        pm = re.search(r'pos="([-\d.eE]+),([-\d.eE]+),([-\d.eE]+)"', m.group(2))
        if pm:
            pos[int(m.group(1))] = (
                float(pm.group(1)), float(pm.group(2)), float(pm.group(3))
            )
    return pos


def compute_hierarchical_layout(
    graphml_path: Path,
    entity_names: list[str],
    matrix: np.ndarray,
    clusters: np.ndarray,
    target_radius: float = 5.0,
    cluster_tightness: float = 0.35,
    random_state: int = 42,
) -> np.ndarray:
    """Hierarchical presentation-first layout: brain sectors + filaments.

    Pure force-directed layouts collapse on hub-dominated graphs (median
    degree 1, mega-hubs with 3k+ edges): one dense ball with hair, no
    sectors. This layout builds the sector structure explicitly:

    1. SKELETON: the embedding-space cluster centroids are laid out by
       sfdp on the cluster adjacency graph (an edge exists where KG edges
       connect the two clusters; isolated clusters are anchored to their
       3 nearest centroid neighbors in embedding space), so related
       sectors sit together and unrelated ones are pushed apart.
    2. LOBES: within each cluster, nodes are placed by a local
       randomized-PCA projection of their own embeddings (top-3
       components), giving each sector an oriented, elongated lobe shape
       instead of a uniform ball.
    3. Lobe radius scales with size^(1/3) relative to centroid spacing,
       controlled by cluster_tightness (smaller = more separated lobes).

    Requires cluster labels up front (embedding-space clustering); pure
    graph isolates belong to a cluster like everyone else, so no node is
    left unplaced.

    Returns:
        positions: (N, 3) float32 array, centered and scale-normalized.
    """
    from sklearn.decomposition import PCA

    N = len(entity_names)
    labels = clusters.astype(np.int64)
    K = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=K)
    print(f"Hierarchical layout: {N} nodes in {K} sectors "
          f"(size {sizes.min()}..{sizes.max()})")

    # --- Skeleton: cluster adjacency from the KG -------------------------
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)
    name_to_idx = {name: i for i, name in enumerate(entity_names)}
    adjacency: dict[tuple[int, int], int] = {}
    for u, v in G.edges():
        if u in name_to_idx and v in name_to_idx:
            ca, cb = labels[name_to_idx[u]], labels[name_to_idx[v]]
            if ca != cb:
                key = (ca, cb) if ca < cb else (cb, ca)
                adjacency[key] = adjacency.get(key, 0) + 1
    print(f"Cluster adjacency: {len(adjacency)} inter-sector edge pairs")

    # Centroid embeddings (unit-normalized) for the kNN fallback anchors
    centroid_emb = np.zeros((K, matrix.shape[1]), dtype=np.float64)
    for c in range(K):
        rows = matrix[labels == c]
        centroid_emb[c] = rows.mean(axis=0)
    centroid_emb /= np.linalg.norm(centroid_emb, axis=1, keepdims=True) + 1e-12

    connected = {c for edge in adjacency for c in edge}
    skeleton_edges = sorted(adjacency)
    isolated = [c for c in range(K) if c not in connected]
    if isolated:
        sims = centroid_emb @ centroid_emb.T
        np.fill_diagonal(sims, -np.inf)
        for c in isolated:
            for nb in np.argpartition(-sims[c], 3)[:3]:
                skeleton_edges.append((c, int(nb)) if c < nb else (int(nb), c))
        skeleton_edges = sorted(set(skeleton_edges))
        print(f"Anchored {len(isolated)} isolated sectors by embedding kNN")

    print("Running sfdp on the sector skeleton...")
    centroid_pos = _run_sfdp_3d(skeleton_edges)
    C = np.zeros((K, 3), dtype=np.float64)
    for c in range(K):
        C[c] = centroid_pos.get(c, (0.0, 0.0, 0.0))
    C -= C.mean(axis=0)

    # Normalize skeleton, then size lobes relative to centroid spacing
    r99 = np.percentile(np.linalg.norm(C, axis=1), 99)
    if r99 > 0:
        C *= target_radius / r99
    if K > 1:
        d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(d, np.inf)
        med_spacing = float(np.median(d.min(axis=1)))
    else:
        med_spacing = target_radius
    radii = cluster_tightness * med_spacing * (
        sizes / np.median(sizes)) ** (1.0 / 3.0)
    print(f"Sector spacing: median={med_spacing:.2f}, lobe radii "
          f"{radii.min():.2f}..{radii.max():.2f}")

    # --- Lobes: local PCA per cluster ------------------------------------
    rng = np.random.default_rng(random_state)
    positions = np.zeros((N, 3), dtype=np.float32)
    for c in range(K):
        idx = np.where(labels == c)[0]
        n = len(idx)
        if n >= 4:
            local = PCA(n_components=3, svd_solver="randomized",
                        random_state=random_state).fit_transform(matrix[idx])
            lr99 = np.percentile(
                np.linalg.norm(local - local.mean(axis=0), axis=1), 99)
            if lr99 > 0:
                local /= lr99
        else:
            local = rng.normal(0, 0.3, size=(n, 3))
        positions[idx] = (C[c] + local * radii[c]).astype(np.float32)

    # Global normalization (center + r99 radius -> target_radius)
    positions -= positions.mean(axis=0)
    r99 = np.percentile(np.linalg.norm(positions, axis=1), 99)
    if r99 > 0:
        scale = np.float32(target_radius / r99)
        positions *= scale
        print(f"Normalized layout scale: factor={float(scale):.4f}, "
              f"extent={(positions.max(axis=0) - positions.min(axis=0)).round(2)}")
    return positions


def compute_embedding_clusters(
    matrix: np.ndarray,
    min_cluster_size: int = 400,
    n_clusters: int = 250,
    pca_dim: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """Cluster in the high-dim embedding space (semantically meaningful)
    instead of on the 3D layout positions.

    KMeans on 3D positions of a distance-faithful (ball-like) layout produces
    even Voronoi wedges — clusters that are spatial slices, not topics.
    HDBSCAN on the embeddings finds density-based topic clusters of varying
    sizes plus a noise set (label -1). Embeddings are unit-normalized, so
    euclidean distance on them is monotonically equivalent to cosine.

    PCA first: HDBSCAN scales poorly in 3840-dim, and the top ~50 components
    carry the cluster structure.

    Returns:
        labels: (N,) int32 array of cluster labels (-1 = noise)
    """
    import hdbscan
    from sklearn.decomposition import PCA

    pca_dim = min(pca_dim, matrix.shape[1], len(matrix))
    print(f"Reducing to {pca_dim} dims via PCA for clustering...")
    X = PCA(n_components=pca_dim, random_state=random_state).fit_transform(matrix)

    print(f"Computing HDBSCAN clusters in embedding space: "
          f"min_cluster_size={min_cluster_size}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)

    n_found = len(set(labels) - {-1})
    noise_ratio = float((labels == -1).mean())
    # Homogeneous corpora have no density valleys: HDBSCAN degenerates to a
    # handful of mega-clusters plus a huge noise set. That is not usable for
    # presentation — fall back to fine-grained KMeans in the same embedding
    # space (RealDeepResearch-style granularity, ~hundreds of clusters).
    if n_found < 10 or noise_ratio > 0.3:
        from sklearn.cluster import KMeans

        print(f"HDBSCAN degenerate ({n_found} clusters, {noise_ratio:.0%} "
              f"noise) — falling back to KMeans(n_clusters={n_clusters}) "
              f"in embedding space")
        labels = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        ).fit_predict(X)
        n_found = len(set(labels))
        noise_ratio = 0.0

    n_clusters_out = len(set(labels))
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters_out} clusters, {n_noise} noise points "
          f"({n_noise / len(labels):.1%})")
    # Remap noise (-1) to a dedicated trailing id: the binary layout stores
    # clusters as uint16, where -1 would wrap to 65535.
    labels = labels.copy()
    labels[labels == -1] = labels.max() + 1
    return labels.astype(np.int32)


def compute_clusters(
    positions: np.ndarray, min_cluster_size: int = 50, n_clusters: int = 50
) -> np.ndarray:
    """Cluster 3D positions using HDBSCAN.

    Returns:
        labels: (N,) int32 array of cluster labels (-1 = noise)
    """
    from sklearn.cluster import KMeans

    # HDBSCAN can be slow on large datasets; fall back to KMeans
    try:
        import hdbscan
        print(f"Computing HDBSCAN clusters: min_cluster_size={min_cluster_size}")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(positions)
    except ImportError:
        # Fallback to KMeans if HDBSCAN not available
        n_clusters = min(n_clusters, len(positions) // 100)
        print(f"HDBSCAN not available, using KMeans with n_clusters={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(positions)

    unique_labels = len(set(labels))
    print(f"Found {unique_labels} clusters")
    return labels.astype(np.int32)


def write_binary_layout(
    output_path: Path,
    positions: np.ndarray,
    clusters: np.ndarray,
    entity_names: list[str],
) -> None:
    """Write binary layout file in RealDeepResearch format.

    Format:
        - Float32Array: positions (N * 3)
        - Uint16Array: clusters (N)
        - Uint32Array: id_offsets (N + 1)
        - Uint16Array: id_data (variable length, concatenated entity names)
    """
    print(f"Writing binary layout to {output_path}...")

    # Prepare ID data (concatenated entity names as UTF-16)
    id_data = []
    id_offsets = [0]
    for name in entity_names:
        # Encode as UTF-16 code units (without BOM)
        encoded = name.encode("utf-16-le")
        code_units = len(encoded) // 2
        id_data.extend(struct.unpack(f"<{code_units}H", encoded))
        id_offsets.append(len(id_data))

    # Write to file
    with open(output_path, "wb") as f:
        # Positions
        f.write(positions.tobytes())
        # Clusters
        f.write(clusters.astype(np.uint16).tobytes())
        # ID offsets
        f.write(np.array(id_offsets, dtype=np.uint32).tobytes())
        # ID data
        f.write(np.array(id_data, dtype=np.uint16).tobytes())

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Binary layout written: {file_size_mb:.2f} MB")


def write_metadata(
    output_path: Path,
    entity_names: list[str],
    metadata: list[dict],
    clusters: np.ndarray,
    edges: list[list[int]],
    positions: np.ndarray,
) -> None:
    """Write JSON metadata file with entity info and graph structure."""
    print(f"Writing metadata to {output_path}...")

    # Build entity list with positions and clusters
    entities = []
    for i, (name, meta) in enumerate(zip(entity_names, metadata)):
        entities.append({
            "id": i,
            "name": name,
            "cluster": int(clusters[i]),
            "x": float(positions[i, 0]),
            "y": float(positions[i, 1]),
            "z": float(positions[i, 2]),
            "description": meta.get("content", ""),
            "source_id": meta.get("source_id", ""),
            "file_path": meta.get("file_path", ""),
        })

    output = {
        "version": "1.0",
        "entity_count": len(entities),
        "edge_count": len(edges),
        "cluster_count": int(len(set(clusters))),
        "entities": entities,
        "edges": edges,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Metadata written: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Compute 3D layout for knowledge graph")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help="LightRAG working directory (contains VDB and graph files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to working-dir)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (larger values spread clusters apart)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=50,
        help="Number of clusters for the KMeans fallback",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "mds", "sfdp", "hier"],
        default="umap",
        help="Layout method: umap (local structure, pretty clusters), "
        "mds (landmark MDS, preserves relative distances), "
        "sfdp (force-directed on the graph topology: hairball on "
        "hub-dominated graphs), or hier (hierarchical: cluster centroids "
        "via sfdp as sectors, within-cluster PCA lobes — the "
        "neuron/brain-sector look)",
    )
    parser.add_argument(
        "--cluster-tightness",
        type=float,
        default=0.35,
        help="Lobe radius vs sector spacing for --method hier "
        "(smaller = tighter, more separated lobes)",
    )
    parser.add_argument(
        "--n-landmarks",
        type=int,
        default=3000,
        help="Number of landmarks for --method mds (SMACOF runs on these)",
    )
    parser.add_argument(
        "--dist-gamma",
        type=float,
        default=1.0,
        help="Distance exponent for --method mds (d ** gamma). gamma < 1 "
        "expands local structure visually; rank fidelity is unchanged "
        "because the transform is monotonic.",
    )
    parser.add_argument(
        "--cluster-space",
        choices=["layout", "embedding"],
        default="layout",
        help="Cluster the 3D layout positions (layout) or the original "
        "embeddings (embedding: PCA + HDBSCAN, varying-size topic clusters "
        "instead of even spatial wedges).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=50,
        help="Minimum cluster size for HDBSCAN",
    )
    args = parser.parse_args()

    working_dir = args.working_dir
    output_dir = args.output_dir or working_dir

    # Paths
    vdb_path = working_dir / "vdb_entities.json"
    graphml_path = working_dir / "graph_chunk_entity_relation.graphml"
    bin_output = output_dir / "graph3d_layout.bin"
    meta_output = output_dir / "graph3d_layout_meta.json"

    if not vdb_path.exists():
        raise FileNotFoundError(f"VDB file not found: {vdb_path}")
    if not graphml_path.exists():
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")

    start_time = time.time()

    # Step 1: Load entity vectors
    entity_names, matrix, metadata = load_entity_vectors(vdb_path)

    # Step 2: Compute 3D layout
    clusters = None
    if args.method == "hier":
        # Hierarchical layout is built ON the embedding-space clusters
        # (sectors), so clustering must happen first.
        clusters = compute_embedding_clusters(
            matrix,
            min_cluster_size=args.min_cluster_size,
            n_clusters=args.n_clusters,
        )
        positions = compute_hierarchical_layout(
            graphml_path, entity_names, matrix, clusters,
            cluster_tightness=args.cluster_tightness,
        )
    elif args.method == "sfdp":
        positions = compute_sfdp_layout(graphml_path, entity_names, matrix)
    elif args.method == "mds":
        positions = compute_mds_layout(
            matrix, n_landmarks=args.n_landmarks, dist_gamma=args.dist_gamma
        )
    else:
        positions = compute_umap_layout(
            matrix, n_neighbors=args.n_neighbors, min_dist=args.min_dist
        )

    # Step 3: Compute clusters (skipped when the layout consumed them)
    if clusters is None:
        if args.cluster_space == "embedding":
            clusters = compute_embedding_clusters(
                matrix,
                min_cluster_size=args.min_cluster_size,
                n_clusters=args.n_clusters,
            )
        else:
            clusters = compute_clusters(
                positions,
                min_cluster_size=args.min_cluster_size,
                n_clusters=args.n_clusters,
            )

    # Step 4: Load graph structure and remap edges to VDB indices
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)
    vdb_name_to_idx = {name: i for i, name in enumerate(entity_names)}
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    remapped_edges = []
    for u, v in G.edges():
        if u in vdb_name_to_idx and v in vdb_name_to_idx:
            remapped_edges.append([vdb_name_to_idx[u], vdb_name_to_idx[v]])

    print(f"Remapped {len(remapped_edges)} edges to VDB indices")

    # Step 6: Write outputs
    write_binary_layout(bin_output, positions, clusters, entity_names)
    write_metadata(meta_output, entity_names, metadata, clusters, remapped_edges, positions)

    elapsed = time.time() - start_time
    print(f"\nLayout computation completed in {elapsed:.2f} seconds")
    print("Output files:")
    print(f"  Binary layout: {bin_output}")
    print(f"  Metadata: {meta_output}")


if __name__ == "__main__":
    main()
