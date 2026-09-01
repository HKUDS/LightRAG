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
        d_sq = (1.0 - np.clip(sims, -1, 1)) ** 2
        b = -0.5 * (d_sq - col_means)
        positions[start:end] = (b @ Y_pinv.T).astype(np.float32)
        if (start // chunk) % 5 == 0:
            print(f"  Placed {end}/{N} points...")

    print(f"MDS output shape: {positions.shape}")
    return positions


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
        choices=["umap", "mds"],
        default="umap",
        help="Layout method: umap (local structure, pretty clusters) or "
        "mds (landmark MDS, preserves relative distances)",
    )
    parser.add_argument(
        "--n-landmarks",
        type=int,
        default=3000,
        help="Number of landmarks for --method mds (SMACOF runs on these)",
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
    if args.method == "mds":
        positions = compute_mds_layout(matrix, n_landmarks=args.n_landmarks)
    else:
        positions = compute_umap_layout(
            matrix, n_neighbors=args.n_neighbors, min_dist=args.min_dist
        )

    # Step 3: Compute clusters
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
    print(f"Output files:")
    print(f"  Binary layout: {bin_output}")
    print(f"  Metadata: {meta_output}")


if __name__ == "__main__":
    main()
