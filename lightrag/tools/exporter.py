"""
LightRAG Data Exporter/Importer

This module provides functionality to export data from a LightRAG instance 
and import it into another instance, enabling:
1. Decoupling insert operations from inference
2. Migration between different storage backends
3. Backup and restoration of LightRAG data
"""

import os
import json
import asyncio
import numpy as np
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Any, Optional

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id


class LightRAGExporter:
    """Storage-agnostic exporter/importer for LightRAG data"""
    
    @staticmethod
    async def export_data(
        lightrag_instance: LightRAG, 
        output_dir: str,
        include_cache: bool = False,
        compress: bool = True
    ) -> str:
        """
        Export all data from a LightRAG instance to a directory.
        
        Args:
            lightrag_instance: The LightRAG instance to export data from
            output_dir: Directory to store exported data
            include_cache: Whether to include LLM response cache
            compress: Whether to compress the output files
        
        Returns:
            Path to the export directory containing all data
        """
        # Create timestamp-based export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(output_dir, f"lightrag_export_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save configuration metadata
        config = {
            "export_time": timestamp,
            "namespace_prefix": lightrag_instance.namespace_prefix,
            "storage_types": {
                "kv_storage": lightrag_instance.kv_storage,
                "vector_storage": lightrag_instance.vector_storage,
                "graph_storage": lightrag_instance.graph_storage,
                "doc_status_storage": lightrag_instance.doc_status_storage
            },
            "embedding_dim": lightrag_instance.embedding_func.embedding_dim,
            "include_cache": include_cache
        }
        
        with open(os.path.join(export_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Create tasks for all export operations
        tasks = [
            LightRAGExporter._export_kv_store(lightrag_instance, export_dir, include_cache),
            LightRAGExporter._export_vector_store(lightrag_instance, export_dir),
            LightRAGExporter._export_graph_store(lightrag_instance, export_dir),
            LightRAGExporter._export_doc_status(lightrag_instance, export_dir)
        ]
        
        # Run all export tasks concurrently
        await asyncio.gather(*tasks)
        
        print(f"Export completed to {export_dir}")
        return export_dir
    
    @staticmethod
    async def _export_kv_store(
        lightrag_instance: LightRAG, 
        export_dir: str,
        include_cache: bool
    ) -> None:
        """Export all KV stores"""
        kv_dir = os.path.join(export_dir, "kv_stores")
        os.makedirs(kv_dir, exist_ok=True)
        
        # Export full documents
        full_docs = await lightrag_instance.full_docs.get_all()
        with open(os.path.join(kv_dir, "full_docs.json"), "w") as f:
            json.dump(full_docs, f)
        
        # Export text chunks
        text_chunks = await lightrag_instance.text_chunks.get_all()
        with open(os.path.join(kv_dir, "text_chunks.json"), "w") as f:
            json.dump(text_chunks, f)
        
        # Export LLM cache if requested
        if include_cache:
            llm_cache = await lightrag_instance.llm_response_cache.get_all()
            with open(os.path.join(kv_dir, "llm_response_cache.json"), "w") as f:
                json.dump(llm_cache, f)
    
    @staticmethod
    async def _export_vector_store(
        lightrag_instance: LightRAG, 
        export_dir: str
    ) -> None:
        """Export all vector stores with vectors and metadata"""
        vector_dir = os.path.join(export_dir, "vector_stores")
        os.makedirs(vector_dir, exist_ok=True)
        
        # Helper function to handle vector data serialization
        async def export_vector_db(vdb, name):
            # Get all IDs first by querying with a broad query
            # This approach works regardless of underlying storage
            results = await vdb.query("", top_k=10000)  # Query everything
            
            # Directly get all items by ID for complete data
            if results:
                all_ids = [item["id"] for item in results]
                vectors_data = await vdb.get_by_ids(all_ids)
                
                # Need special handling for numpy arrays in vectors
                serializable_data = []
                for item in vectors_data:
                    item_copy = dict(item)
                    # Convert numpy arrays to lists for JSON serialization
                    if "__vector__" in item_copy:
                        if isinstance(item_copy["__vector__"], np.ndarray):
                            item_copy["__vector__"] = item_copy["__vector__"].tolist()
                    serializable_data.append(item_copy)
                
                with open(os.path.join(vector_dir, f"{name}.json"), "w") as f:
                    json.dump(serializable_data, f)
        
        # Export all three vector stores
        await export_vector_db(lightrag_instance.entities_vdb, "entities_vdb")
        await export_vector_db(lightrag_instance.relationships_vdb, "relationships_vdb")
        await export_vector_db(lightrag_instance.chunks_vdb, "chunks_vdb")
    
    @staticmethod
    async def _export_graph_store(
        lightrag_instance: LightRAG, 
        export_dir: str
    ) -> None:
        """Export graph structure: nodes and edges"""
        graph_dir = os.path.join(export_dir, "graph_store")
        os.makedirs(graph_dir, exist_ok=True)
        
        # Get all node labels
        all_labels = await lightrag_instance.chunk_entity_relation_graph.get_all_labels()
        
        # Export nodes
        nodes = {}
        for node_id in all_labels:
            node_data = await lightrag_instance.chunk_entity_relation_graph.get_node(node_id)
            if node_data:
                nodes[node_id] = node_data
        
        with open(os.path.join(graph_dir, "nodes.json"), "w") as f:
            json.dump(nodes, f)
        
        # Export edges
        edges = []
        seen_edges = set()  # To avoid duplicate edges
        
        for source_id in all_labels:
            node_edges = await lightrag_instance.chunk_entity_relation_graph.get_node_edges(source_id)
            if node_edges:
                for src, tgt in node_edges:
                    edge_key = f"{src}:{tgt}"
                    if edge_key not in seen_edges:
                        edge_data = await lightrag_instance.chunk_entity_relation_graph.get_edge(src, tgt)
                        if edge_data:
                            edges.append({
                                "source": src,
                                "target": tgt,
                                "data": edge_data
                            })
                            seen_edges.add(edge_key)
        
        with open(os.path.join(graph_dir, "edges.json"), "w") as f:
            json.dump(edges, f)
        
        # Export complete knowledge graph (for verification)
        kg = await lightrag_instance.get_knowledge_graph("*", max_depth=5, max_nodes=100000)
        
        # Convert knowledge graph to serializable format
        kg_dict = {}
        try:
            # Try to use asdict if it's a dataclass
            kg_dict = asdict(kg)
        except TypeError:
            # Fall back to manual conversion if not a dataclass
            kg_dict = {
                "nodes": [{"id": node.id, **vars(node)} for node in getattr(kg, "nodes", [])],
                "edges": [{"source": edge.source, "target": edge.target, **vars(edge)} 
                         for edge in getattr(kg, "edges", [])]
            }
        
        with open(os.path.join(graph_dir, "knowledge_graph.json"), "w") as f:
            json.dump(kg_dict, f)
    
    @staticmethod
    async def _export_doc_status(
        lightrag_instance: LightRAG, 
        export_dir: str
    ) -> None:
        """Export document processing status"""
        # Get status counts
        status_counts = await lightrag_instance.doc_status.get_status_counts()
        
        # Create status directory
        status_dir = os.path.join(export_dir, "doc_status")
        os.makedirs(status_dir, exist_ok=True)
        
        # Export status counts
        with open(os.path.join(status_dir, "status_counts.json"), "w") as f:
            json.dump(status_counts, f)
        
        # Export document statuses by status type
        all_statuses = {}
        for status in [DocStatus.PENDING, DocStatus.PROCESSING, DocStatus.PROCESSED, DocStatus.FAILED]:
            docs = await lightrag_instance.doc_status.get_docs_by_status(status)
            for doc_id, doc_status in docs.items():
                # Convert DocProcessingStatus to dict for serialization
                all_statuses[doc_id] = asdict(doc_status)
        
        with open(os.path.join(status_dir, "doc_statuses.json"), "w") as f:
            json.dump(all_statuses, f)
    
    @staticmethod
    async def import_data(
        lightrag_instance: LightRAG,
        import_dir: str,
        include_cache: bool = False
    ) -> None:
        """
        Import data from an export directory into a LightRAG instance.
        
        Args:
            lightrag_instance: The LightRAG instance to import data into
            import_dir: Directory containing exported data
            include_cache: Whether to import LLM response cache
        """
        # Load configuration metadata
        config_path = os.path.join(import_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Invalid import directory. Missing config.json in {import_dir}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Validate compatibility
        current_embed_dim = lightrag_instance.embedding_func.embedding_dim
        export_embed_dim = config.get("embedding_dim")
        if export_embed_dim and current_embed_dim != export_embed_dim:
            print(f"WARNING: Embedding dimensions mismatch. Export: {export_embed_dim}, Current: {current_embed_dim}")
        
        # Create tasks for all import operations
        tasks = [
            LightRAGExporter._import_kv_store(lightrag_instance, import_dir, include_cache),
            LightRAGExporter._import_vector_store(lightrag_instance, import_dir),
            LightRAGExporter._import_graph_store(lightrag_instance, import_dir),
            LightRAGExporter._import_doc_status(lightrag_instance, import_dir)
        ]
        
        # Run all import tasks concurrently
        await asyncio.gather(*tasks)
        
        # Finalize all storages to ensure persistence
        await lightrag_instance.finalize_storages()
        
        print(f"Import completed from {import_dir}")
    
    @staticmethod
    async def _import_kv_store(
        lightrag_instance: LightRAG,
        import_dir: str,
        include_cache: bool
    ) -> None:
        """Import all KV stores"""
        kv_dir = os.path.join(import_dir, "kv_stores")
        
        # Import full documents
        full_docs_path = os.path.join(kv_dir, "full_docs.json")
        if os.path.exists(full_docs_path):
            with open(full_docs_path, "r") as f:
                full_docs = json.load(f)
            await lightrag_instance.full_docs.upsert(full_docs)
        
        # Import text chunks
        text_chunks_path = os.path.join(kv_dir, "text_chunks.json")
        if os.path.exists(text_chunks_path):
            with open(text_chunks_path, "r") as f:
                text_chunks = json.load(f)
            await lightrag_instance.text_chunks.upsert(text_chunks)
        
        # Import LLM cache if requested
        if include_cache:
            cache_path = os.path.join(kv_dir, "llm_response_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    llm_cache = json.load(f)
                await lightrag_instance.llm_response_cache.upsert(llm_cache)
    
    @staticmethod
    async def _import_vector_store(
        lightrag_instance: LightRAG,
        import_dir: str
    ) -> None:
        """Import all vector stores with vectors and metadata"""
        vector_dir = os.path.join(import_dir, "vector_stores")
        
        # Helper function to import vector DB data
        async def import_vector_db(vdb, name):
            db_path = os.path.join(vector_dir, f"{name}.json")
            if os.path.exists(db_path):
                with open(db_path, "r") as f:
                    vector_data = json.load(f)
                
                # Process data for import
                data_for_upsert = {}
                for item in vector_data:
                    # Handle vector conversion back to numpy if needed
                    if "__vector__" in item and isinstance(item["__vector__"], list):
                        item["__vector__"] = np.array(item["__vector__"], dtype=np.float32)
                    
                    # Organize by ID for upsert
                    item_id = item.get("__id__") or item.get("id")
                    if item_id:
                        # Create a properly structured item for upsert
                        data_for_upsert[item_id] = {
                            # Include basic necessary fields
                            "content": item.get("content", ""),
                            "__vector__": item.get("__vector__"),
                            # Include all metadata fields
                            **{k: v for k, v in item.items() 
                               if k not in ("__id__", "id", "__vector__") and v is not None}
                        }
                
                # Upsert data
                if data_for_upsert:
                    await vdb.upsert(data_for_upsert)
        
        # Import all three vector stores
        await import_vector_db(lightrag_instance.entities_vdb, "entities_vdb")
        await import_vector_db(lightrag_instance.relationships_vdb, "relationships_vdb")
        await import_vector_db(lightrag_instance.chunks_vdb, "chunks_vdb")
    
    @staticmethod
    async def _import_graph_store(
        lightrag_instance: LightRAG,
        import_dir: str
    ) -> None:
        """Import graph structure: nodes and edges"""
        graph_dir = os.path.join(import_dir, "graph_store")
        
        # Import nodes
        nodes_path = os.path.join(graph_dir, "nodes.json")
        if os.path.exists(nodes_path):
            with open(nodes_path, "r") as f:
                nodes = json.load(f)
            
            for node_id, node_data in nodes.items():
                await lightrag_instance.chunk_entity_relation_graph.upsert_node(node_id, node_data)
        
        # Import edges
        edges_path = os.path.join(graph_dir, "edges.json")
        if os.path.exists(edges_path):
            with open(edges_path, "r") as f:
                edges = json.load(f)
            
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                data = edge.get("data", {})
                
                if source and target:
                    await lightrag_instance.chunk_entity_relation_graph.upsert_edge(source, target, data)
    
    @staticmethod
    async def _import_doc_status(
        lightrag_instance: LightRAG,
        import_dir: str
    ) -> None:
        """Import document processing status"""
        status_dir = os.path.join(import_dir, "doc_status")
        doc_statuses_path = os.path.join(status_dir, "doc_statuses.json")
        
        if os.path.exists(doc_statuses_path):
            with open(doc_statuses_path, "r") as f:
                doc_statuses = json.load(f)
            
            # Convert back to dictionary for upsert
            await lightrag_instance.doc_status.upsert(doc_statuses)


# Synchronous wrapper functions for ease of use
def export_lightrag_data(lightrag_instance, output_dir, include_cache=False, compress=True):
    """
    Export all data from a LightRAG instance to a directory.
    
    Args:
        lightrag_instance: The LightRAG instance to export data from
        output_dir: Directory to store exported data
        include_cache: Whether to include LLM response cache
        compress: Whether to compress the output files
    
    Returns:
        Path to the export directory
    """
    import asyncio
    return asyncio.run(
        LightRAGExporter.export_data(lightrag_instance, output_dir, include_cache, compress)
    )


def import_lightrag_data(lightrag_instance, import_dir, include_cache=False):
    """
    Import data from an export directory into a LightRAG instance.
    
    Args:
        lightrag_instance: The LightRAG instance to import data into
        import_dir: Directory containing exported data
        include_cache: Whether to import LLM response cache
    """
    import asyncio
    asyncio.run(
        LightRAGExporter.import_data(lightrag_instance, import_dir, include_cache)
    ) 