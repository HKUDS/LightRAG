"""FrameRAG: Frame-Semantic Event Hypergraph RAG system."""
from .framerag import FrameRAG
from .rerank import make_reranker, RerankFunc
from . import constants
from . import evaluation  # noqa: F401 — expose evaluation sub-package
# Legacy alias: framerag.eval still works after rename to framerag.evaluation
from . import evaluation as eval  # noqa: F401,A001
from .types import (
    ChunkSchema,
    EntityMentionSchema,
    CanonicalEntitySchema,
    FEAssignment,
    FrameInstanceSchema,
    EventSchema,
    InfoNodeSchema,
    CoreFESchema,
    NonCoreFESchema,
    FrameDefinitionSchema,
    CausalEdgeSchema,
    QuerySignals,
    RetrievalResult,
)

__all__ = [
    "FrameRAG",
    "make_reranker",
    "RerankFunc",
    "ChunkSchema",
    "EntityMentionSchema",
    "CanonicalEntitySchema",
    "FEAssignment",
    "FrameInstanceSchema",
    "EventSchema",
    "InfoNodeSchema",
    "CoreFESchema",
    "NonCoreFESchema",
    "FrameDefinitionSchema",
    "CausalEdgeSchema",
    "QuerySignals",
    "RetrievalResult",
    "constants",
    "evaluation",
    "eval",
]
