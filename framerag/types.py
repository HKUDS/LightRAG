"""Data schemas for FrameRAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkSchema:
    chunk_id: str
    text: str
    source_doc: str
    chunk_index: int
    tokens: int


@dataclass
class EntityMentionSchema:
    mention_id: str
    chunk_id: str
    name: str
    entity_type: str
    description: str
    aliases: list[str]
    salience: str                        # HIGH | MEDIUM | LOW
    embedding: Optional[list[float]] = None
    canonical_id: Optional[str] = None   # set after entity coref


@dataclass
class CanonicalEntitySchema:
    canonical_id: str
    canonical_name: str
    entity_type: str
    descriptions: list[str]             # all descriptions from coreferent mentions
    mention_ids: list[str]
    embedding: Optional[list[float]] = None


@dataclass
class FEAssignment:
    fe_name: str
    filler_id: Optional[str]            # mention_id or info_id; None if MISSING
    filler_type: str                     # ENTITY | VALUE | MISSING
    filler_text: str                     # raw text span
    is_core: bool


@dataclass
class FrameInstanceSchema:
    fi_id: str
    event_id: str
    frame_name: str
    lexical_unit: str                    # trigger.POS e.g. "acquire.v"
    core_assignments: list[FEAssignment]
    noncore_assignments: list[FEAssignment]
    embedding: Optional[list[float]] = None

    def all_assignments(self) -> list[FEAssignment]:
        return self.core_assignments + self.noncore_assignments

    def filled_entity_ids(self) -> list[str]:
        return [
            a.filler_id
            for a in self.all_assignments()
            if a.filler_id and a.filler_type == "ENTITY"
        ]

    def filled_info_ids(self) -> list[str]:
        return [
            a.filler_id
            for a in self.all_assignments()
            if a.filler_id and a.filler_type == "VALUE"
        ]


@dataclass
class EventSchema:
    event_id: str
    chunk_id: str
    trigger: str
    trigger_lemma: str
    trigger_pos: str                     # VERB | NOUN | ADJ
    event_span: str                      # raw text span
    event_description: str               # paraphrased description
    frame_name: str                      # frame evoked by this event
    participant_mention_ids: list[str]   # entity mentions involved (no role yet)
    frame_instance_ids: list[str] = field(default_factory=list)
    canonical_event_id: Optional[str] = None  # set after event coref
    embedding: Optional[list[float]] = None


@dataclass
class InfoNodeSchema:
    info_id: str
    value: str                           # "$3 billion", "2014", "New York"
    info_type: str                       # TIME | PRICE | LOCATION | MANNER | QUANTITY | OTHER
    embedding: Optional[list[float]] = None


@dataclass
class CoreFESchema:
    fe_name: str
    fe_definition: str
    semantic_type: str


@dataclass
class NonCoreFESchema:
    fe_name: str
    fe_definition: str
    semantic_type: str


@dataclass
class FrameDefinitionSchema:
    frame_name: str
    lexical_units: list[str]             # ["acquire.v", "buy.v"]
    frame_definition: str
    core_fes: list[CoreFESchema]
    noncore_fes: list[NonCoreFESchema]
    is_from_framenet: bool = False
    usage_count: int = 0
    embedding: Optional[list[float]] = None  # embed(frame_name + " [SEP] " + definition)

    def core_fe_names(self) -> set[str]:
        return {fe.fe_name for fe in self.core_fes}

    def noncore_fe_names(self) -> set[str]:
        return {fe.fe_name for fe in self.noncore_fes}


@dataclass
class CausalEdgeSchema:
    edge_id: str
    source_event_id: str
    target_event_id: str
    relation_type: str                   # CAUSES | PRECEDES | ENABLES
    confidence: float
    evidence_span: str


@dataclass
class QuerySignals:
    entity_hints: list[str]
    event_hints: list[str]
    frame_hints: str                     # primary frame name or description
    fe_focus: list[str]                  # FE names most relevant to answering query
    temporal_hints: list[str]            # time values to filter on


@dataclass
class RetrievalResult:
    chunks: list[dict]                   # [{chunk_id, text, score}]
    frame_instances: list[dict]          # [{fi_id, frame_name, assignments, score}]
    entities: list[dict]                 # [{canonical_id, name, description, score}]
    causal_chain: list[dict]             # [{source, relation, target}]
