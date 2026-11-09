"""Authority- and effective-time-aware evidence policy (opt-in).

When a clause was amended, the answer to "what does it say" depends on the date
asked about, not on which chunk a retriever ranked first. This package decides
which *version* applies, then hands the pipeline a closed evidence group per
clause: the version in force plus the wording it replaced, each span traceable
to a real chunk.

Storage is accessed only through the supplied chunk reader; this package does
not call a model. :mod:`lightrag.operate` passes seed chunk IDs and a
``text_chunks`` reader, and receives a typed outcome. The policy is off unless
a request names it via
``QueryParam.evidence_policy``.

Contracts, limits and the residue this version accepts:
``docs/design/EvidencePolicyContract.md``.
"""

from .binding import (
    Binding,
    BindingProblem,
    BoundSpan,
    ChunkReader,
    StorageBackendError,
    bind_versions,
)
from .catalog import catalog_from_mapping, load_catalog
from .contracts import (
    AuthorityPolicy,
    Catalog,
    CatalogDocument,
    CatalogError,
    CatalogVersion,
    ChunkRef,
    ReasonCode,
    RequestError,
    ResolutionStatus,
    ScopeKey,
    canonical_json,
    parse_full_date,
    sha256_text,
)
from .groups import (
    OLD_SPAN_LABEL,
    SELECTED_SPAN_LABEL,
    BudgetDecision,
    EvidenceGroup,
    EvidenceSpan,
    GroupError,
    apply_budget,
    build_groups,
    citations_of,
    rank_groups,
    render_prompt,
    verify_citations,
)
from .resolver import (
    Resolution,
    ScopeResolution,
    VersionVerdict,
    resolve,
    resolve_scope,
)
from .runtime import (
    MAX_EXTRA_CHUNKS,
    MAX_SCOPES,
    MAX_VERSIONS_PER_SCOPE,
    POLICY_VERSIONS,
    SUPPORTED_MODES,
    EvidenceOutcome,
    EvidencePolicyConfig,
    build_policy_config,
    failure_content,
    run_evidence_policy,
    scopes_from_seeds,
)

__all__ = [
    "MAX_EXTRA_CHUNKS",
    "MAX_SCOPES",
    "MAX_VERSIONS_PER_SCOPE",
    "OLD_SPAN_LABEL",
    "POLICY_VERSIONS",
    "SELECTED_SPAN_LABEL",
    "SUPPORTED_MODES",
    "AuthorityPolicy",
    "Binding",
    "BindingProblem",
    "BoundSpan",
    "BudgetDecision",
    "Catalog",
    "CatalogDocument",
    "CatalogError",
    "CatalogVersion",
    "ChunkReader",
    "ChunkRef",
    "EvidenceGroup",
    "EvidenceOutcome",
    "EvidencePolicyConfig",
    "EvidenceSpan",
    "GroupError",
    "ReasonCode",
    "RequestError",
    "Resolution",
    "ResolutionStatus",
    "ScopeKey",
    "ScopeResolution",
    "StorageBackendError",
    "VersionVerdict",
    "apply_budget",
    "bind_versions",
    "build_groups",
    "build_policy_config",
    "canonical_json",
    "catalog_from_mapping",
    "citations_of",
    "failure_content",
    "load_catalog",
    "parse_full_date",
    "rank_groups",
    "render_prompt",
    "resolve",
    "resolve_scope",
    "run_evidence_policy",
    "scopes_from_seeds",
    "sha256_text",
    "verify_citations",
]
