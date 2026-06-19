from __future__ import annotations

import pytest

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import validate_proposal


def test_add_hierarchy_branch_is_a_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    validate_proposal(proposal)


def test_add_hierarchy_branch_without_approval_requires_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=False,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    with pytest.raises(
        ValueError, match="^proposal type add_hierarchy_branch requires approval$"
    ):
        validate_proposal(proposal)
