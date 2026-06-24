from __future__ import annotations

from lightrag.kb_iteration.profiles import influenza_rules


def test_influenza_profile_maps_typed_disease_to_precise_pathogen() -> None:
    assert (
        influenza_rules.typed_pathogen_for_disease("influenza A")
        == "influenza A virus"
    )
    assert (
        influenza_rules.typed_pathogen_for_disease("influenza b")
        == "influenza B virus"
    )
    assert influenza_rules.typed_pathogen_for_disease("common cold") == ""


def test_influenza_profile_detects_bare_lab_marker_orders() -> None:
    assert influenza_rules.looks_like_bare_lab_marker_order(
        "influenza",
        "creatinine",
    )
    assert not influenza_rules.looks_like_bare_lab_marker_order(
        "pneumonia",
        "creatinine",
    )


def test_influenza_profile_detects_nonspecific_diagnostic_shortcuts() -> None:
    assert influenza_rules.looks_like_nonspecific_diagnostic_criterion(
        "influenza",
        "MRI",
    )
    assert influenza_rules.looks_like_nonspecific_evidence_supporting_influenza(
        "MRI",
        "influenza",
    )
    assert not influenza_rules.looks_like_nonspecific_diagnostic_criterion(
        "influenza",
        "RT-PCR",
    )


def test_influenza_profile_detects_avoidable_zanamivir_review() -> None:
    assert influenza_rules.looks_like_avoidable_zanamivir_review(
        "zanamivir review for asthma and children dosing"
    )
    assert not influenza_rules.looks_like_avoidable_zanamivir_review(
        "oseltamivir review for adults"
    )


def test_influenza_profile_detects_parent_to_subtype_taxonomy_direction() -> None:
    assert influenza_rules.looks_like_parent_to_subtype_is_a(
        "influenza virus",
        "influenza A H1N1 virus",
    )
    assert not influenza_rules.looks_like_parent_to_subtype_is_a(
        "influenza A H1N1 virus",
        "influenza virus",
    )
