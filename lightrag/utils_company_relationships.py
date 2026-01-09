"""
Advanced Company Relationship Inference

This module provides advanced inference capabilities for detecting:
1. Competitor relationships between organizations
2. Partnership/collaboration relationships
3. Supply chain relationships
4. Market overlap analysis

It uses contextual signals from document metadata and entity co-occurrence patterns
to infer implicit business relationships.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import re


# Competitor indicators in text
COMPETITOR_INDICATORS = {
    "explicit": [
        "competitor", "competing", "rival", "versus", "vs", "alternative to",
        "competes with", "competition", "competitive", "market leader",
        "market share", "compared to", "comparison"
    ],
    "implicit": [
        "similar to", "like", "also offers", "alternative", "other providers",
        "same market", "same industry", "same sector", "same space"
    ]
}

# Partnership indicators
PARTNERSHIP_INDICATORS = {
    "explicit": [
        "partner", "partnership", "collaboration", "joint venture",
        "strategic alliance", "works with", "teams up", "cooperates",
        "co-develops", "integrates with", "certified partner"
    ],
    "implicit": [
        "ecosystem", "network", "member", "alliance", "consortium",
        "reseller", "distributor", "together with"
    ]
}

# Supply chain indicators
SUPPLY_CHAIN_INDICATORS = [
    "supplier", "vendor", "customer", "client", "provides to",
    "supplies", "procurement", "sourcing from", "buys from"
]


def extract_main_company_from_summary(
    chunk_content: str,
    all_organizations: Set[str]
) -> Optional[str]:
    """
    Extract the main company name from a document summary section.

    The main company is typically mentioned in phrases like:
    - "Summary: [Company Name] is a..."
    - "Entity: [Company Name]"
    - "Company: [Company Name]"

    Args:
        chunk_content: The text content of the chunk
        all_organizations: Set of all known organization names

    Returns:
        Main company name if found, None otherwise
    """
    # Look for explicit entity/company markers
    patterns = [
        r"(?:^|\n)Entity\s*:\s*([^\n]+)",
        r"(?:^|\n)Company\s*:\s*([^\n]+)",
        r"(?:^|\n)Organization\s*:\s*([^\n]+)",
        r"(?:^|\n)Summary\s*:\s*([^\n.]+?)(?:\sis\s|\sare\s)",
    ]

    for pattern in patterns:
        match = re.search(pattern, chunk_content, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Check if this matches any known organization
            for org in all_organizations:
                if org.lower() in candidate.lower() or candidate.lower() in org.lower():
                    return org

    return None


def analyze_relationship_context(
    entity1: str,
    entity2: str,
    shared_chunks: List[str],
    chunk_contents: Dict[str, str]
) -> Dict[str, Any]:
    """
    Analyze the context in which two entities co-occur to determine relationship type.

    Args:
        entity1: First entity name
        entity2: Second entity name
        shared_chunks: List of chunk IDs where both entities appear
        chunk_contents: Mapping of chunk ID to chunk text content

    Returns:
        Dictionary with relationship analysis:
        - is_competitor: bool
        - is_partner: bool
        - is_supply_chain: bool
        - confidence: float (0-1)
        - evidence: list of text snippets
    """
    competitor_signals = 0
    partner_signals = 0
    supply_chain_signals = 0
    evidence = []

    for chunk_id in shared_chunks:
        content = chunk_contents.get(chunk_id, "")
        if not content:
            continue

        content_lower = content.lower()

        # Look for entities mentioned together in sentences
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Check if both entities are mentioned in the same sentence
            if entity1.lower() in sentence_lower and entity2.lower() in sentence_lower:
                # Check for competitor indicators
                for indicator in COMPETITOR_INDICATORS["explicit"]:
                    if indicator in sentence_lower:
                        competitor_signals += 2
                        evidence.append(f"Competitor signal: '{sentence.strip()[:100]}...'")
                        break

                for indicator in COMPETITOR_INDICATORS["implicit"]:
                    if indicator in sentence_lower:
                        competitor_signals += 1
                        break

                # Check for partnership indicators
                for indicator in PARTNERSHIP_INDICATORS["explicit"]:
                    if indicator in sentence_lower:
                        partner_signals += 2
                        evidence.append(f"Partnership signal: '{sentence.strip()[:100]}...'")
                        break

                for indicator in PARTNERSHIP_INDICATORS["implicit"]:
                    if indicator in sentence_lower:
                        partner_signals += 1
                        break

                # Check for supply chain indicators
                for indicator in SUPPLY_CHAIN_INDICATORS:
                    if indicator in sentence_lower:
                        supply_chain_signals += 1
                        evidence.append(f"Supply chain signal: '{sentence.strip()[:100]}...'")
                        break

    # Determine relationship type based on signals
    total_signals = competitor_signals + partner_signals + supply_chain_signals

    if total_signals == 0:
        # No explicit signals - use heuristics
        return {
            "is_competitor": False,
            "is_partner": False,
            "is_supply_chain": False,
            "confidence": 0.0,
            "evidence": []
        }

    # Calculate confidence based on signal strength
    max_signals = max(competitor_signals, partner_signals, supply_chain_signals)
    confidence = min(max_signals / 5.0, 1.0)  # Max confidence at 5+ signals

    return {
        "is_competitor": competitor_signals > partner_signals and competitor_signals > supply_chain_signals,
        "is_partner": partner_signals > competitor_signals and partner_signals > supply_chain_signals,
        "is_supply_chain": supply_chain_signals > max(competitor_signals, partner_signals),
        "confidence": confidence,
        "competitor_signals": competitor_signals,
        "partner_signals": partner_signals,
        "supply_chain_signals": supply_chain_signals,
        "evidence": evidence[:3]  # Top 3 pieces of evidence
    }


def infer_company_relationships(
    all_nodes: Dict[str, List[Dict]],
    all_edges: Dict[Tuple[str, str], List[Dict]],
    chunk_contents: Optional[Dict[str, str]] = None,
    min_cooccurrence: int = 2,
    confidence_threshold: float = 0.5,
    enable_inference: bool = True,
) -> Tuple[Dict[Tuple[str, str], List[Dict]], Dict[str, Any]]:
    """
    Infer advanced company relationships (competitors, partnerships) based on context.

    This function goes beyond basic co-occurrence by analyzing:
    1. Document structure (summary sections, main entities)
    2. Contextual indicators (competitor/partnership language)
    3. Entity mention patterns

    Args:
        all_nodes: Dictionary of entity names to entity node data
        all_edges: Existing relationships (edges) between entities
        chunk_contents: Optional mapping of chunk IDs to chunk text content
        min_cooccurrence: Minimum co-occurrence count (default: 2)
        confidence_threshold: Minimum confidence for inference (0-1, default: 0.5)
        enable_inference: Whether to enable this inference

    Returns:
        Tuple of (updated_edges, statistics)
    """
    if not enable_inference:
        return all_edges, {}

    # Filter to only organization entities
    organizations = {
        name: nodes for name, nodes in all_nodes.items()
        if any(node.get("entity_type", "").lower() in ["organization", "company", "organisation"]
               for node in nodes)
    }

    if len(organizations) < 2:
        return all_edges, {"message": "Less than 2 organizations found"}

    # Build co-occurrence map for organizations
    cooccurrence_map = defaultdict(lambda: defaultdict(set))
    chunk_to_orgs = defaultdict(set)
    main_companies = {}  # chunk_id -> main company name

    for org_name, org_list in organizations.items():
        for org_data in org_list:
            chunk_id = org_data.get("source_id")
            if not chunk_id:
                continue

            chunk_to_orgs[chunk_id].add(org_name)

            # Try to identify main company from chunk content
            if chunk_contents and chunk_id in chunk_contents:
                main_company = extract_main_company_from_summary(
                    chunk_contents[chunk_id],
                    set(organizations.keys())
                )
                if main_company:
                    main_companies[chunk_id] = main_company

            # Find other organizations in the same chunk
            for other_org, other_org_list in organizations.items():
                if other_org == org_name:
                    continue

                for other_data in other_org_list:
                    other_chunk_id = other_data.get("source_id")
                    if chunk_id == other_chunk_id:
                        cooccurrence_map[org_name][other_org].add(chunk_id)

    # Infer relationships
    updated_edges = dict(all_edges)
    stats = {
        "competitor_relationships": 0,
        "partnership_relationships": 0,
        "supply_chain_relationships": 0,
        "total_analyzed": 0,
        "skipped_low_confidence": 0,
        "skipped_existing": 0,
    }

    for org1 in sorted(organizations.keys()):
        if org1 not in cooccurrence_map:
            continue

        for org2, chunk_ids in cooccurrence_map[org1].items():
            count = len(chunk_ids)

            if count < min_cooccurrence:
                continue

            stats["total_analyzed"] += 1

            # Check if relationship already exists
            edge_key = tuple(sorted([org1, org2]))
            if edge_key in updated_edges:
                stats["skipped_existing"] += 1
                continue

            # Analyze relationship context if chunk contents available
            if chunk_contents:
                analysis = analyze_relationship_context(
                    org1, org2, list(chunk_ids), chunk_contents
                )

                if analysis["confidence"] < confidence_threshold:
                    stats["skipped_low_confidence"] += 1
                    continue

                # Create relationship based on analysis
                if analysis["is_competitor"]:
                    rel_type = "competitor"
                    keywords = "competes_with, competitor, market_rival"
                    description = (
                        f"{org1} and {org2} are competitors in the market "
                        f"(confidence: {analysis['confidence']:.2f}, co-occurs {count} times)"
                    )
                    stats["competitor_relationships"] += 1

                elif analysis["is_partner"]:
                    rel_type = "partnership"
                    keywords = "partners_with, collaborates_with, strategic_alliance"
                    description = (
                        f"{org1} and {org2} have a partnership or collaboration "
                        f"(confidence: {analysis['confidence']:.2f}, co-occurs {count} times)"
                    )
                    stats["partnership_relationships"] += 1

                elif analysis["is_supply_chain"]:
                    rel_type = "supply_chain"
                    keywords = "business_relationship, supply_chain, vendor_customer"
                    description = (
                        f"{org1} and {org2} have a supply chain relationship "
                        f"(confidence: {analysis['confidence']:.2f}, co-occurs {count} times)"
                    )
                    stats["supply_chain_relationships"] += 1

                else:
                    # Generic business association
                    rel_type = "business_association"
                    keywords = "business_association, ecosystem_connection"
                    description = (
                        f"{org1} and {org2} have a business association "
                        f"(co-occurs {count} times)"
                    )
                    stats["partnership_relationships"] += 1

                # Create inferred relationship
                inferred_rel = {
                    "src_id": org1,
                    "tgt_id": org2,
                    "weight": count * analysis["confidence"],  # Weight by confidence
                    "keywords": keywords,
                    "description": description,
                    "source_id": "inferred_company_relationship",
                    "inferred": True,
                    "inference_method": "contextual_company_analysis",
                    "cooccurrence_count": count,
                    "confidence": analysis["confidence"],
                    "relationship_type": rel_type,
                    "evidence": analysis.get("evidence", []),
                }

                updated_edges[edge_key] = [inferred_rel]

            else:
                # No chunk contents - use basic co-occurrence only
                # Default to business_association
                inferred_rel = {
                    "src_id": org1,
                    "tgt_id": org2,
                    "weight": count,
                    "keywords": "business_association, ecosystem_connection",
                    "description": f"{org1} and {org2} have a business association (co-occurs {count} times)",
                    "source_id": "inferred_company_relationship",
                    "inferred": True,
                    "inference_method": "cooccurrence_only",
                    "cooccurrence_count": count,
                }

                updated_edges[edge_key] = [inferred_rel]
                stats["partnership_relationships"] += 1

    return updated_edges, stats


def enable_company_relationship_inference_in_config() -> dict:
    """
    Returns addon_params configuration to enable company relationship inference.

    Usage:
        config = {
            ...
            "addon_params": {
                **enable_company_relationship_inference_in_config(),
                # ... other params
            }
        }
    """
    return {
        "enable_company_relationship_inference": True,
        "company_inference_min_cooccurrence": 2,
        "company_inference_confidence_threshold": 0.5,
    }
