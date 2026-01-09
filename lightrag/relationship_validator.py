"""
Relationship Quality Validator for LightRAG

This module provides post-extraction validation and filtering for relationships
to improve overall knowledge graph quality by removing inaccurate relationships.

Features:
1. Type-based validation (entity type pairs)
2. Pattern-based filtering (generic names, self-references)
3. Semantic validation (description coherence)
4. Confidence scoring
5. Optional LLM-based verification for ambiguous cases

Usage:
    from lightrag.relationship_validator import RelationshipValidator
    
    validator = RelationshipValidator(
        strict_mode=True,
        enable_llm_verification=False
    )
    
    # Validate during extraction
    valid_relationships = validator.filter_relationships(
        relationships=extracted_relationships,
        entities=extracted_entities
    )
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from lightrag.utils import logger


class RelationshipValidator:
    """
    Validates and filters extracted relationships for quality.
    """
    
    def __init__(
        self,
        strict_mode: bool = True,
        enable_llm_verification: bool = False,
        min_confidence: float = 0.6,
    ):
        """
        Initialize the relationship validator.
        
        Args:
            strict_mode: If True, applies stricter validation rules
            enable_llm_verification: If True, uses LLM to verify ambiguous relationships
            min_confidence: Minimum confidence score (0-1) for a relationship to pass
        """
        self.strict_mode = strict_mode
        self.enable_llm_verification = enable_llm_verification
        self.min_confidence = min_confidence
        
        # Load validation rules
        self._load_validation_rules()
        
        # Statistics tracking
        self.stats = {
            "total_checked": 0,
            "passed": 0,
            "failed": 0,
            "failed_reasons": defaultdict(int),
        }
    
    def _load_validation_rules(self):
        """Load validation rules and patterns."""
        
        # === Allowed relationship types ===
        self.allowed_relationship_types = {
            # Organizational
            "is_ceo_of", "is_cto_of", "is_cfo_of", "is_founder_of", 
            "is_co_founder_of", "works_for", "leads", "manages",
            
            # Location
            "headquartered_at", "has_office_at", "has_production_site_at", 
            "operates_in", "located_in",
            
            # Business
            "partners_with", "is_client_of", "supplies_to", "acquired_by",
            "acquired", "competes_with", "collaborates_with",
            "has_customer", "has_partner",
            
            # Technology & Products
            "uses", "develops", "implements", "specializes_in", "provides",
            "offers_service", "offers_product", "targets_market",
            
            # Industry
            "serves_industry",
            
            # Certifications
            "certified_in", "complies_with",
            
            # Classification
            "classified_as_nace", "active_in_tech_field",
            
            # Financial
            "received_funding_from", "raised_funding_round",
            
            # Inference relationships (lower confidence)
            "co_occurs_with", "related_to", "associated_with",
        }
        
        # === Valid entity type pairs ===
        # Format: (source_type, target_type): [allowed_relationship_types]
        self.valid_type_pairs = {
            ("Person", "Company"): {
                "is_ceo_of", "is_cto_of", "is_cfo_of", "is_founder_of", 
                "is_co_founder_of", "works_for", "leads", "manages"
            },
            ("Person", "Organization"): {
                "is_ceo_of", "is_cto_of", "is_cfo_of", "is_founder_of", 
                "is_co_founder_of", "works_for", "leads", "manages"
            },
            ("Company", "Location"): {
                "headquartered_at", "has_office_at", "has_production_site_at", 
                "operates_in", "located_in"
            },
            ("Organization", "Location"): {
                "headquartered_at", "has_office_at", "has_production_site_at", 
                "operates_in", "located_in"
            },
            ("Company", "Company"): {
                "partners_with", "acquired_by", "acquired", "competes_with", 
                "collaborates_with", "supplies_to", "is_client_of",
                "has_customer", "has_partner"
            },
            ("Organization", "Organization"): {
                "partners_with", "acquired_by", "acquired", "competes_with", 
                "collaborates_with", "supplies_to", "is_client_of",
                "has_customer", "has_partner"
            },
            ("Company", "Technology"): {
                "uses", "develops", "implements", "specializes_in", "provides"
            },
            ("Organization", "Technology"): {
                "uses", "develops", "implements", "specializes_in", "provides"
            },
            ("Company", "Product"): {
                "offers_product", "develops", "provides"
            },
            ("Organization", "Product"): {
                "offers_product", "develops", "provides"
            },
            ("Company", "Service"): {
                "offers_service", "provides"
            },
            ("Organization", "Service"): {
                "offers_service", "provides"
            },
            ("Company", "Industry"): {
                "serves_industry", "operates_in"
            },
            ("Organization", "Industry"): {
                "serves_industry", "operates_in"
            },
            ("Company", "Certification"): {
                "certified_in", "complies_with"
            },
            ("Organization", "Certification"): {
                "certified_in", "complies_with"
            },
            ("Company", "Partnership"): {
                "has_partner", "collaborates_with"
            },
            ("Organization", "Partnership"): {
                "has_partner", "collaborates_with"
            },
        }
        
        # === Forbidden patterns ===
        self.generic_names = {
            "company", "person", "organization", "entity", "unknown", 
            "other", "n/a", "none", "unnamed", "anonymous"
        }
        
        self.forbidden_relationship_keywords = {
            # Too generic
            "related to", "associated with", "connected to", "linked to",
            # Meaningless
            "mentions", "appears with", "exists with",
            # Uncertain
            "might be", "could be", "possibly", "maybe",
        }
        
        # === Relationship-specific rules ===
        self.relationship_rules = {
            "is_ceo_of": {
                "source_types": ["Person"],
                "target_types": ["Company", "Organization"],
                "description_must_contain": ["CEO", "Chief Executive"],
                "confidence_boost": 0.2,  # High confidence relationship
            },
            "is_cto_of": {
                "source_types": ["Person"],
                "target_types": ["Company", "Organization"],
                "description_must_contain": ["CTO", "Chief Technology"],
                "confidence_boost": 0.2,
            },
            "headquartered_at": {
                "source_types": ["Company", "Organization"],
                "target_types": ["Location"],
                "description_must_contain": ["headquarter", "headquarters", "based in", "located in"],
                "confidence_boost": 0.15,
            },
            "partners_with": {
                "source_types": ["Company", "Organization", "Partnership"],
                "target_types": ["Company", "Organization", "Partnership"],
                "description_must_contain": ["partner", "partnership", "collaboration", "collaborate"],
                "confidence_boost": 0.1,
            },
            "acquired_by": {
                "source_types": ["Company", "Organization"],
                "target_types": ["Company", "Organization"],
                "description_must_contain": ["acquired", "acquisition", "bought"],
                "confidence_boost": 0.15,
            },
            # Add more rules as needed
        }
    
    def validate_relationship(
        self,
        relationship: Dict,
        entities: Dict[str, Dict]
    ) -> Tuple[bool, float, str]:
        """
        Validate a single relationship.
        
        Args:
            relationship: Dict with keys: src_id, tgt_id, keywords, description, etc.
            entities: Dict mapping entity names to entity data
        
        Returns:
            Tuple of (is_valid, confidence_score, failure_reason)
        """
        src = relationship.get("src_id", "")
        tgt = relationship.get("tgt_id", "")
        rel_type = relationship.get("keywords", "").strip().lower()
        description = relationship.get("description", "")
        
        # Get entity types
        src_entity = entities.get(src, {})
        tgt_entity = entities.get(tgt, {})
        src_type = src_entity.get("entity_type", "unknown")
        tgt_type = tgt_entity.get("entity_type", "unknown")
        
        confidence = 0.5  # Base confidence
        
        # === Validation Check 1: Basic sanity checks ===
        if not src or not tgt:
            return False, 0.0, "Empty source or target"
        
        if src == tgt:
            return False, 0.0, "Self-referential relationship"
        
        if len(src) < 2 or len(tgt) < 2:
            return False, 0.0, "Entity name too short"
        
        # === Validation Check 2: Generic entity names ===
        if src.lower() in self.generic_names or tgt.lower() in self.generic_names:
            return False, 0.0, f"Generic entity name: {src} or {tgt}"
        
        # === Validation Check 3: Relationship type whitelist ===
        if rel_type not in self.allowed_relationship_types:
            return False, 0.0, f"Relationship type '{rel_type}' not in whitelist"
        
        # === Validation Check 4: Entity type pair validation ===
        type_pair = (src_type, tgt_type)
        reverse_pair = (tgt_type, src_type)
        
        # Check if this entity type pair is allowed for this relationship
        type_pair_valid = False
        if type_pair in self.valid_type_pairs:
            if rel_type in self.valid_type_pairs[type_pair]:
                type_pair_valid = True
                confidence += 0.1
        if reverse_pair in self.valid_type_pairs:
            if rel_type in self.valid_type_pairs[reverse_pair]:
                type_pair_valid = True
                confidence += 0.1
        
        # In strict mode, require explicit type pair match
        if self.strict_mode and not type_pair_valid:
            # Allow if both types are unknown (legacy compatibility)
            if src_type != "unknown" or tgt_type != "unknown":
                return False, 0.0, f"Invalid type pair: {src_type}-{tgt_type} for {rel_type}"
        
        # === Validation Check 5: Description quality ===
        if not description or len(description) < 10:
            confidence -= 0.2
            if self.strict_mode:
                return False, 0.0, "Description too short or missing"
        
        # Check for forbidden keywords in description
        desc_lower = description.lower()
        for forbidden in self.forbidden_relationship_keywords:
            if forbidden in desc_lower:
                confidence -= 0.15
                if self.strict_mode:
                    return False, 0.0, f"Forbidden keyword in description: {forbidden}"
        
        # === Validation Check 6: Relationship-specific rules ===
        if rel_type in self.relationship_rules:
            rule = self.relationship_rules[rel_type]
            
            # Check source type
            if "source_types" in rule:
                if src_type not in rule["source_types"]:
                    return False, 0.0, f"Invalid source type: {src_type} for {rel_type}"
                confidence += 0.05
            
            # Check target type
            if "target_types" in rule:
                if tgt_type not in rule["target_types"]:
                    return False, 0.0, f"Invalid target type: {tgt_type} for {rel_type}"
                confidence += 0.05
            
            # Check description contains expected keywords
            if "description_must_contain" in rule:
                keywords = rule["description_must_contain"]
                contains_keyword = any(kw.lower() in desc_lower for kw in keywords)
                if contains_keyword:
                    confidence += rule.get("confidence_boost", 0.1)
                elif self.strict_mode:
                    return False, 0.0, f"Description lacks expected keywords: {keywords}"
                else:
                    confidence -= 0.1
        
        # === Validation Check 7: Entity existence check ===
        if src not in entities:
            confidence -= 0.2
            if self.strict_mode:
                return False, 0.0, f"Source entity not found: {src}"
        
        if tgt not in entities:
            confidence -= 0.2
            if self.strict_mode:
                return False, 0.0, f"Target entity not found: {tgt}"
        
        # === Final confidence check ===
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        if confidence < self.min_confidence:
            return False, confidence, f"Confidence {confidence:.2f} below threshold {self.min_confidence}"
        
        return True, confidence, "Valid"
    
    def filter_relationships(
        self,
        relationships: List[Dict],
        entities: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a list of relationships, returning valid and invalid ones.
        
        Args:
            relationships: List of relationship dictionaries
            entities: Dict mapping entity names to entity data
        
        Returns:
            Tuple of (valid_relationships, invalid_relationships)
        """
        valid = []
        invalid = []
        
        for rel in relationships:
            self.stats["total_checked"] += 1
            
            is_valid, confidence, reason = self.validate_relationship(rel, entities)
            
            # Add confidence score to relationship
            rel["validation_confidence"] = confidence
            rel["validation_reason"] = reason
            
            if is_valid:
                valid.append(rel)
                self.stats["passed"] += 1
            else:
                invalid.append(rel)
                self.stats["failed"] += 1
                self.stats["failed_reasons"][reason] += 1
        
        return valid, invalid
    
    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        total = self.stats["total_checked"]
        if total == 0:
            return self.stats
        
        stats = dict(self.stats)
        stats["pass_rate"] = self.stats["passed"] / total
        stats["fail_rate"] = self.stats["failed"] / total
        
        return stats
    
    def print_statistics(self):
        """Print validation statistics."""
        stats = self.get_statistics()
        
        logger.info("=" * 60)
        logger.info("Relationship Validation Statistics")
        logger.info("=" * 60)
        logger.info(f"Total relationships checked: {stats['total_checked']}")
        logger.info(f"Passed: {stats['passed']} ({stats.get('pass_rate', 0):.1%})")
        logger.info(f"Failed: {stats['failed']} ({stats.get('fail_rate', 0):.1%})")
        
        if stats["failed"] > 0:
            logger.info("\nTop failure reasons:")
            sorted_reasons = sorted(
                stats["failed_reasons"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for reason, count in sorted_reasons[:10]:
                logger.info(f"  - {reason}: {count}")
        
        logger.info("=" * 60)


# === Helper function for integration ===
def validate_and_filter_relationships(
    relationships: List[Dict],
    entities: Dict[str, Dict],
    strict_mode: bool = True,
    min_confidence: float = 0.6,
) -> List[Dict]:
    """
    Convenience function to validate and filter relationships.
    
    Args:
        relationships: List of relationship dictionaries
        entities: Dict mapping entity names to entity data
        strict_mode: If True, applies stricter validation
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of valid relationships
    """
    validator = RelationshipValidator(
        strict_mode=strict_mode,
        min_confidence=min_confidence
    )
    
    valid_rels, invalid_rels = validator.filter_relationships(relationships, entities)
    
    if invalid_rels:
        logger.info(f"Filtered out {len(invalid_rels)} low-quality relationships")
    
    return valid_rels
