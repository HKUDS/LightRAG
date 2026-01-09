"""
High-Quality Relationship Extraction Configuration for Company Profiles

This example demonstrates how to configure LightRAG for maximum relationship
extraction accuracy by using:
1. High-quality prompts (now default in lightrag.prompt)
2. Enhanced quality validation
3. Better chunking strategy
4. Higher co-occurrence thresholds for inference

Usage:
    from lightrag import LightRAG
    from examples.company_profile_extraction_config import (
        get_high_quality_extraction_config
    )

    # Initialize with high-quality config
    config = get_high_quality_extraction_config()
    rag = LightRAG(working_dir="./rag_storage", **config)

    # The high-quality prompts are now used by default!
"""


def get_high_quality_extraction_config() -> dict:
    """
    Returns configuration optimized for high-quality relationship extraction.
    
    Returns:
        dict: Configuration parameters for LightRAG initialization
    """
    return {
        # === Chunking Strategy ===
        # Larger overlap preserves entity context across chunks
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 400,  # 33% overlap (was 250/20%, originally 100/8%)
        
        # === Extraction Quality ===
        # More gleaning passes = better coverage but slower
        "entity_extract_max_gleaning": 2,  # Increased from default 1
        
        # === Relationship Inference ===
        # Co-occurrence inference helps reduce isolated nodes
        # but should have HIGH thresholds to avoid false relationships
        "addon_params": {
            "enable_cooccurrence_inference": True,
            "min_cooccurrence": 4,  # Increased from 3 for higher quality

            # === Advanced Company Relationship Inference ===
            # Infer competitor, partnership, and supply chain relationships
            "enable_company_relationship_inference": True,
            "company_inference_min_cooccurrence": 2,
            "company_inference_confidence_threshold": 0.5,

            "language": "English",
            "entity_types": [
                "Company", "Person", "Technology", "Location",
                "Service", "Product", "Industry", "Partnership",
                "Certification", "Project"
            ],
        },
        
        # === Caching ===
        # Enable caching to save costs during development/testing
        "enable_llm_cache": True,
    }


# Note: High-quality prompts are now the default in lightrag.prompt!
# No need for a separate enable_strict_prompts() function.


def add_relationship_validation_rules():
    """
    Enhanced validation rules to filter out low-quality relationships.
    
    These rules can be integrated into your extraction pipeline to
    post-filter relationships after LLM extraction.
    """
    
    RELATIONSHIP_VALIDATION_RULES = {
        # === Rule 1: Relationship Type Whitelist ===
        "allowed_relationship_types": {
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
        },
        
        # === Rule 2: Entity Type Pair Rules ===
        # Only allow relationships between meaningful entity type pairs
        "valid_entity_type_pairs": {
            ("Person", "Company"): ["is_ceo_of", "is_cto_of", "is_cfo_of", "is_founder_of", "is_co_founder_of", "works_for", "leads", "manages"],
            ("Company", "Location"): ["headquartered_at", "has_office_at", "has_production_site_at", "operates_in"],
            ("Company", "Company"): ["partners_with", "acquired_by", "acquired", "competes_with", "collaborates_with", "supplies_to", "is_client_of"],
            ("Company", "Technology"): ["uses", "develops", "implements", "specializes_in"],
            ("Company", "Product"): ["offers_product", "develops"],
            ("Company", "Service"): ["offers_service", "provides"],
            ("Company", "Industry"): ["serves_industry"],
            ("Company", "Certification"): ["certified_in", "complies_with"],
            # Add more valid pairs as needed
        },
        
        # === Rule 3: Forbidden Patterns ===
        "forbidden_patterns": [
            # Generic/vague relationships
            {"source_pattern": r"^(company|organization|entity|unknown)$", "reason": "Generic source entity"},
            {"target_pattern": r"^(company|organization|entity|unknown)$", "reason": "Generic target entity"},
            
            # Too short entity names (likely extraction errors)
            {"min_entity_length": 2, "reason": "Entity name too short"},
            
            # Self-referential relationships
            {"same_source_target": True, "reason": "Self-referential relationship"},
        ],
        
        # === Rule 4: Relationship-Specific Validation ===
        "relationship_specific_rules": {
            "is_ceo_of": {
                "required_source_type": "Person",
                "required_target_type": "Company",
                "description_keywords": ["CEO", "Chief Executive"],
            },
            "headquartered_at": {
                "required_source_type": "Company",
                "required_target_type": "Location",
                "description_keywords": ["headquarters", "headquartered", "based in"],
            },
            "partners_with": {
                "allowed_source_types": ["Company", "Partnership"],
                "allowed_target_types": ["Company", "Partnership"],
                "description_keywords": ["partner", "partnership", "collaboration"],
            },
            # Add more specific rules as needed
        },
    }
    
    return RELATIONSHIP_VALIDATION_RULES


def validate_relationship(relationship: dict, validation_rules: dict) -> tuple[bool, str]:
    """
    Validate a single relationship against quality rules.
    
    Args:
        relationship: Dict with keys: src_id, tgt_id, keywords, description, entity_type_src, entity_type_tgt
        validation_rules: Validation rules from add_relationship_validation_rules()
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    import re
    
    src = relationship.get("src_id", "")
    tgt = relationship.get("tgt_id", "")
    rel_type = relationship.get("keywords", "")
    description = relationship.get("description", "")
    src_type = relationship.get("entity_type_src", "")
    tgt_type = relationship.get("entity_type_tgt", "")
    
    # Rule 1: Check relationship type whitelist
    allowed_types = validation_rules["allowed_relationship_types"]
    if rel_type not in allowed_types:
        return False, f"Relationship type '{rel_type}' not in whitelist"
    
    # Rule 2: Check entity type pair validity
    valid_pairs = validation_rules["valid_entity_type_pairs"]
    type_pair = (src_type, tgt_type)
    reverse_pair = (tgt_type, src_type)
    
    if type_pair in valid_pairs:
        if rel_type not in valid_pairs[type_pair]:
            return False, f"Relationship '{rel_type}' not valid for {src_type}-{tgt_type} pair"
    elif reverse_pair in valid_pairs:
        if rel_type not in valid_pairs[reverse_pair]:
            return False, f"Relationship '{rel_type}' not valid for {tgt_type}-{src_type} pair"
    else:
        # Allow if not explicitly restricted
        pass
    
    # Rule 3: Check forbidden patterns
    for pattern in validation_rules["forbidden_patterns"]:
        # Check generic entity names
        if "source_pattern" in pattern:
            if re.match(pattern["source_pattern"], src.lower(), re.IGNORECASE):
                return False, pattern["reason"]
        
        if "target_pattern" in pattern:
            if re.match(pattern["target_pattern"], tgt.lower(), re.IGNORECASE):
                return False, pattern["reason"]
        
        # Check minimum length
        if "min_entity_length" in pattern:
            min_len = pattern["min_entity_length"]
            if len(src) < min_len or len(tgt) < min_len:
                return False, pattern["reason"]
        
        # Check self-referential
        if pattern.get("same_source_target") and src == tgt:
            return False, pattern["reason"]
    
    # Rule 4: Relationship-specific validation
    specific_rules = validation_rules["relationship_specific_rules"]
    if rel_type in specific_rules:
        rule = specific_rules[rel_type]
        
        # Check required types
        if "required_source_type" in rule:
            if src_type != rule["required_source_type"]:
                return False, f"Source must be {rule['required_source_type']}, got {src_type}"
        
        if "required_target_type" in rule:
            if tgt_type != rule["required_target_type"]:
                return False, f"Target must be {rule['required_target_type']}, got {tgt_type}"
        
        # Check description contains expected keywords
        if "description_keywords" in rule:
            keywords = rule["description_keywords"]
            if not any(kw.lower() in description.lower() for kw in keywords):
                return False, f"Description lacks expected keywords: {keywords}"
    
    return True, "Valid"


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    print("=" * 70)
    print("High-Quality Relationship Extraction Configuration")
    print("=" * 70)
    
    # 1. Get optimized config
    config = get_high_quality_extraction_config()
    print("\n1. Optimized Configuration:")
    print(f"   - Chunk size: {config['chunk_token_size']} tokens")
    print(f"   - Chunk overlap: {config['chunk_overlap_token_size']} tokens (33%)")
    print(f"   - Gleaning passes: {config['entity_extract_max_gleaning']}")
    print(f"   - Min co-occurrence: {config['addon_params']['min_cooccurrence']}")
    
    # 2. Show validation rules
    validation_rules = add_relationship_validation_rules()
    print(f"\n2. Validation Rules Loaded:")
    print(f"   - {len(validation_rules['allowed_relationship_types'])} allowed relationship types")
    print(f"   - {len(validation_rules['valid_entity_type_pairs'])} entity type pair rules")
    print(f"   - {len(validation_rules['forbidden_patterns'])} forbidden patterns")
    print(f"   - {len(validation_rules['relationship_specific_rules'])} specific relationship rules")
    
    # 3. Test validation
    print("\n3. Example Validation:")
    
    # Good relationship
    good_rel = {
        "src_id": "Dr. Maria Schmidt",
        "tgt_id": "TechFlow Solutions GmbH",
        "keywords": "is_ceo_of",
        "description": "Dr. Maria Schmidt is the CEO of TechFlow Solutions GmbH",
        "entity_type_src": "Person",
        "entity_type_tgt": "Company"
    }
    valid, reason = validate_relationship(good_rel, validation_rules)
    print(f"   ✓ Good relationship: {valid} - {reason}")
    
    # Bad relationship (wrong type pair)
    bad_rel = {
        "src_id": "TechFlow Solutions GmbH",
        "tgt_id": "Munich",
        "keywords": "is_ceo_of",  # Wrong! Company can't be CEO of Location
        "description": "TechFlow is CEO of Munich",
        "entity_type_src": "Company",
        "entity_type_tgt": "Location"
    }
    valid, reason = validate_relationship(bad_rel, validation_rules)
    print(f"   ✗ Bad relationship: {valid} - {reason}")
    
    print("\n" + "=" * 70)
    print("Configuration ready! Use in your LightRAG setup.")
    print("=" * 70)
