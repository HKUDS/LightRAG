"""
Enhanced Relationship Classifier for Type-Specific Validation
Based on actual Neo4j relationship data patterns and existing registry infrastructure.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .relationship_registry import RelationshipTypeRegistry
from ...utils import logger


class EnhancedRelationshipClassifier:
    """
    Enhanced relationship classifier that builds on the existing relationship registry
    with data-driven categories based on actual Neo4j relationship patterns.
    """
    
    def __init__(self):
        """Initialize with relationship registry and data-driven categories."""
        self.registry = RelationshipTypeRegistry()
        self._initialize_data_driven_categories()
        self._initialize_confidence_thresholds()
    
    def _initialize_data_driven_categories(self):
        """
        Initialize categories based on actual Neo4j relationship data analysis.
        Categories are based on high-volume relationship patterns from production data.
        """
        self.categories = {
            "technical_core": {
                "description": "High-precision technical dependencies and integrations",
                "types": {
                    # High-volume technical relationships (1000+ instances)
                    "USES", "INTEGRATES_WITH", "RUNS_ON", "IMPLEMENTS", "EXECUTES", 
                    "CALLS_API", "ACCESSES", "CONNECTS_TO",
                    # Direct technical operations
                    "AUTHENTICATES", "PROCESSES", "QUERIES", "RUNS", "LOADS",
                    # CRITICAL: Add missing technical relationships from your data
                    "RUNS", "OPERATES", "DEPLOYED_ON", "HOSTED_ON", "EXECUTES_ON"
                },
                "require_explicit_mention": False,  # RELAXED: Allow technical inference
                "allow_technical_inference": True,
                "examples": ["Redis USES cache", "API CALLS_API endpoint", "Docker RUNS_ON server"]
            },
            
            "development_operations": {
                "description": "Software development and deployment activities",
                "types": {
                    # Core development activities (100-300 instances)
                    "CREATES", "CONFIGURES", "DEVELOPS", "BUILDS", "DEPLOYS", 
                    "INSTALLS", "TESTS", "DEBUGS", "MODIFIES", "EDITS",
                    # Development lifecycle
                    "IMPLEMENTS", "EXTENDS", "BUILDS_FROM", "DEPLOYS_TO"
                },
                "require_explicit_mention": True,
                "allow_technical_inference": False,
                "examples": ["Developer CREATES component", "CI/CD DEPLOYS application"]
            },
            
            "system_interactions": {
                "description": "System-level operations and data management",
                "types": {
                    # System operations (50-100 instances)
                    "HOSTS", "STORES", "MANAGES", "CONTROLS", "MONITORS",
                    "PROVIDES", "TRIGGERS", "SCHEDULES", "QUEUES",
                    # Data operations
                    "READS_FROM", "WRITES_TO", "STORES_DATA_IN", "CACHES_IN"
                },
                "require_explicit_mention": False,
                "allow_technical_inference": True,
                "examples": ["Server HOSTS application", "Database STORES data"]
            },
            
            "troubleshooting_support": {
                "description": "Debugging and problem resolution activities",
                "types": {
                    # High-volume support relationships (500+ instances)
                    "TROUBLESHOOTS", "DEBUGS", "SOLVES", "RESOLVES", 
                    "INVESTIGATES", "ADDRESSES", "ASSISTS", "GUIDES",
                    # Error handling
                    "ENCOUNTERS", "BLOCKS", "CAUSES", "AFFECTS",
                    # CRITICAL: Ensure troubleshooting always maps here
                    "FIXES", "DIAGNOSES", "REPAIRS", "HANDLES_ERROR"
                },
                "require_explicit_mention": False,
                "allow_technical_inference": True,
                "examples": ["Engineer TROUBLESHOOTS issue", "Error BLOCKS deployment"]
            },
            
            "abstract_conceptual": {
                "description": "Conceptual and hierarchical relationships",
                "types": {
                    # Abstract relationships (50-200 instances)
                    "RELATED", "AFFECTS", "SUPPORTS", "REQUIRES", "PART_OF", 
                    "CONTAINS", "INCLUDES", "INVOLVES", "ENABLES",
                    # Hierarchical
                    "IS_A", "INSTANCE_OF", "BELONGS_TO", "ASSOCIATED_WITH"
                },
                "require_explicit_mention": False,
                "allow_technical_inference": True,
                "examples": ["Component PART_OF system", "Issue RELATED problem"]
            },
            
            "data_flow": {
                "description": "Data movement and transformation operations",
                "types": {
                    # Data operations (10-50 instances)
                    "EXTRACTS_DATA_FROM", "PROVIDES_DATA_TO", "EXTRACTS_FROM",
                    "FORMATS", "TRANSFORMS_DATA", "CONVERTS", "PROCESSES_DATA",
                    # File operations  
                    "SCRAPES", "DOWNLOADS_FROM", "UPLOADS_TO", "IMPORTS_FROM"
                },
                "require_explicit_mention": True,
                "allow_technical_inference": True,
                "examples": ["Scraper EXTRACTS_DATA_FROM website", "API PROVIDES_DATA_TO client"]
            },
            
            "structural_composition": {
                "description": "Part-whole and containment relationships",
                "types": {
                    # Structural relationships
                    "PART_OF", "CONTAINS", "COMPRISES", "INCLUDES", "COMPOSED_OF",
                    "MEMBER_OF", "ELEMENT_OF", "COMPONENT_OF", "SUBSET_OF"
                },
                "require_explicit_mention": False,
                "allow_technical_inference": True,
                "examples": ["HTTP request PART_OF n8n workflow", "Component PART_OF system"]
            }
        }
        
        # Create reverse lookup for relationship type to category
        self.type_to_category = {}
        for category, metadata in self.categories.items():
            for rel_type in metadata["types"]:
                self.type_to_category[rel_type] = category
        
        # CRITICAL FIX: Classification overrides for problematic relationships
        self.classification_overrides = {
            # Technical relationships that were being misclassified
            "RUNS_ON": "technical_core",
            "INTEGRATES_WITH": "technical_core", 
            "TROUBLESHOOTS": "troubleshooting_support",
            "USES": "technical_core",
            "ACCESSES": "technical_core",
            "CONNECTS_TO": "technical_core",
            "HOSTS": "technical_core",
            "DEPLOYED_ON": "technical_core",
            "EXECUTES_ON": "technical_core",
            "PROCESSES": "system_interactions",
            "MANAGES": "system_interactions",
            "OPERATES": "system_interactions",
            "CONFIGURES": "development_operations",
            "DEVELOPS": "development_operations",
            "CREATES": "development_operations",
            "BUILDS": "development_operations",
            "DEPLOYS": "development_operations",
            # Additional overrides for commonly misclassified relationships
            "RUNS": "technical_core",
            "CALLS_API": "technical_core",
            "CONFIGURED_BY": "development_operations",
            "HOSTED_ON": "technical_core",
            "OPERATES_ON": "system_interactions",
            "SUPPORTS": "system_interactions",
            # CRITICAL: Add missing verb-based overrides from your analysis
            "CONTROLS": "technical_core",
            "STORED_IN": "system_interactions",
            "EDITS": "development_operations",
            "EXTRACTS": "data_flow",
            "RETURNS": "system_interactions",
            "AFFECTS": "system_interactions",
            "TRIGGERS": "system_interactions",
            "PART_OF": "structural_composition",
            "STORES": "system_interactions",
            "CALLS": "technical_core",
            "SENDS_TO": "system_interactions",
            "RECEIVES_FROM": "system_interactions",
            # CRITICAL: Add missing relationships from debugging analysis
            "ASSISTS": "troubleshooting_support",
            "ASSISTS_WITH": "troubleshooting_support",
            "EVOLVED_FROM": "system_interactions",  # Historical context
            "DERIVED_FROM": "system_interactions",
            "BASED_ON": "system_interactions",
            "IMPROVED_FROM": "system_interactions",
            "REPLACES": "system_interactions"
        }
    
    def _initialize_confidence_thresholds(self):
        """
        Initialize confidence thresholds based on category characteristics and data analysis.
        Higher thresholds for technical precision, lower for conceptual flexibility.
        """
        self.confidence_thresholds = {
            "technical_core": 0.45,          # MODERATE: Preserve quality technical relationships
            "development_operations": 0.45,  # MODERATE: Don't filter configures/creates  
            "system_interactions": 0.40,     # BALANCED: System operations and historical context
            "troubleshooting_support": 0.30,  # VERY PERMISSIVE: Always preserve debugging
            "abstract_conceptual": 0.30,     # LOWERED: Was too high at 0.75
            "data_flow": 0.45,               # MODERATE: Data operations
            "structural_composition": 0.40,  # BALANCED: Structural relationships
            "default": 0.50                   # MODERATE: Default fallback
        }
    
    def classify_relationship(self, relationship_type: str, src_entity: str = "", 
                            tgt_entity: str = "", description: str = "") -> Dict[str, Any]:
        """
        Classify a relationship type with enhanced categorization and confidence scoring.
        
        Args:
            relationship_type: The relationship type to classify
            src_entity: Source entity (optional, for context)
            tgt_entity: Target entity (optional, for context)  
            description: Relationship description (optional, for context)
            
        Returns:
            Dictionary with classification results
        """
        if not relationship_type:
            return self._create_classification_result("abstract_conceptual", 0.1, "RELATED")
        
        # Normalize the relationship type
        normalized_type = relationship_type.upper().replace(' ', '_')
        
        # Store current relationship type for confidence scoring
        self._current_rel_type = normalized_type
        
        # EMERGENCY FIX: Force classification for critical debugging verbs
        CRITICAL_DEBUG_VERBS = {
            "TROUBLESHOOTS": "troubleshooting_support",
            "DEBUGS": "troubleshooting_support", 
            "FIXES": "troubleshooting_support",
            "RESOLVES": "troubleshooting_support",
            "ASSISTS": "troubleshooting_support",
            "DIAGNOSES": "troubleshooting_support",
            "REPAIRS": "troubleshooting_support",
            "HANDLES_ERROR": "troubleshooting_support",
            "INVESTIGATES": "troubleshooting_support",
            "ADDRESSES": "troubleshooting_support"
        }
        
        # Force debug verb classification (emergency override)
        if normalized_type in CRITICAL_DEBUG_VERBS:
            category = CRITICAL_DEBUG_VERBS[normalized_type]
            confidence = 0.90  # High confidence for critical verbs
            return self._create_classification_result(category, confidence, normalized_type, message="Emergency debug verb override")
        
        # CRITICAL FIX: Check classification overrides first
        category = self.classification_overrides.get(normalized_type)
        
        if not category:
            # Direct category lookup
            category = self.type_to_category.get(normalized_type)
        
        if category:
            # Direct match found
            confidence = self._calculate_base_confidence(category, True)
            
            # Apply context-based adjustments
            confidence = self._apply_context_adjustments(
                confidence, category, src_entity, tgt_entity, description
            )
            
            return self._create_classification_result(category, confidence, normalized_type)
        
        # No direct match - use registry fuzzy matching
        best_match, registry_confidence = self.registry.find_best_match_with_confidence(relationship_type)
        
        if best_match and registry_confidence >= 0.4:
            # Found a registry match - classify the matched type
            matched_normalized = best_match.upper().replace(' ', '_')
            matched_category = self.type_to_category.get(matched_normalized, "abstract_conceptual")
            
            # Combine registry confidence with category confidence
            base_confidence = self._calculate_base_confidence(matched_category, False)
            final_confidence = base_confidence * registry_confidence
            
            # Apply context adjustments
            final_confidence = self._apply_context_adjustments(
                final_confidence, matched_category, src_entity, tgt_entity, description
            )
            
            return self._create_classification_result(
                matched_category, final_confidence, matched_normalized, 
                best_match=best_match, registry_confidence=registry_confidence
            )
        
        # EMERGENCY: Check for debug patterns if nothing matched
        debug_patterns = ['troubleshoot', 'debug', 'fix', 'resolve', 'assist', 'diagnose', 'repair']
        normalized_lower = normalized_type.lower()
        for pattern in debug_patterns:
            if pattern in normalized_lower:
                return self._create_classification_result(
                    "troubleshooting_support", 0.85, normalized_type,
                    message="Emergency debug pattern match"
                )
        
        # No good match found - check if it's a technical pattern before defaulting to abstract
        # CRITICAL FIX: Many technical relationships were being misclassified as abstract
        technical_patterns = ['run', 'host', 'call', 'use', 'integrate', 'configure', 'deploy', 
                            'process', 'manage', 'operate', 'connect', 'access', 'execute',
                            'control', 'store', 'edit', 'extract', 'return', 'affect', 'trigger']
        
        for pattern in technical_patterns:
            if pattern in normalized_lower:
                # This is likely a technical relationship - determine category by pattern
                if pattern in ['control', 'store', 'manage', 'process', 'operate']:
                    category = "system_interactions"
                elif pattern in ['edit', 'configure', 'deploy', 'create', 'build']:
                    category = "development_operations"
                elif pattern in ['extract', 'transform', 'convert']:
                    category = "data_flow"
                else:
                    category = "technical_core"
                    
                confidence = 0.45  # Give technical patterns good confidence
                return self._create_classification_result(
                    category, confidence, normalized_type,
                    message="Matched technical pattern"
                )
        
        # Only classify as abstract if no technical patterns found
        return self._create_classification_result(
            "abstract_conceptual", 0.35, normalized_type, 
            message="No registry match found"
        )
    
    def _calculate_base_confidence(self, category: str, is_direct_match: bool) -> float:
        """
        Calculate base confidence score for a category.
        
        Args:
            category: The relationship category
            is_direct_match: Whether this was a direct type match
            
        Returns:
            Base confidence score (0.0-1.0)
        """
        threshold = self.confidence_thresholds.get(category, self.confidence_thresholds["default"])
        
        if is_direct_match:
            # Direct matches get high confidence
            return min(threshold + 0.2, 1.0)
        else:
            # Fuzzy matches get base threshold confidence
            return threshold
    
    def _apply_context_adjustments(self, base_confidence: float, category: str,
                                 src_entity: str, tgt_entity: str, description: str) -> float:
        """
        Apply context-based confidence adjustments.
        
        Args:
            base_confidence: Base confidence score
            category: Relationship category
            src_entity: Source entity
            tgt_entity: Target entity  
            description: Relationship description
            
        Returns:
            Adjusted confidence score
        """
        confidence = base_confidence
        
        # EMERGENCY FIX: Remove problematic confidence floor that was causing 0.350 defaults
        # Let confidence calculation work naturally without artificial floors
        if confidence < 0.15:  # Only boost extremely low confidence
            confidence = 0.20  # Lower floor to prevent over-filtering
        
        # MODERATE context length boost (prevent over-inflation)
        if description and len(description) > 100:
            confidence += 0.15  # Reduced boost for detailed descriptions
        elif description and len(description) > 50:
            confidence += 0.1   # Smaller boost
        elif description and len(description) > 20:
            confidence += 0.05  # Minimal boost for short descriptions
        
        # Enhanced entity specificity boost
        if src_entity and tgt_entity:
            # Expanded technical entity indicators
            technical_indicators = [
                'api', 'database', 'server', 'service', 'component', 'system',
                'docker', 'redis', 'postgres', 'nginx', 'kubernetes', 'n8n',
                'workflow', 'automation', 'bot', 'assistant', 'ai', 'ml',
                'application', 'app', 'platform', 'tool', 'framework',
                'google', 'drive', 'content', 'video', 'tagger', 'processor',
                'node', 'code', 'error', 'developer', 'claude', 'model'  # Added debugging entities
            ]
            
            # Special boost for debugging/assistance relationships
            debug_entities = ['developer', 'claude', 'ai', 'assistant', 'error', 'issue', 'problem']
            src_debug = any(entity in src_entity.lower() for entity in debug_entities)
            tgt_debug = any(entity in tgt_entity.lower() for entity in debug_entities)
            
            if src_debug or tgt_debug:
                confidence += 0.15  # Moderate boost for debugging context
            
            src_technical = any(indicator in src_entity.lower() for indicator in technical_indicators)
            tgt_technical = any(indicator in tgt_entity.lower() for indicator in technical_indicators)
            
            if src_technical and tgt_technical:
                confidence += 0.10  # Moderate boost for technical entity pairs
            elif src_technical or tgt_technical:
                confidence += 0.08   # Small boost for partial technical context
        
        # Action verb boost with expanded patterns
        action_relationship_types = [
            'runs_on', 'integrates_with', 'uses', 'troubleshoots', 'accesses',
            'connects_to', 'calls_api', 'hosts', 'deploys', 'processes',
            'runs', 'operates', 'configured_by', 'hosted_on', 'supports',
            'configures', 'creates', 'builds', 'debugs', 'fixes', 'assists',
            'evolved_from', 'derived_from', 'based_on', 'improved_from', 'replaces'
        ]
        
        current_rel_type = getattr(self, '_current_rel_type', '').lower()
        if any(action_type in current_rel_type for action_type in action_relationship_types):
            confidence += 0.15   # Moderate boost for action verbs - concrete relationships
        
        # Category-specific adjustments (more permissive)
        category_metadata = self.categories.get(category, {})
        
        # Remove harsh penalties for explicit mention requirement
        if category_metadata.get("require_explicit_mention", False):
            if not description or len(description) < 10:  # Much more lenient
                confidence -= 0.05  # Smaller penalty
        
        # Enhanced technical inference boost
        if category_metadata.get("allow_technical_inference", False):
            tech_words = ['configure', 'deploy', 'install', 'setup', 'run', 'execute', 
                         'integrate', 'connect', 'process', 'manage', 'host', 'operate']
            if any(tech_word in (description or "").lower() for tech_word in tech_words):
                confidence += 0.1  # Bigger boost for technical context
        
        # NEVER-FILTER rules for critical relationships
        never_filter_verbs = ['troubleshoots', 'debugs', 'fixes', 'resolves', 'assists']
        if any(verb in current_rel_type for verb in never_filter_verbs):
            confidence = max(confidence, 0.6)  # Ensure debugging relationships pass threshold
        
        # Error/problem entity boost
        error_keywords = ['error', 'issue', 'problem', 'bug', 'failure', 'exception']
        if src_entity and tgt_entity:
            if any(keyword in (src_entity + ' ' + tgt_entity).lower() for keyword in error_keywords):
                confidence += 0.20  # Boost relationships involving errors
        
        # BALANCED: Calibrated minimums for 85-95% retention target
        category_minimums = {
            "technical_core": 0.25,            # Preserve technical relationships
            "troubleshooting_support": 0.80,  # NEVER filter debugging - force high confidence
            "development_operations": 0.25,    # Development operations flexible
            "system_interactions": 0.20,      # System operations very flexible
            "data_flow": 0.25,                # Data operations moderate
            "abstract_conceptual": 0.15,      # Allow filtering only very weak abstracts
            "structural_composition": 0.20    # Structural relationships flexible
        }
        
        min_confidence = category_minimums.get(category, 0.25)
        confidence = max(min_confidence, confidence)
        
        return max(0.0, min(confidence, 1.0))  # Clamp to [0.0, 1.0]
    
    def _create_classification_result(self, category: str, confidence: float, 
                                    normalized_type: str, best_match: Optional[str] = None,
                                    registry_confidence: Optional[float] = None,
                                    message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a classification result dictionary.
        
        Args:
            category: Relationship category
            confidence: Confidence score
            normalized_type: Normalized relationship type
            best_match: Best registry match (if any)
            registry_confidence: Registry matching confidence (if any)
            message: Additional message (if any)
            
        Returns:
            Classification result dictionary
        """
        result = {
            "category": category,
            "confidence": confidence,
            "normalized_type": normalized_type,
            "threshold": self.confidence_thresholds.get(category, self.confidence_thresholds["default"]),
            "should_keep": confidence >= self.confidence_thresholds.get(category, self.confidence_thresholds["default"]),
            "category_metadata": self.categories.get(category, {})
        }
        
        if best_match:
            result["registry_match"] = best_match
            result["registry_confidence"] = registry_confidence
        
        if message:
            result["message"] = message
        
        return result
    
    def get_category_stats(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics for a list of relationships by category.
        
        Args:
            relationships: List of relationship dictionaries with 'rel_type' field
            
        Returns:
            Statistics dictionary by category
        """
        stats = defaultdict(lambda: {
            "count": 0,
            "confidence_scores": [],
            "types": set(),
            "avg_confidence": 0.0,
            "should_keep_count": 0
        })
        
        total_relationships = len(relationships)
        
        for rel in relationships:
            rel_type = rel.get("rel_type", "")
            classification = self.classify_relationship(
                rel_type,
                rel.get("src_id", ""),
                rel.get("tgt_id", ""), 
                rel.get("description", "")
            )
            
            category = classification["category"]
            confidence = classification["confidence"]
            should_keep = classification["should_keep"]
            
            stats[category]["count"] += 1
            stats[category]["confidence_scores"].append(confidence)
            stats[category]["types"].add(rel_type)
            
            if should_keep:
                stats[category]["should_keep_count"] += 1
        
        # Calculate averages and add metadata
        for category, data in stats.items():
            if data["confidence_scores"]:
                data["avg_confidence"] = sum(data["confidence_scores"]) / len(data["confidence_scores"])
                data["retention_rate"] = data["should_keep_count"] / data["count"]
                data["percentage_of_total"] = data["count"] / total_relationships
                data["types"] = list(data["types"])  # Convert set to list for JSON serialization
        
        return dict(stats)
    
    def get_validation_recommendations(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate validation recommendations based on relationship analysis.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Recommendations dictionary
        """
        stats = self.get_category_stats(relationships)
        total_relationships = len(relationships)  # FIX: Define total_relationships
        
        recommendations = {
            "overall_quality": self._assess_overall_quality(stats),
            "category_insights": {},
            "threshold_suggestions": {},
            "problematic_types": []
        }
        
        for category, data in stats.items():
            category_metadata = self.categories.get(category, {})
            current_threshold = self.confidence_thresholds.get(category, 0.6)
            
            # Category-specific insights
            insights = []
            
            if data["retention_rate"] < 0.5:
                insights.append("Low retention rate - consider lowering threshold")
                recommendations["threshold_suggestions"][category] = max(0.3, current_threshold - 0.1)
            elif data["retention_rate"] > 0.95:
                insights.append("Very high retention rate - consider raising threshold") 
                recommendations["threshold_suggestions"][category] = min(0.9, current_threshold + 0.1)
            
            if data["avg_confidence"] < current_threshold - 0.1:
                insights.append("Low average confidence - review relationship extraction")
            
            if data["count"] > total_relationships * 0.3:
                insights.append("High volume category - optimize for performance")
            
            recommendations["category_insights"][category] = {
                "insights": insights,
                "current_threshold": current_threshold,
                "suggested_threshold": recommendations["threshold_suggestions"].get(category, current_threshold)
            }
        
        # Find problematic relationship types
        for rel in relationships:
            classification = self.classify_relationship(rel.get("rel_type", ""))
            if classification["confidence"] < 0.3:
                recommendations["problematic_types"].append({
                    "type": rel.get("rel_type"),
                    "confidence": classification["confidence"],
                    "category": classification["category"]
                })
        
        return recommendations
    
    def _assess_overall_quality(self, stats: Dict[str, Any]) -> str:
        """
        Assess overall relationship quality based on category statistics.
        
        Args:
            stats: Category statistics
            
        Returns:
            Quality assessment string
        """
        total_relationships = sum(data["count"] for data in stats.values())
        total_should_keep = sum(data["should_keep_count"] for data in stats.values())
        
        if total_relationships == 0:
            return "No relationships to assess"
        
        overall_retention = total_should_keep / total_relationships
        
        # Weight by category importance (technical_core is most important)
        weighted_confidence = 0
        total_weight = 0
        
        category_weights = {
            "technical_core": 3.0,
            "development_operations": 2.5, 
            "system_interactions": 2.0,
            "troubleshooting_support": 1.5,
            "data_flow": 2.0,
            "abstract_conceptual": 1.0
        }
        
        for category, data in stats.items():
            weight = category_weights.get(category, 1.0)
            weighted_confidence += data["avg_confidence"] * weight * data["count"]
            total_weight += weight * data["count"]
        
        if total_weight > 0:
            weighted_avg_confidence = weighted_confidence / total_weight
        else:
            weighted_avg_confidence = 0
        
        # Quality assessment
        if overall_retention >= 0.8 and weighted_avg_confidence >= 0.7:
            return "Excellent - High retention with strong confidence scores"
        elif overall_retention >= 0.7 and weighted_avg_confidence >= 0.6:
            return "Good - Solid retention and confidence levels"
        elif overall_retention >= 0.6 and weighted_avg_confidence >= 0.5:
            return "Fair - Moderate quality, room for improvement"
        else:
            return "Poor - Low retention or confidence, needs attention"


# Convenience function for backward compatibility
def classify_relationship_type(relationship_type: str, src_entity: str = "", 
                             tgt_entity: str = "", description: str = "") -> Dict[str, Any]:
    """
    Convenience function to classify a single relationship type.
    
    Args:
        relationship_type: The relationship type to classify
        src_entity: Source entity (optional)
        tgt_entity: Target entity (optional)
        description: Relationship description (optional)
        
    Returns:
        Classification result dictionary
    """
    classifier = EnhancedRelationshipClassifier()
    return classifier.classify_relationship(relationship_type, src_entity, tgt_entity, description)


if __name__ == "__main__":
    # Test with some example relationships from the Neo4j data
    classifier = EnhancedRelationshipClassifier()
    
    test_relationships = [
        {"rel_type": "USES", "src_id": "application", "tgt_id": "redis"},
        {"rel_type": "TROUBLESHOOTS", "src_id": "engineer", "tgt_id": "bug"},
        {"rel_type": "INTEGRATES_WITH", "src_id": "api", "tgt_id": "database"},
        {"rel_type": "RELATED", "src_id": "component", "tgt_id": "system"},
        {"rel_type": "UNKNOWN_TYPE", "src_id": "foo", "tgt_id": "bar"}
    ]
    
    print("Testing Enhanced Relationship Classifier...")
    print("=" * 50)
    
    for rel in test_relationships:
        result = classifier.classify_relationship(
            rel["rel_type"], rel["src_id"], rel["tgt_id"]
        )
        
        print(f"\nRelationship: {rel['src_id']} -[{rel['rel_type']}]-> {rel['tgt_id']}")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Threshold: {result['threshold']:.2f}")
        print(f"  Should Keep: {result['should_keep']}")
        
        if "registry_match" in result:
            print(f"  Registry Match: {result['registry_match']} ({result['registry_confidence']:.2f})")
    
    print("\n" + "=" * 50)
    print("Category Statistics:")
    stats = classifier.get_category_stats(test_relationships)
    
    for category, data in stats.items():
        print(f"\n{category.title()}:")
        print(f"  Count: {data['count']}")
        print(f"  Avg Confidence: {data['avg_confidence']:.2f}")
        print(f"  Retention Rate: {data['retention_rate']:.1%}")
        print(f"  Types: {data['types']}")