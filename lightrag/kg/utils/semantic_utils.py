import numpy as np
import logging
from typing import Dict, Optional, List, Union, Tuple, Any
from ...utils import logger
from ...prompt import PROMPTS
from .threshold_manager import ThresholdManager

# Global threshold manager instance with default settings
# This will be used if no specific manager is provided to functions
_default_threshold_manager = ThresholdManager()


def get_default_threshold_manager() -> ThresholdManager:
    """
    Get the default threshold manager instance.

    Returns:
        The default ThresholdManager instance
    """
    return _default_threshold_manager


def set_default_threshold_manager(manager: ThresholdManager) -> None:
    """
    Set the default threshold manager instance.

    Args:
        manager: The ThresholdManager instance to use as default
    """
    global _default_threshold_manager
    _default_threshold_manager = manager


def cosine_similarity(
    v1: np.ndarray, v2: np.ndarray, min_similarity: float = 0.2
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector
        min_similarity: Minimum similarity value to return (default: 0.2)

    Returns:
        Cosine similarity value between -1 and 1, or min_similarity if calculation fails
    """
    if v1 is None or v2 is None:
        logger.warning("Received None vectors for cosine similarity calculation")
        return min_similarity

    try:
        # Check for zero vectors
        if np.all(np.isclose(v1, 0)) or np.all(np.isclose(v2, 0)):
            logger.warning("Zero vector detected in cosine similarity calculation")
            return min_similarity

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            logger.warning("Zero norm detected in cosine similarity calculation")
            return min_similarity

        similarity = dot_product / (norm_v1 * norm_v2)

        # Ensure the result is within valid range
        similarity = max(min(similarity, 1.0), -1.0)

        # Apply minimum threshold
        similarity = max(similarity, min_similarity)

        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return min_similarity


def process_relationship_weight(
    weight: float,
    relationship_type: str = None,
    type_modifiers: Optional[Dict[str, float]] = None,
    min_threshold: float = 0.2,
    threshold_manager: Optional[ThresholdManager] = None,
) -> float:
    """
    Process relationship weights with type-specific modifiers and dynamic thresholds.

    Args:
        weight: Original relationship weight
        relationship_type: Type of relationship
        type_modifiers: Dictionary mapping relationship types to weight modifiers
        min_threshold: Minimum weight threshold (default: 0.2, used if threshold_manager is None)
        threshold_manager: Optional ThresholdManager for dynamic thresholding

    Returns:
        Processed weight
    """
    try:
        # Default type modifiers if none provided
        if type_modifiers is None:
            type_modifiers = {
                # High-value relationship types get a boost
                "generates": 1.15,
                "enables": 1.12,
                "improves": 1.10,
                "facilitates": 1.10,
                "recommends": 1.08,
                "implements": 1.08,
                "converts": 1.08,
                "motivates": 1.07,
                "attracts": 1.05,
                "retains": 1.05,
                "upsells": 1.05,
                "cross_sells": 1.05,
                # Common but less specific types get a slight boost
                "supports": 1.02,
                "complements": 1.02,
                "leverages": 1.02,
                # No boost for default/generic types
                "related": 1.0,
                # Negative/complex relationships require more confidence
                "competes_with": 0.95,
                "refutes": 0.90,
                "prevents": 0.90,
            }

        # Get the appropriate threshold manager
        manager = threshold_manager or _default_threshold_manager

        # Get the dynamic threshold if applicable
        dynamic_threshold = (
            manager.get_threshold(relationship_type) if manager else min_threshold
        )

        # Normalize relationship_type to lowercase for case-insensitive comparison
        rel_type_lower = relationship_type.lower() if relationship_type else None

        # Use the appropriate threshold - relationship-specific when possible
        effective_threshold = dynamic_threshold

        # Higher minimum threshold for specific complex relationships that require more evidence
        if rel_type_lower in ["refutes", "prevents", "competes_with"]:
            effective_threshold = max(
                effective_threshold, 0.25
            )  # 25% minimum for antagonistic relationships

        if weight is None:
            return effective_threshold

        # Convert weight to float if it's a string
        if isinstance(weight, str):
            try:
                weight = float(weight)
            except ValueError:
                logger.warning(
                    f"Invalid weight value: {weight}, using {effective_threshold}"
                )
                return effective_threshold

        # Apply type-specific modifier if available
        if rel_type_lower and type_modifiers:
            if rel_type_lower in type_modifiers:
                weight *= type_modifiers[rel_type_lower]
            else:
                # Try to find closest match
                closest_type = None
                max_similarity = 0

                for mod_type in type_modifiers:
                    # Simple string similarity comparison
                    set1 = set(rel_type_lower)
                    set2 = set(mod_type)
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union > 0 else 0

                    if (
                        similarity > max_similarity and similarity > 0.5
                    ):  # Only use if reasonably similar
                        max_similarity = similarity
                        closest_type = mod_type

                if closest_type:
                    weight *= type_modifiers[closest_type]
                    logger.debug(
                        f"Using modifier for '{closest_type}' ({type_modifiers[closest_type]}) for relationship type '{rel_type_lower}'"
                    )

        # Ensure weight is within valid range
        weight = max(min(weight, 1.0), effective_threshold)

        # Add the observed weight to the threshold manager for future threshold calculations
        if manager:
            manager.add_weight(weight, relationship_type)

        return weight
    except Exception as e:
        logger.error(f"Error processing relationship weight: {str(e)}")
        return min_threshold


def calculate_semantic_weight(
    src_embedding: np.ndarray,
    tgt_embedding: np.ndarray,
    relationship_type: str = None,
    type_modifiers: Optional[Dict[str, float]] = None,
    min_threshold: float = 0.2,
    threshold_manager: Optional[ThresholdManager] = None,
) -> float:
    """
    Calculate semantic weight based on embeddings and relationship type with dynamic thresholds.

    Args:
        src_embedding: Source entity embedding
        tgt_embedding: Target entity embedding
        relationship_type: Type of relationship
        type_modifiers: Dictionary mapping relationship types to weight modifiers
        min_threshold: Minimum weight threshold (default: 0.2, used if threshold_manager is None)
        threshold_manager: Optional ThresholdManager for dynamic thresholding

    Returns:
        Semantic weight between the dynamic threshold and 1.0
    """
    # Get the appropriate threshold manager
    manager = threshold_manager or _default_threshold_manager

    # Get the dynamic threshold if applicable
    dynamic_threshold = (
        manager.get_threshold(relationship_type) if manager else min_threshold
    )

    # Check if embeddings are valid
    if src_embedding is None or tgt_embedding is None:
        # If embeddings are missing, return the effective threshold
        return dynamic_threshold

    if len(src_embedding) == 0 or len(tgt_embedding) == 0:
        # Empty embeddings, return effective threshold
        return dynamic_threshold

    # Calculate base similarity
    similarity = cosine_similarity(src_embedding, tgt_embedding, dynamic_threshold)

    # Process with type-specific modifiers and add to threshold manager
    weight = process_relationship_weight(
        similarity,
        relationship_type=relationship_type,
        type_modifiers=type_modifiers,
        min_threshold=dynamic_threshold,
        threshold_manager=manager,
    )

    return weight


def analyze_weight_distribution(
    weights: List[float], threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Analyze a list of relationship weights and return statistics.

    Args:
        weights: List of relationship weights
        threshold: Threshold for low weights (default: 0.2)

    Returns:
        Dictionary with statistics
    """
    if not weights:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "below_threshold": 0,
            "at_threshold": 0,
        }

    weights_arr = np.array(weights)

    return {
        "count": len(weights),
        "min": float(np.min(weights_arr)),
        "max": float(np.max(weights_arr)),
        "mean": float(np.mean(weights_arr)),
        "median": float(np.median(weights_arr)),
        "below_threshold": int(np.sum(weights_arr < threshold)),
        "at_threshold": int(np.sum(np.isclose(weights_arr, threshold))),
    }
