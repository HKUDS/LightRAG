import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
from ...utils import logger


class ThresholdManager:
    """
    Manages dynamic thresholds for relationship weights based on observed data.

    This class tracks relationship weights, calculates appropriate thresholds
    based on percentiles, and provides adaptive thresholding for different
    relationship types.
    """

    def __init__(
        self,
        default_threshold: float = 0.2,
        min_samples: int = 50,
        update_interval: int = 100,
        percentile: float = 10.0,
        use_dynamic_thresholds: bool = True,
        floor: float = 0.1,
        ceiling: float = 0.4,
    ):
        """
        Initialize the ThresholdManager.

        Args:
            default_threshold: Default threshold to use when not enough data is available (default: 0.2)
            min_samples: Minimum number of samples required before calculating dynamic thresholds (default: 50)
            update_interval: Number of new samples after which thresholds are recalculated (default: 100)
            percentile: Percentile to use for threshold calculation (default: 10.0)
            use_dynamic_thresholds: Whether to use dynamic thresholds or always use default (default: True)
            floor: Minimum allowed threshold value (default: 0.1)
            ceiling: Maximum allowed threshold value (default: 0.4)
        """
        self.default_threshold = default_threshold
        self.min_samples = min_samples
        self.update_interval = update_interval
        self.percentile = percentile
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.floor = floor
        self.ceiling = ceiling

        # Storage for observed weights
        self._all_weights: List[float] = []
        self._type_weights: Dict[str, List[float]] = defaultdict(list)

        # Calculated thresholds
        self._global_threshold: float = default_threshold
        self._type_thresholds: Dict[str, float] = {}

        # Counters
        self._samples_since_update: int = 0

    def add_weight(
        self, weight: float, relationship_type: Optional[str] = None
    ) -> None:
        """
        Add an observed weight to the manager.

        Args:
            weight: The relationship weight value
            relationship_type: The type of relationship (optional)
        """
        if not isinstance(weight, (int, float)):
            try:
                weight = float(weight)
            except (ValueError, TypeError):
                logger.warning(f"Invalid weight value: {weight}, ignoring")
                return

        # Store the weight
        self._all_weights.append(weight)

        # If relationship type is provided, store by type
        if relationship_type:
            self._type_weights[relationship_type].append(weight)

        # Increment counter
        self._samples_since_update += 1

        # Check if update is needed
        if (
            self._samples_since_update >= self.update_interval
            and len(self._all_weights) >= self.min_samples
        ):
            self._update_thresholds()
            self._samples_since_update = 0

    def _update_thresholds(self) -> None:
        """
        Update the thresholds based on observed weights.

        This method calculates new thresholds based on the percentile of observed weights.
        """
        # Calculate global threshold
        if len(self._all_weights) >= self.min_samples:
            self._global_threshold = np.percentile(self._all_weights, self.percentile)
            # Apply floor and ceiling
            self._global_threshold = max(
                self.floor, min(self.ceiling, self._global_threshold)
            )
            logger.info(
                f"Updated global threshold to {self._global_threshold:.3f} based on {len(self._all_weights)} samples"
            )

        # Calculate type-specific thresholds
        for rel_type, weights in self._type_weights.items():
            if len(weights) >= self.min_samples:
                threshold = np.percentile(weights, self.percentile)
                # Apply floor and ceiling
                threshold = max(self.floor, min(self.ceiling, threshold))
                self._type_thresholds[rel_type] = threshold
                logger.info(
                    f"Updated threshold for '{rel_type}' to {threshold:.3f} based on {len(weights)} samples"
                )

    def get_threshold(self, relationship_type: Optional[str] = None) -> float:
        """
        Get the appropriate threshold for a relationship type.

        Args:
            relationship_type: The type of relationship (optional)

        Returns:
            The threshold value to use
        """
        # If dynamic thresholds are disabled, always return default
        if not self.use_dynamic_thresholds:
            return self.default_threshold

        # If relationship type is provided and we have a type-specific threshold
        if relationship_type and relationship_type in self._type_thresholds:
            return self._type_thresholds[relationship_type]

        # If we have a global threshold
        if len(self._all_weights) >= self.min_samples:
            return self._global_threshold

        # Otherwise use default
        return self.default_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about thresholds and observed weights.

        Returns:
            Dictionary with threshold statistics
        """
        stats = {
            "default_threshold": self.default_threshold,
            "global_threshold": self._global_threshold,
            "total_samples": len(self._all_weights),
            "type_thresholds": self._type_thresholds.copy(),
            "type_sample_counts": {
                rel_type: len(weights)
                for rel_type, weights in self._type_weights.items()
            },
            "min_samples_required": self.min_samples,
            "use_dynamic_thresholds": self.use_dynamic_thresholds,
        }

        # Add weight distribution stats if we have enough samples
        if len(self._all_weights) > 0:
            weights_arr = np.array(self._all_weights)
            stats.update(
                {
                    "weight_min": float(np.min(weights_arr)),
                    "weight_max": float(np.max(weights_arr)),
                    "weight_mean": float(np.mean(weights_arr)),
                    "weight_median": float(np.median(weights_arr)),
                    "weight_std": float(np.std(weights_arr)),
                    "weight_percentile_10": float(np.percentile(weights_arr, 10)),
                    "weight_percentile_25": float(np.percentile(weights_arr, 25)),
                    "weight_percentile_75": float(np.percentile(weights_arr, 75)),
                    "weight_percentile_90": float(np.percentile(weights_arr, 90)),
                }
            )

        return stats


# Global instance for default threshold manager
_default_threshold_manager: Optional[ThresholdManager] = None


def get_default_threshold_manager() -> ThresholdManager:
    """
    Get the default global ThresholdManager instance.
    If one doesn't exist, it creates a default one.

    Returns:
        The default ThresholdManager instance
    """
    global _default_threshold_manager
    if _default_threshold_manager is None:
        logger.info("Creating default ThresholdManager instance")
        _default_threshold_manager = ThresholdManager()
    return _default_threshold_manager


def set_default_threshold_manager(manager: ThresholdManager) -> None:
    """
    Set the default global ThresholdManager instance.

    Args:
        manager: The ThresholdManager instance to set as default
    """
    global _default_threshold_manager
    if not isinstance(manager, ThresholdManager):
        raise TypeError("Provided manager must be an instance of ThresholdManager")
    logger.info(f"Setting default ThresholdManager instance to: {manager}")
    _default_threshold_manager = manager
