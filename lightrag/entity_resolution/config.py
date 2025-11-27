"""Configuration for Entity Resolution

Uses the same LLM that LightRAG is configured with - no separate model config needed.
"""

from dataclasses import dataclass, field


@dataclass
class EntityResolutionConfig:
    """Configuration for the entity resolution system."""

    # Whether entity resolution is enabled
    enabled: bool = True

    # Fuzzy pre-resolution: Enable/disable within-batch fuzzy matching before
    # VDB lookup. When enabled, entities in the same batch are matched by string
    # similarity alone. Set to False to skip fuzzy pre-resolution entirely (only
    # exact case-insensitive matches will be accepted within batch; all other
    # resolution goes to VDB/LLM). Disabling reduces false positives but may
    # miss obvious typo corrections.
    fuzzy_pre_resolution_enabled: bool = True

    # Fuzzy string matching threshold (0-1)
    # Above this = auto-match (catches typos like Dupixant/Dupixent at 0.88)
    # Below this = continue to vector search
    # Tuning advice:
    #   0.90+ = Very conservative, near-identical strings (Dupixent/Dupixant)
    #   0.85  = Balanced default, catches typos, avoids most false positives
    #   0.80  = Aggressive, may merge distinct entities with similar names
    #   <0.75 = Not recommended, high false positive risk (Celebrex/Cerebyx=0.67)
    # Test with your domain data; pharmaceutical names need higher thresholds.
    fuzzy_threshold: float = 0.85

    # Vector similarity threshold for finding candidates
    # Low threshold = cast wide net, LLM will verify
    # 0.5 catches FDA/US Food and Drug Administration at 0.67
    vector_threshold: float = 0.5

    # Maximum number of vector candidates to verify with LLM
    # Limits cost - uses same LLM as LightRAG main config
    max_candidates: int = 3

    # LLM verification prompt template
    llm_prompt_template: str = field(
        default="""Are these two terms referring to the same entity?
Consider typos, misspellings, abbreviations, or alternate names.

Term A: {term_a}
Term B: {term_b}

Answer only YES or NO.""",
    )


# Default configuration
DEFAULT_CONFIG = EntityResolutionConfig()
