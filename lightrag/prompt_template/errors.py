from __future__ import annotations


class PromptTemplateError(Exception):
    """Base error for prompt template management."""


class PromptTemplateNotFoundError(PromptTemplateError):
    """Raised when a template cannot be found in system/user stores."""


class PromptTemplateValidationError(PromptTemplateError):
    """Raised when template content or render variables fail validation."""


class PromptTemplateGitError(PromptTemplateError):
    """Raised when git initialization/commit operations fail."""
