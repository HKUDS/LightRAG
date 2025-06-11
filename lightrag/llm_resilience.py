"""
LLM Resilience Manager for enhanced error handling and circuit breaker patterns.
Part of Phase 2: Enhanced Error Handling implementation.
"""

import asyncio
import time
import logging
import numpy as np
from collections import Counter
from typing import Callable, Tuple
from . import utils
from .monitoring import (
    get_performance_monitor,
    get_processing_monitor,
    get_enhanced_logger,
)

utils.setup_logger("lightrag.llm_resilience")
logger = logging.getLogger("lightrag.llm_resilience")


class LLMResilienceManager:
    """Manage LLM call resilience with circuit breaker, fallbacks, and error recovery"""

    def __init__(self, max_failures: int = 10, failure_window: int = 600):
        self.max_failures = max_failures
        self.failure_window = failure_window  # seconds
        self.failure_count = 0
        self.failure_timestamps = []
        self.is_circuit_open = False
        self.circuit_open_time = 0
        self.circuit_timeout = 120  # seconds before attempting to close circuit
        self.consecutive_successes = 0
        self.required_successes = 3  # successes needed to close circuit

        # Rate limiting tracking
        self.rate_limit_until = 0
        self.rate_limit_backoff = 1  # start with 1 second
        self.max_rate_limit_backoff = 300  # max 5 minutes

        # Response validation stats
        self.malformed_responses = 0
        self.successful_responses = 0

    def record_failure(self, error_type: str = "general"):
        """Record an LLM call failure"""
        current_time = time.time()
        self.failure_timestamps.append((current_time, error_type))

        # Remove old failures outside the window
        self.failure_timestamps = [
            (ts, err_type)
            for ts, err_type in self.failure_timestamps
            if current_time - ts < self.failure_window
        ]

        self.failure_count = len(self.failure_timestamps)
        self.consecutive_successes = 0

        # Handle rate limiting
        if error_type == "rate_limit":
            self.rate_limit_until = current_time + self.rate_limit_backoff
            self.rate_limit_backoff = min(
                self.rate_limit_backoff * 2, self.max_rate_limit_backoff
            )
            logger.warning(
                f"Rate limited until {self.rate_limit_until}, backoff: {self.rate_limit_backoff}s"
            )

        # Open circuit if too many failures
        if self.failure_count >= self.max_failures and not self.is_circuit_open:
            self.is_circuit_open = True
            self.circuit_open_time = current_time
            logger.error(
                f"LLM circuit breaker opened after {self.failure_count} failures in {self.failure_window}s"
            )

    def record_success(self):
        """Record a successful LLM call"""
        self.consecutive_successes += 1
        self.successful_responses += 1

        # Reset rate limiting on success
        if time.time() > self.rate_limit_until:
            self.rate_limit_backoff = 1

        # Close circuit after enough consecutive successes
        if (
            self.is_circuit_open
            and self.consecutive_successes >= self.required_successes
        ):
            self.is_circuit_open = False
            self.circuit_open_time = 0
            logger.info("LLM circuit breaker closed after successful operations")

        # Gradually reduce failure count on success
        self.failure_count = max(0, self.failure_count - 1)
        if self.failure_count == 0:
            self.failure_timestamps.clear()

    def should_allow_request(self) -> Tuple[bool, str]:
        """Check if LLM requests should be allowed"""
        current_time = time.time()

        # Check rate limiting
        if current_time < self.rate_limit_until:
            wait_time = self.rate_limit_until - current_time
            return False, f"Rate limited, wait {wait_time:.1f}s"

        # Check circuit breaker
        if self.is_circuit_open:
            # Check if circuit timeout has passed
            if current_time - self.circuit_open_time > self.circuit_timeout:
                logger.info(
                    "LLM circuit breaker attempting to close - allowing test request"
                )
                return True, "Testing circuit closure"
            else:
                return False, "Circuit breaker is open"

        return True, "Request allowed"

    def get_failure_stats(self) -> dict:
        """Get failure statistics"""
        current_time = time.time()
        recent_failures = [
            err_type
            for ts, err_type in self.failure_timestamps
            if current_time - ts < 300  # last 5 minutes
        ]

        failure_counts = Counter(recent_failures)

        return {
            "total_failures": self.failure_count,
            "recent_failures": len(recent_failures),
            "failure_types": dict(failure_counts),
            "circuit_open": self.is_circuit_open,
            "rate_limited": current_time < self.rate_limit_until,
            "consecutive_successes": self.consecutive_successes,
            "successful_responses": self.successful_responses,
            "malformed_responses": self.malformed_responses,
        }


# Global LLM resilience manager instance
global_llm_resilience_manager = LLMResilienceManager()


async def enhanced_llm_call(
    llm_func: Callable,
    prompt: str,
    prompt_type: str = "general",
    max_retries: int = 3,
    timeout: float = 30.0,
    **kwargs,
) -> Tuple[str, bool]:
    """
    Enhanced LLM call with comprehensive error handling, resilience, and monitoring.

    Args:
        llm_func: The LLM function to call
        prompt: The prompt to send to the LLM
        prompt_type: Type of prompt for intelligent fallbacks
        max_retries: Maximum number of retry attempts
        timeout: Timeout for each call in seconds
        **kwargs: Additional arguments to pass to llm_func

    Returns:
        Tuple of (response, success) where success indicates if call was successful
    """
    # Initialize monitoring
    perf_monitor = get_performance_monitor()
    proc_monitor = get_processing_monitor()
    enhanced_logger = get_enhanced_logger("lightrag.llm")

    # Start monitoring the LLM call
    with perf_monitor.measure(
        "llm_call_total", prompt_type=prompt_type, max_retries=max_retries
    ):
        enhanced_logger.debug(
            f"Starting LLM call", prompt_type=prompt_type, prompt_length=len(prompt)
        )

        retry_count = 0
        last_exception = None

        while retry_count < max_retries:
            try:
                # Check if request should be allowed
                allowed, reason = global_llm_resilience_manager.should_allow_request()
                if not allowed:
                    if reason == "Circuit breaker is open":
                        logger.warning(
                            f"LLM request blocked ({reason}), using fallback"
                        )
                        return get_fallback_response(prompt), False
                    else:
                        raise Exception(f"LLM request blocked: {reason}")

                # Make the LLM call with timeout
                result = await asyncio.wait_for(
                    llm_func(prompt, **kwargs), timeout=timeout
                )

                # Validate response
                if validate_llm_response(result):
                    global_llm_resilience_manager.record_success()
                    proc_monitor.record_llm_call(success=True)
                    enhanced_logger.debug(
                        "LLM call successful", response_length=len(result)
                    )
                    return result, True
                else:
                    global_llm_resilience_manager.malformed_responses += 1
                    proc_monitor.record_llm_call(success=False)
                    logger.warning(
                        f"LLM returned malformed response (attempt {retry_count + 1})"
                    )
                    if retry_count == max_retries - 1:
                        proc_monitor.record_llm_call(success=False)
                        return get_fallback_response(prompt_type), False

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(
                    f"LLM call timed out (attempt {retry_count + 1}/{max_retries})"
                )
                global_llm_resilience_manager.record_failure("timeout")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Classify error types for better handling
                if "rate" in error_str and "limit" in error_str:
                    logger.warning(
                        f"Rate limit error (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                    )
                    global_llm_resilience_manager.record_failure("rate_limit")

                    # Special handling for rate limits - longer wait
                    if retry_count < max_retries - 1:
                        wait_time = min(30 * (2**retry_count), 300)
                        logger.info(f"Waiting {wait_time}s for rate limit recovery")
                        await asyncio.sleep(wait_time)

                elif "authorization" in error_str or "auth" in error_str:
                    logger.error(f"Authorization error: {str(e)}")
                    global_llm_resilience_manager.record_failure("auth")
                    break  # Don't retry auth errors

                elif "connection" in error_str or "network" in error_str:
                    logger.warning(
                        f"Network error (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                    )
                    global_llm_resilience_manager.record_failure("network")

                else:
                    logger.warning(
                        f"LLM error (attempt {retry_count + 1}/{max_retries}): {str(e)}"
                    )
                    global_llm_resilience_manager.record_failure("general")

            retry_count += 1
            if retry_count < max_retries:
                # Exponential backoff with jitter
                wait_time = min(2**retry_count + np.random.uniform(0, 1), 30)
                logger.debug(f"Waiting {wait_time:.1f}s before retry")
                await asyncio.sleep(wait_time)

        # All retries failed
        proc_monitor.record_llm_call(success=False)
        enhanced_logger.error(
            f"LLM call failed after {max_retries} retries",
            last_error=str(last_exception),
        )

        if retry_count >= max_retries:
            logger.warning(
                f"LLM call failed after {max_retries} retries, using fallback"
            )
            return get_fallback_response(prompt_type), False
        else:
            logger.error(f"LLM call failed after {max_retries} retries")
            raise last_exception or Exception("LLM call failed")


def validate_llm_response(response: str) -> bool:
    """Validate LLM response for basic sanity checks"""
    if not response or not isinstance(response, str):
        return False

    # Check for minimum length
    if len(response.strip()) < 10:
        return False

    # Check for common error patterns
    error_patterns = [
        "i'm sorry",
        "i cannot",
        "error occurred",
        "something went wrong",
        "try again",
        "cannot process",
    ]

    response_lower = response.lower()
    for pattern in error_patterns:
        if pattern in response_lower:
            return False

    return True


def get_fallback_response(prompt: str) -> str:
    """Generate a fallback response when LLM calls fail"""
    # Analyze prompt to determine appropriate fallback
    prompt_lower = prompt.lower()

    if "extract" in prompt_lower and "entities" in prompt_lower:
        return """{"entities": [], "relationships": []}"""

    elif "extract" in prompt_lower and (
        "relations" in prompt_lower or "relationships" in prompt_lower
    ):
        return """{"relationships": []}"""

    elif "keywords" in prompt_lower:
        return """{"high_level_keywords": [], "low_level_keywords": []}"""

    elif "summary" in prompt_lower or "summarize" in prompt_lower:
        return "Unable to generate summary due to service limitations. Please try again later."

    elif "query" in prompt_lower or "question" in prompt_lower:
        return "I apologize, but I'm unable to process your query at this time due to service limitations. Please try again later."

    else:
        return "Service temporarily unavailable. Please try again later."


def get_resilience_stats() -> dict:
    """Get global resilience statistics"""
    return global_llm_resilience_manager.get_failure_stats()
