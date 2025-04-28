import os
import asyncio
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from typing import List, Optional, Dict, Any
import logging

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Gemini Integration Functions ---

def _configure_gemini():
    """Checks for API key and configures the genai client if not already done."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    # Configure only if needed (e.g., if running functions standalone)
    # In the server context, this might be configured globally once.
    # Adding a check avoids re-configuration warnings/errors if already configured.
    # Always configure; the library handles idempotency.
    genai.configure(api_key=api_key)


async def gemini_complete(
    prompt: str,
    model_name: str = "gemini-1.5-flash-latest", # Default model, server should override
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    keyword_extraction: bool = False, # Note: Keyword extraction specific format not implemented here
    **kwargs: Any,
) -> str:
    """
    Uses Google Gemini for text completion, adapting history format.

    Args:
        prompt (str): The user's prompt.
        model_name (str): The specific Gemini model to use (e.g., "gemini-1.5-flash-latest").
        system_prompt (Optional[str]): System instructions.
        history_messages (Optional[List[Dict[str, str]]]): Conversation history.
        keyword_extraction (bool): Flag for keyword extraction mode (currently ignored).
        **kwargs: Additional arguments for the Gemini API (e.g., temperature, top_p).

    Returns:
        str: The generated text response.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    _configure_gemini() # Ensure client is configured

    model = genai.GenerativeModel(model_name)
    chat_history = []

    # Gemini API prefers system instructions via specific parameter or start of history.
    # Prepending to the prompt is a common workaround if system_instruction param isn't used/available.
    effective_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    if history_messages:
        for msg in history_messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})

    # Add the current prompt as the last user message
    chat_history.append({"role": "user", "parts": [effective_prompt]})

    # TODO: Make safety settings configurable via kwargs or server config
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    try:
        # Filter kwargs to only include those supported by GenerationConfig
        supported_keys = {'temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}

        logger.debug(f"Calling Gemini model {model_name} with prompt: {prompt[:100]}... and history (len: {len(chat_history)})")
        response = await model.generate_content_async(
            contents=chat_history,
            safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(**filtered_kwargs)
        )

        # Handle potential blocks or empty responses
        if response.candidates and response.candidates[0].content.parts:
             completion = response.candidates[0].content.parts[0].text
             logger.debug(f"Gemini response received: {completion[:100]}...")
             return completion
        elif response.prompt_feedback.block_reason:
             logger.warning(f"Gemini generation blocked for model {model_name}: {response.prompt_feedback.block_reason}")
             return f"Blocked: {response.prompt_feedback.block_reason}"
        else:
             logger.warning(f"Gemini model {model_name} returned no content.")
             return "Error: No content generated."

    except Exception as e:
        logger.error(f"Error during Gemini completion with model {model_name}: {e}", exc_info=True)
        # Depending on the error, you might want to raise it or return a specific error message
        return f"Error: {e}"


@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(multiplier=1, min=2, max=10), # Wait 2s, 4s, 8s between retries
    retry=retry_if_exception_type(ResourceExhausted) # Only retry on 429 errors
)
async def gemini_embed(
    texts: List[str],
    model_name: str = "models/embedding-001", # Default model, server should override
    dim: Optional[int] = None, # Dimension is often model-specific, not needed for API call
    task_type: str = "retrieval_document"
) -> np.ndarray:
    """
    Uses Google Gemini for generating embeddings.

    Args:
        texts (List[str]): A list of texts to embed.
        model_name (str): The specific Gemini embedding model to use (e.g., "models/embedding-001").
        dim (Optional[int]): Expected dimension (used for error handling if API fails).
        task_type (str): The task type for embedding (e.g., "retrieval_document", "retrieval_query").

    Returns:
        np.ndarray: A numpy array of embeddings.

    Raises:
        ValueError: If GEMINI_API_KEY is not set.
    """
    _configure_gemini() # Ensure client is configured

    try:
        logger.debug(f"Calling Gemini embedding model {model_name} for {len(texts)} texts, task type: {task_type}")
        # The embed_content method handles batching.
        result = await genai.embed_content_async(
            model=model_name,
            content=texts,
            task_type=task_type
        )
        embeddings = np.array(result['embedding'])
        logger.debug(f"Received {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        # Optional: Check dimension if provided
        if dim is not None and embeddings.shape[1] != dim:
             logger.warning(f"Expected dimension {dim} but got {embeddings.shape[1]} for model {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Error during Gemini embedding with model {model_name}: {e}", exc_info=True)
        # Return an array of zeros or raise an error based on desired handling
        # Returning zeros allows processing to potentially continue but might skew results
        logger.warning(f"Returning zero vectors due to embedding error.")
        # Use the provided dim if available, otherwise attempt to guess a common default or raise error
        fallback_dim = dim if dim is not None else 768 # Common default, adjust if needed
        return np.zeros((len(texts), fallback_dim))