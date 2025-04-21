import base64
import logging
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from lightrag.utils import create_meta_text

logger = logging.getLogger(__name__)  # Example logger


async def use_google_gemini_multimodal(
    input_text: str,
    max_tokens: Optional[int] = None,
    history_messages: Optional[List[Dict]] = None,
    images: Optional[List[Dict]] = None,
) -> str:
    """
    Calls the Google Gemini multimodal API.

    Args:
        input_text: The base text prompt.
        images: List of image dictionaries (mime_type, data as base64 str, meta).
        max_tokens: Optional max tokens for the response.
        history_messages: Optional conversation history (not directly used by generate_content).

    Returns:
        The response object from the Gemini API or an error string.
    """
    content_list = [input_text]

    if images:
        for img_data in images:
            meta_text = create_meta_text(img_data["meta"])
            # Add meta text Part
            content_list.append(types.Part.from_text(f"\n- {meta_text}\n"))
            try:
                # Decode base64 string back to bytes
                image_bytes = base64.b64decode(img_data["data"])
                # Add image Part
                content_list.append(
                    types.Part.from_bytes(
                        data=image_bytes, mime_type=img_data["mime_type"]
                    )
                )
            except (TypeError, base64.binascii.Error) as e:
                logger.error(f"Failed to decode/process image data for Gemini: {e}")
                # Return an error object or raise? Returning string for simplicity here.
                return f"[Error processing image: {e}]"
            except Exception as e:
                logger.error(f"Unexpected error processing image: {e}")
                return f"[Error processing image: {e}]"

    # Handle max_tokens if passed, via GenerationConfig
    generation_config = None
    if max_tokens is not None:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens
            # Add other config like temperature if needed
        )

    try:
        client = genai.client.Client(
            project_id="your_project_id",
            location="us-central1",
            vertexai=True,
        )
        # Use the asynchronous method
        response = await client.generate_content_async(
            contents=content_list,
            generation_config=generation_config,
        )
        return response.text  # Return the whole response object
    except Exception as e:
        logger.error(f"Error calling Google Gemini API: {e}")
        # Return an error string or re-raise
        return f"[Error calling LLM: {e}]"
