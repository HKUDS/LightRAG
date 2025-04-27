import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger, logger
from lightrag.kg.shared_storage import initialize_pipeline_status
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Setup logger globally
setup_logger("lightrag", level="INFO") # Configure the imported logger

# --- Configuration ---
# Ensure GEMINI_API_KEY environment variable is set
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

WORKING_DIR = "./rag_gemini_custom_storage"
LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Specific preview model
EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07" # Standard embedding model
EMBEDDING_DIM = 3072 # Specific dimension for this model

# Setup logger

# Create working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# --- Gemini Integration Functions ---

async def gemini_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    keyword_extraction: bool = False, # Note: Keyword extraction specific format not implemented here
    **kwargs,
) -> str:
    """
    Uses Google Gemini for text completion, adapting history format.
    """
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    chat_history = []
    if system_prompt:
         # Gemini API prefers system instructions via specific parameter or start of history
         # For simplicity, we'll prepend it to the history if provided.
         # More complex handling might involve the system_instruction parameter if available for the model/task.
         # Note: The exact handling might depend on the specific Gemini model version and task type.
         # Prepending as a user message might not be ideal, but it's a common workaround.
         # Consider using the system_instruction parameter in GenerativeModel if appropriate.
         # For now, we will add it as the first part of the prompt or history.
         # Let's integrate it into the prompt for this basic example.
         prompt = f"{system_prompt}\n\n{prompt}"


    if history_messages:
        for msg in history_messages:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})

    # Add the current prompt as the last user message
    chat_history.append({"role": "user", "parts": [prompt]})

    # Filter out potentially harmful content generation
    # TODO: Make safety settings configurable
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    try:
        # Use generate_content for potentially better history management
        # Filter kwargs to only include those supported by GenerationConfig
        supported_keys = {'temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_keys}

        response = await model.generate_content_async(
            contents=chat_history,
            safety_settings=safety_settings,
            # Pass only supported kwargs
            generation_config=genai.types.GenerationConfig(**filtered_kwargs)
        )
        # Handle potential blocks or empty responses
        if response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        elif response.prompt_feedback.block_reason:
             logger.warning(f"Gemini generation blocked: {response.prompt_feedback.block_reason}")
             return f"Blocked: {response.prompt_feedback.block_reason}"
        else:
             logger.warning("Gemini returned no content.")
             return "Error: No content generated."

    except Exception as e:
        logger.error(f"Error during Gemini completion: {e}")
        # Depending on the error, you might want to raise it or return a specific error message
        return f"Error: {e}"


async def gemini_embed(texts: list[str]) -> np.ndarray:
    """
    Uses Google Gemini for generating embeddings.
    Handles potential API errors.
    """
    try:
        # The embed_content method handles batching.
        # Task type can be specified, default is RETRIEVAL_DOCUMENT
        # Other types: RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING
        result = await genai.embed_content_async(
            model=EMBEDDING_MODEL_NAME,
            content=texts,
            task_type="retrieval_document" # Assuming document embedding context
        )
        return np.array(result['embedding']) # Adapt based on actual API response structure if needed
    except Exception as e:
        logger.error(f"Error during Gemini embedding: {e}")
        # Return an array of zeros or raise an error based on desired handling
        # Returning zeros allows processing to potentially continue but might skew results
        return np.zeros((len(texts), EMBEDDING_DIM))


# --- LightRAG Initialization ---

async def initialize_rag():
    """Initializes the LightRAG instance with Gemini functions."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_complete, # Use the new Gemini completion function
        llm_model_name=LLM_MODEL_NAME, # Pass model name for potential internal use/logging
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM, # Use the dimension for the Gemini model
            max_token_size=8192, # Default value as requested
            func=gemini_embed # Use the new Gemini embedding function
        ),
        # Add other LightRAG configurations as needed (storage types, etc.)
    )

    await rag.initialize_storages()
    # Initialize pipeline status (important for tracking document processing)
    await initialize_pipeline_status()

    return rag

# --- Main Execution Logic ---

async def main():
    rag = None # Initialize rag to None for finally block
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # Insert some text (replace with your actual data)
        logger.info("Inserting text...")
        await rag.ainsert("The quick brown fox jumps over the lazy dog.")
        await rag.ainsert("Artificial intelligence is transforming various industries.")
        logger.info("Insertion complete.")

        # Wait briefly for potential background processing if needed (depends on storage)
        await asyncio.sleep(2)

        # Perform hybrid search (example query)
        query = "Tell me about AI."
        logger.info(f"Performing query: {query}")
        mode="hybrid" # Example mode
        response = await rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )
        print("\nQuery Response:")
        print(response)

        # Example of getting context only
        logger.info(f"Getting context only for query: {query}")
        context_only = await rag.aquery(
            query,
            param=QueryParam(mode=mode, only_need_context=True)
        )
        print("\nContext Only:")
        print(context_only)


    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)
    finally:
        if rag:
            logger.info("Finalizing storages...")
            await rag.finalize_storages()
            logger.info("Storages finalized.")

if __name__ == "__main__":
    # Ensure you have an event loop running if calling async functions directly
    # asyncio.run() handles this automatically.
    asyncio.run(main())