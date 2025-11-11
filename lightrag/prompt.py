from __future__ import annotations
from typing import Any
from pathlib import Path


PROMPTS: dict[str, Any] = {}

# Get the directory where this file is located
_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt_from_file(filename: str) -> str:
    """Load a prompt from a text file in the prompts directory.
    
    Args:
        filename: Name of the file (without path) to load
        
    Returns:
        The content of the file as a string
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    file_path = _PROMPT_DIR / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt file not found: {file_path}. "
            f"Please ensure the file exists in the prompts directory."
        )


def _load_examples_from_files(base_name: str, count: int) -> list[str]:
    """Load multiple example files with a common base name.
    
    Args:
        base_name: Base name of the example files (e.g., "entity_extraction_example")
        count: Number of example files to load
        
    Returns:
        List of example strings loaded from files
    """
    examples = []
    for i in range(1, count + 1):
        filename = f"{base_name}_{i}.md"
        try:
            content = _load_prompt_from_file(filename)
            examples.append(content)
        except FileNotFoundError:
            # If we can't find the file, stop loading more
            break
    return examples


# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Load main prompts from files
PROMPTS["entity_extraction_system_prompt"] = _load_prompt_from_file(
    "entity_extraction_system_prompt.md"
)
PROMPTS["entity_extraction_user_prompt"] = _load_prompt_from_file(
    "entity_extraction_user_prompt.md"
)
PROMPTS["entity_continue_extraction_user_prompt"] = _load_prompt_from_file(
    "entity_continue_extraction_user_prompt.md"
)
PROMPTS["summarize_entity_descriptions"] = _load_prompt_from_file(
    "summarize_entity_descriptions.md"
)
PROMPTS["fail_response"] = _load_prompt_from_file("fail_response.md")
PROMPTS["rag_response"] = _load_prompt_from_file("rag_response.md")
PROMPTS["naive_rag_response"] = _load_prompt_from_file("naive_rag_response.md")
PROMPTS["kg_query_context"] = _load_prompt_from_file("kg_query_context.md")
PROMPTS["naive_query_context"] = _load_prompt_from_file("naive_query_context.md")
PROMPTS["keywords_extraction"] = _load_prompt_from_file("keywords_extraction.md")

# Load examples from files
PROMPTS["entity_extraction_examples"] = _load_examples_from_files(
    "entity_extraction_example", 3
)
PROMPTS["keywords_extraction_examples"] = _load_examples_from_files(
    "keywords_extraction_example", 3
)
