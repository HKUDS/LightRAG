import os
import toml

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

def load_prompts(toml_file=os.path.join(os.path.dirname(__file__), "prompts", "code.toml")):
    """Load prompts from a TOML file and merge them into the existing PROMPTS dictionary."""
    try:
        # Load prompts from the TOML file.
        toml_data = toml.load(toml_file)

        # Merge TOML prompts into the existing PROMPTS dictionary.
        PROMPTS.update({k: v for k, v in toml_data.items() if v})

    except Exception as e:
        print(f"Error loading and merging prompts: {e}")

# Example usage: Load TOML prompts and merge with existing PROMPTS.
load_prompts()