import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Function to initialize OpenAI client based on summarize_chunks.py logic
def initialize_openai_client():
    """Initializes and returns the OpenAI client based on environment variables."""
    llm_api_key = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    llm_api_host = os.environ.get("LLM_BINDING_HOST")
    llm_model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-16k") # Default model

    if not llm_api_key:
        logging.error("OpenAI API key not found. Please set OPENAI_API_KEY, LLM_BINDING_API_KEY, or SILICONFLOW_API_KEY in your .env file.")
        return None, None

    client_params = {"api_key": llm_api_key}
    if llm_api_host:
        client_params["base_url"] = llm_api_host
        logging.info(f"Using custom API base URL: {llm_api_host}")

    try:
        client = OpenAI(**client_params)
        logging.info(f"OpenAI client initialized successfully. Using model: {llm_model}")
        return client, llm_model
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return None, None

# Function to load categories with descriptions from config.yaml
def load_categories_with_descriptions(config_path="tests/config.yaml"):
    """Loads category names and their descriptions/focus from the config file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        category_configs = config.get("category_configs", {})
        if not category_configs:
            logging.warning(f"No 'category_configs' found in {config_path}.")
            return {}

        categories_with_desc = {}
        for cat_name, cat_config in category_configs.items():
            # Try to extract focus from entity extraction prompt template
            description = f"涉及 {cat_name} 相关内容。" # Default description
            try:
                # Look for the specific instruction line in the entity extraction prompt
                # Navigate safely through the potentially missing keys
                prompts_config = cat_config.get("prompts", {})
                entity_extraction_config = prompts_config.get("entity_extraction", {})
                template = entity_extraction_config.get("template", "")

                if template:
                    # Find the line after the introductory phrase which often summarizes the focus
                    focus_marker = "请从以下文本中提取实体，"
                    start_index = template.find(focus_marker)
                    if start_index != -1:
                        start_index += len(focus_marker)
                        # Find the next newline character after the marker
                        end_index = template.find("\n", start_index)
                        if end_index == -1: # If no newline found, take the rest of the string
                            end_index = len(template)

                        focus_line = template[start_index:end_index].strip()
                        # Clean up common markdown/emphasis characters
                        focus_line = focus_line.replace("**", "").strip()
                        if focus_line:
                            description = f"内容侧重: {focus_line}"
                        else:
                             logging.debug(f"Could not extract focus line for category: {cat_name}")
                    else:
                        logging.debug(f"Focus marker '{focus_marker}' not found in template for category: {cat_name}")

            except Exception as e:
                 # Keep default description if parsing fails, log the error
                 logging.warning(f"Error parsing description for category {cat_name}: {e}. Using default.")

            categories_with_desc[cat_name] = description

        logging.info(f"Loaded {len(categories_with_desc)} categories with descriptions.")
        # Example log of one category description for verification
        if categories_with_desc:
             first_cat = list(categories_with_desc.keys())[0]
             logging.debug(f"Example description for '{first_cat}': {categories_with_desc[first_cat]}")

        return categories_with_desc
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        return {}
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading categories: {e}")
        return {}


# Function to classify content using OpenAI LLM with retries
def classify_content_with_llm(client, model, content, categories_with_desc, max_retries=3, delay=5):
    """Classifies the content into one of the provided categories using LLM, leveraging category descriptions."""
    if not client or not model:
        logging.error("LLM client or model not initialized. Cannot classify content.")
        return "Classification Error"
    if not categories_with_desc:
        logging.warning("No categories with descriptions provided.")
        return "No Categories Defined"

    # Build the category list with descriptions for the prompt
    category_details_str = "\n".join([f"- **{name}**: {desc}" for name, desc in categories_with_desc.items()])

    prompt = f"""
请根据以下提供的分类定义，仔细阅读文本内容，判断其**主要**归属于哪个分类。

**分类定义**: {category_details_str}

**待分类文本内容**:
---
{content[:2000]} # 注意: 文本内容可能被截断以适应处理长度限制
---

**任务要求**: 
1.  理解每个分类标签旁边的描述，把握其核心内容和侧重点。
2.  分析待分类文本的主要议题和核心信息。
3.  从上面的分类定义列表中，选择**唯一一个与文本主要内容最匹配**的分类标签。
4.  **仅输出你选择的分类标签的名称**。例如，如果文本主要关于应急预案及其流程，直接输出 `Emergency_Response_Plans`。
5.  确保输出结果就是标签名称本身，不包含任何其他文字、解释、列表标记或格式。

**输出 (仅分类标签名称)**:
"""
    # print(prompt) # Uncomment for debugging the prompt

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本分类引擎。你的任务是严格按照提供的分类定义，将用户输入的文本精准地归入最合适的唯一类别，并只输出该类别的标签名称。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Keep low temperature for consistency
                max_tokens=100 # Increased slightly, but should only need category name
            )
            category = response.choices[0].message.content.strip()

            # Refined Validation: Check if the exact returned category is in our list of keys
            if category in categories_with_desc:
                logging.info(f"Successfully classified content chunk into: {category}")
                return category
            else:
                # Try cleaning potential markdown or quotes from the response
                cleaned_category = category.strip("`'\"* ")
                if cleaned_category in categories_with_desc:
                    logging.warning(f"LLM returned '{category}', but matched cleaned version '{cleaned_category}'. Using cleaned version.")
                    return cleaned_category

                logging.warning(f"LLM returned an invalid or unexpected category '{category}'. Attempting fallback by checking substrings.")

                # Enhanced Fallback: Check if any valid category is a substring of the response, or vice versa
                # Prioritize exact match within the response if multiple valid categories are substrings
                possible_matches = []
                for valid_cat in categories_with_desc.keys():
                    if valid_cat in category:
                        possible_matches.append(valid_cat)
                    # Also check if the response is a substring of a valid category (less likely but possible)
                    # elif category in valid_cat: # Be cautious with this, might lead to wrong matches
                    #    possible_matches.append(valid_cat)

                if len(possible_matches) == 1:
                    chosen_cat = possible_matches[0]
                    logging.warning(f"Fallback: Matched '{category}' to '{chosen_cat}' based on substring presence.")
                    return chosen_cat
                elif len(possible_matches) > 1:
                     logging.warning(f"Fallback: Ambiguous match for '{category}'. Found multiple possible categories: {possible_matches}. Marking as Unclassified.")
                     return "Unclassified"
                else:
                    logging.warning(f"Fallback: No valid category found as substring for '{category}'. Marking as Unclassified.")
                    return "Unclassified" # Mark as unclassified if no match found

        except Exception as e:
            logging.error(f"Error during LLM classification (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Failed to classify content.")
                return "Classification Error"

    return "Classification Error" # Should not be reached if retries > 0


# Main script logic
def main(input_json_path, output_json_path, config_path="tests/config.yaml"):
    """Main function to load data, classify chunks, and save results."""
    logging.info("Starting chunk classification process...")

    # Initialize OpenAI client and get model name
    client, model = initialize_openai_client()
    if not client or not model:
        logging.error("Exiting due to OpenAI client initialization failure.")
        return

    # Load categories with descriptions
    categories_with_desc = load_categories_with_descriptions(config_path)
    if not categories_with_desc:
        logging.error("Exiting due to failure in loading categories.")
        return

    # Load input JSON data
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded data from {input_json_path}")
    except FileNotFoundError:
        logging.error(f"Input JSON file not found: {input_json_path}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {input_json_path}: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading input JSON: {e}")
        return

    # Process chunks
    chunks = data.get("chunks")
    if not chunks:
        logging.warning(f"No 'chunks' list found in {input_json_path}. Nothing to process.")
        return

    total_chunks = len(chunks)
    logging.info(f"Found {total_chunks} chunks to classify.")

    classified_count = 0
    error_count = 0
    unclassified_count = 0

    for i, chunk in enumerate(chunks):
        content = chunk.get("content")
        chunk_id = chunk.get("chunk_id", f"index_{i}") # Use chunk_id or index for logging
        logging.info(f"Processing chunk {i + 1}/{total_chunks} (ID: {chunk_id})...")

        if not content:
            logging.warning(f"Chunk {chunk_id} has no content. Skipping classification.")
            chunk["category"] = "No Content"
            unclassified_count += 1 # Count as unclassified
            continue

        # Classify content using the updated function
        category = classify_content_with_llm(client, model, content, categories_with_desc)
        chunk["category"] = category # Add category to the chunk

        # Update counters
        if category == "Classification Error":
            error_count += 1
        elif category in ["Unclassified", "No Categories Defined", "No Content"]:
             unclassified_count += 1
        else:
            classified_count +=1


    logging.info(f"Classification finished. Classified: {classified_count}, Unclassified/Skipped: {unclassified_count}, Errors: {error_count}")

    # Save updated data to output JSON file
    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir: 
             os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved classified data to {output_json_path}")
    except IOError as e:
        logging.error(f"Error writing output JSON file {output_json_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving output JSON: {e}")


if __name__ == "__main__":
    # Define input and output file paths
    INPUT_JSON = "data/RegulationDocuments/中国铁路广州局集团有限公司关于发布《广州局集团公司重点、团体用票管理办法》的通知_chunks_merged_with_summary.json"
    OUTPUT_JSON = "data/RegulationDocuments/中国铁路广州局集团有限公司关于发布《广州局集团公司重点、团体用票管理办法》的通知_chunks_classified.json"
    CONFIG_YAML = "tests/config.yaml"

    # Create dummy input file and config if they don't exist (for testing)
    # In a real scenario, these files should already exist.
    # Simplified dummy creation for brevity
    if not os.path.exists(os.path.dirname(INPUT_JSON)):
         os.makedirs(os.path.dirname(INPUT_JSON), exist_ok=True)
    if not os.path.exists(INPUT_JSON):
        try:
            with open(INPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump({"document_info": {}, "chunks": []}, f)
            logging.info(f"Created empty dummy input file: {INPUT_JSON}")
        except IOError as e:
             logging.error(f"Could not create dummy input file {INPUT_JSON}: {e}")

    if not os.path.exists(os.path.dirname(CONFIG_YAML)):
         os.makedirs(os.path.dirname(CONFIG_YAML), exist_ok=True)
    if not os.path.exists(CONFIG_YAML):
         try:
             with open(CONFIG_YAML, 'w', encoding='utf-8') as f:
                 yaml.dump({"category_configs": {}}, f)
             logging.info(f"Created empty dummy config file: {CONFIG_YAML}")
         except IOError as e:
             logging.error(f"Could not create dummy config file {CONFIG_YAML}: {e}")

    # Run the main process only if input and config files exist
    if os.path.exists(INPUT_JSON) and os.path.exists(CONFIG_YAML):
        main(INPUT_JSON, OUTPUT_JSON, CONFIG_YAML)
    else:
        logging.error("Input JSON or Config YAML file not found. Cannot proceed.") 