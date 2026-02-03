# Takes lines like the following from prompt.py and creates the json required to ingest categories into a LightRAG index using _1_custom_index_01.py
#    "category_hub is the hub entity with an entity_type of category_hub and which shares a relationship with every entity that has the entity_type of category.",
#    "place is an entity with an entity_type of category which describes any geographic location.",
#    "event is an entity with an entity_type of category which describes a particular situation at a specific time.",
#    "anatomy is an entity with an entity_type of category which describes any part of a living organism.",


import json


def generate_category_json(input_filepath, output_filepath):
    """
    Generates a JSON file containing chunks, entities, and relationships
    based on a list of category descriptions.

    Args:
        input_filepath (str): The path to the input text file.
        output_filepath (str): The path to the output JSON file.
    """
    chunks_content = []
    entities = []
    relationships = []

    # Hardcode category_hub's correct entry as it's a special case
    category_hub_entity = {
        "entity_name": "category_hub",
        "entity_type": "category_hub",
        "description": "category_hub is the hub entity with an entity_type of category_hub and which shares a relationship with every entity that has the entity_type of category.",
        "source_id": "category_data.json",
    }

    # Add category_hub's entity first
    entities.append(category_hub_entity)
    chunks_content.append(category_hub_entity["description"])

    with open(input_filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        entity_name = ""
        entity_type = ""
        description = ""

        # Skip the original category_hub line, as we've hardcoded it
        if "category_hub (the entity_name is category_hub)" in line:
            continue

        # Remove leading/trailing ".,", "," or "." and quotes if they exist on the raw line
        line = line.strip().strip(",.").strip('"')

        # General parsing for other entities
        parts = line.split(" is an entity with an entity_type of ", 1)
        if len(parts) < 2:
            # This line doesn't conform to the expected "is an entity with an entity_type of" format, skip it
            continue

        entity_name = parts[0].strip().strip('"')  # Strip quotes from entity_name
        type_and_desc_part = parts[1]

        # Determine the correct description split keyword
        description_split_keyword = ""
        if " which describes " in type_and_desc_part:
            description_split_keyword = " which describes "
        elif " which describe " in type_and_desc_part:
            description_split_keyword = " which describe "

        # Only proceed if a valid keyword was found
        if not description_split_keyword:
            # If neither keyword is found, this line doesn't fit the pattern for description extraction
            continue

        type_desc_split = type_and_desc_part.split(description_split_keyword, 1)

        if len(type_desc_split) < 2:
            # This should ideally not happen if description_split_keyword was found,
            # but as a safeguard.
            continue

        entity_type = type_desc_split[0].strip().strip('"')
        description_suffix = (
            type_desc_split[1].strip().strip('",.')
        )  # Strip quotes and punctuation from suffix

        # Reconstruct the description correctly without extra punctuation from source
        # Use the actual split keyword in the reconstructed description for accuracy
        description = f"{entity_name} is an entity with an entity_type of {entity_type}{description_split_keyword}{description_suffix}."

        # Add to entities list
        entity_entry = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_id": "category_data.json",
        }
        entities.append(entity_entry)

        # Add to chunks content (only for included entities)
        chunks_content.append(description)

        # Add to relationships if it's a category entity and not category_hub
        if entity_type == "category" and entity_name != "category_hub":
            relationship_entry = {
                "src_id": "category_hub",
                "tgt_id": entity_name,
                "description": f"{entity_name} is an element of the set category_hub",
                "keywords": f"{entity_name}, element of, category_hub",
                "weight": 7.0,
                "source_id": "category_data.json",
            }
            relationships.append(relationship_entry)

    # Construct the final JSON structure
    output_data = {
        "chunks": [
            {"content": "\n".join(chunks_content), "source_id": "category_data.json"}
        ],
        "entities": entities,
        "relationships": relationships,
    }

    # Write the JSON to the output file
    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    input_file = "test_categories.txt"
    output_file = "test_categories.json"
    generate_category_json(input_file, output_file)
    print(f"JSON data successfully generated in '{output_file}'")
