import os
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./custom_kg"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

custom_kg = {
    "entities": [
        {
            "entity_name": "CompanyA",
            "entity_type": "Organization",
            "description": "A major technology company",
            "source_id": "Source1",
        },
        {
            "entity_name": "ProductX",
            "entity_type": "Product",
            "description": "A popular product developed by CompanyA",
            "source_id": "Source1",
        },
        {
            "entity_name": "PersonA",
            "entity_type": "Person",
            "description": "A renowned researcher in AI",
            "source_id": "Source2",
        },
        {
            "entity_name": "UniversityB",
            "entity_type": "Organization",
            "description": "A leading university specializing in technology and sciences",
            "source_id": "Source2",
        },
        {
            "entity_name": "CityC",
            "entity_type": "Location",
            "description": "A large metropolitan city known for its culture and economy",
            "source_id": "Source3",
        },
        {
            "entity_name": "EventY",
            "entity_type": "Event",
            "description": "An annual technology conference held in CityC",
            "source_id": "Source3",
        },
        {
            "entity_name": "CompanyD",
            "entity_type": "Organization",
            "description": "A financial services company specializing in insurance",
            "source_id": "Source4",
        },
        {
            "entity_name": "ServiceZ",
            "entity_type": "Service",
            "description": "An insurance product offered by CompanyD",
            "source_id": "Source4",
        },
    ],
    "relationships": [
        {
            "src_id": "CompanyA",
            "tgt_id": "ProductX",
            "description": "CompanyA develops ProductX",
            "keywords": "develop, produce",
            "weight": 1.0,
            "source_id": "Source1",
        },
        {
            "src_id": "PersonA",
            "tgt_id": "UniversityB",
            "description": "PersonA works at UniversityB",
            "keywords": "employment, affiliation",
            "weight": 0.9,
            "source_id": "Source2",
        },
        {
            "src_id": "CityC",
            "tgt_id": "EventY",
            "description": "EventY is hosted in CityC",
            "keywords": "host, location",
            "weight": 0.8,
            "source_id": "Source3",
        },
        {
            "src_id": "CompanyD",
            "tgt_id": "ServiceZ",
            "description": "CompanyD provides ServiceZ",
            "keywords": "provide, offer",
            "weight": 1.0,
            "source_id": "Source4",
        },
    ],
}

rag.insert_custom_kg(custom_kg)
