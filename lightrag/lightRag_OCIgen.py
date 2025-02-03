# ankit
import os
import oci
import json

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.oracle_impl import OracleDB

from images.ingestion.src.alembic.env import config

print(os.getcwd())

WORKING_DIR = "./dickens"
AUTH_TYPE = os.getenv("OCI_AUTH_TYPE", "API_KEY")
OCI_PROFILE = os.getenv("OCI_PROFILE")
REGION = os.getenv("REGION", "us-ashburn-1")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
ENVIRONMENT = os.getenv("ENVIRONMENT")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHATMODEL = "cohere.command-r-plus"
EMBEDMODEL = "cohere.embed-multilingual-v3.0"


class LLMModel:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(LLMModel, cls).__new__(cls)
            cls.llm = ChatOCIGenAI(
                model_id="cohere.command-r-plus",       # CHATMODEL
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id=COMPARTMENT_ID,
                auth_profile=OCI_PROFILE,
                auth_type=AUTH_TYPE,
                model_kwargs={"temperature": 0, "max_tokens": 1000},
            )
        return cls.instance


class OCICohereCommandRLLM:
    template_type = "cohere-command-r"

    def __init__(self, model_name="cohere.command-r-plus"):
        """
        Initialize the object with the model attribute set to "cohere.command-r-plus".
        """
        self.model = model_name
        self.get_client()

    def get_client(self):
        """
        Generates the Oracle Cloud Infrastructure (OCI) client based on the authentication type.
        Initializes the client with the appropriate configuration, signer, service endpoint, retry strategy,
        and timeout based on the given authentication type.
        """
        endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        config = {"region": REGION}
        signer = None
        if AUTH_TYPE == "API_KEY":
            config = oci.config.from_file(profile_name=OCI_PROFILE)
            config["region"] = REGION
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        elif AUTH_TYPE == "INSTANCE_PRINCIPAL":
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                signer=signer,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        elif AUTH_TYPE == "RESOURCE_PRINCIPAL":
            signer = oci.auth.signers.get_resource_principals_signer()
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                signer=signer,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        else:
            # log.error(
            #     "Please provide a valid OCI_AUTH_TYPE from the following : API_KEY, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL")
            print(
                "Please provide a valid OCI_AUTH_TYPE from the following : API_KEY, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL"
            )

    def generate_answer(self, preamble, prompt, documents):
        """
        Generate the chat response using the provided preamble, prompt, and documents.

        Parameters:
            preamble (str): The text to set as the preamble override.
            prompt (str): The text prompt for the chat response.
            documents (list): A list of documents to consider during chat generation.

        Returns:
            str: The generated chat response text.
        """
        # profile = OCI_PROFILE
        compartment_id = COMPARTMENT_ID
        generative_ai_inference_client = self.client
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            preamble_override=preamble
        )
        chat_request.message = prompt

        chat_request.max_tokens = 4000
        chat_request.is_stream = False
        chat_request.temperature = 0.00
        chat_request.top_p = 0.7
        chat_request.top_k = 1  # Only support topK within [0, 500]
        chat_request.frequency_penalty = 1.0
        # chat_request.prompt_truncation = 'AUTO_PRESERVE_ORDER'

        chat_request.documents = documents

        chat_detail.serving_mode = (
            oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model)
        )

        chat_detail.compartment_id = compartment_id
        chat_detail.chat_request = chat_request

        chat_response = generative_ai_inference_client.chat(chat_detail)

        chat_response_vars = vars(chat_response)
        resp_json = json.loads(str(chat_response_vars["data"]))
        res = resp_json["chat_response"]["text"]
        # log.debug(res)
        return res


async def embeddings():
    return await OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",          # EMBEDMODEL
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_ID
    )

async def llm_model_ociCohereRLLM(prompt, preamble, documents):
    commandr = OCICohereCommandRLLM()
    return await commandr.generate_answer(
        preamble=preamble, prompt=prompt, documents=documents
    )

async def llm_model():
    return await LLMModel().llm

# embeddings = OCIGenAIEmbeddings(
#     model_id="cohere.embed-multilingual-v3.0",          # EMBEDMODEL
#     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#     compartment_id=COMPARTMENT_ID
# )


# llm = LLMModel().llm

async def main():

    oracle_db = OracleDB(
        config = {
            "user": ORACLE_DB_USER,
            "password": ORACLE_DB_PASSWORD,
            "dsn": ORACLE_DB_DSN,
        })
    await oracle_db.check_tables()

    rag = LightRAG(
                enable_llm_cache=False,
                working_dir=WORKING_DIR,
                chunk_token_size=512,
                llm_model_func=llm_model,
                embedding_func=EmbeddingFunc(max_token_size=512, embedding_dim=512, func=embeddings),
                graph_storage="OracleGraphStorage",
                kv_storage="OracleKVStorage",
                vector_storage="OracleVectorDBStorage",
            )

    # Setthe KV/vector/graph storage's `db` property, so all operation will use same connection pool
    rag.graph_storage_cls.db = oracle_db
    rag.key_string_value_json_storage_cls.db = oracle_db
    rag.vector_db_storage_cls.db = oracle_db
    # add embedding_func for graph database, it's deleted in commit 5661d76860436f7bf5aef2e50d9ee4a59660146c
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

    with open("./book.txt", "r", encoding="utf-8") as f:
        await rag.insert(f.read())

    modes = ["naive", "local", "global", "hybrid"]
    for mode in modes:
        print("=" * 20, mode, "=" * 20)
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode=mode),))
        print("-" * 100, "\n")




