from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import neo4j
import asyncio
import ollama
from openai import AsyncOpenAI
import os
import spacy
from spacy_llm.util import assemble
import json
from collections import Counter
from pathlib import Path


def compute_embeddings_based_on_node(driver, model):
    with driver.session() as session:

        # This is to create an embeddings on each node
        # Retrieve all nodes
        result = session.run("MATCH (n) RETURN n")

        for record in result:
            node = record["n"]
            # Combine node labels and properties into a single string
            node_data = (
                " ".join(node.labels)
                + " "
                + " ".join(f"{k}: {v}" for k, v in node.items())
            )

            node_embedding = model.encode(node_data)

            # Store the embedding back into the node
            session.run(
                f"MATCH (n) WHERE id(n) = {node.element_id} SET n.embedding = {node_embedding.tolist()}"
            )

        session.run("MATCH (n) SET n:Entity")


def find_most_similar_node(driver, question_embedding):

    with driver.session() as session:
        result = session.run(
            f"CALL vector_search.search('tag', 10, {question_embedding.tolist()}) YIELD * RETURN *;"
        )
        nodes_data = []
        for record in result:
            node = record["node"]
            properties = {k: v for k, v in node.items() if k != "embedding"}
            node_data = {
                "distance": record["distance"],
                "id": node.element_id,
                "labels": list(node.labels),
                "properties": properties,
            }
            nodes_data.append(node_data)
        print("All similar nodes:")
        for node in nodes_data:
            print(node)

        return nodes_data[0] if nodes_data else None


def clean_path_nodes(path):
    cleaned_nodes = []
    type(path)
    # print(path.nodes)
    # print(type(path.nodes))
    for node in path.nodes:
        print(path.nodes)
        print(node)
        properties = {k: v for k, v in node.items() if k != "embedding"}
        node_data = {
            "id": node.element_id,
            "labels": list(node.labels),
            "properties": properties,
            "type": node.type,
        }
        cleaned_nodes.append(node_data)

    return clean_path_nodes


def get_relevant_data(driver, node, hops):
    with driver.session() as session:
        query = (
            f"MATCH path=((n)-[r*..{hops}]-(m)) WHERE id(n) = {node['id']} RETURN path"
        )
        result = session.run(query)
        paths = []
        for record in result:
            path_data = []
            for segment in record["path"]:

                # Process start node without 'embedding' property
                start_node_data = {
                    k: v for k, v in segment.start_node.items() if k != "embedding"
                }

                # Process relationship data
                relationship_data = {
                    "type": segment.type,
                    "properties": segment.get("properties", {}),
                }

                # Process end node without 'embedding' property
                end_node_data = {
                    k: v for k, v in segment.end_node.items() if k != "embedding"
                }

                # Add to path_data as a tuple (start_node, relationship, end_node)
                path_data.append((start_node_data, relationship_data, end_node_data))

            paths.append(path_data)

        return paths


def RAG_prompt(question, relevance_expansion_data):
    prompt = f"""
    You are an AI language model. I will provide you with a question and a set of data obtained through a relevance expansion process in a graph database. The relevance expansion process finds nodes connected to a target node within a specified number of hops and includes the relationships between these nodes.

    Question: {question}

    Relevance Expansion Data:
    {relevance_expansion_data}

    Based on the provided data, please answer the question, make sure to base your answers only based on the provided data. Add a context on what data did you base your answer on. If you do not have enough information to answer the question, please state that you do not have enough information to answer the question.
    """
    return prompt


def question_prompt(question):
    prompt = f"""
    You are an AI language model. I will provide you with a question. 
    Extract the key information from the questions. The key information is important information that is required to answer the question.

    Question: {question}

    The output format should be like this: 
    Key Information: [key information 1], [key information 2], ...
    """
    return prompt


async def get_response(client, prompt):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Split document into sentences
def split_document_sent(text, nlp):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def process_text(text, nlp, verbose=False):
    doc = nlp(text)
    if verbose:
        print(f"Text: {doc.text}")
        print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
    return doc


# Pipeline to run entity extraction
def extract_entities(text, nlp, verbose=False):
    processed_data = []
    entity_counts = Counter()

    sentences = split_document_sent(text, nlp)
    for sent in sentences:
        doc = process_text(sent, nlp, verbose)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Store processed data for each sentence
        processed_data.append({"text": doc.text, "entities": entities})

        # Update counters
        entity_counts.update([ent[1] for ent in entities])

    # Export to JSON
    with open("processed_data.json", "w") as f:
        json.dump(processed_data, f)


def generate_cypher_queries(nodes, relationships):
    queries = []

    # Create nodes
    for node in nodes:
        query = f"""
        MERGE (n:{node['type']}:Entity {{name: '{node['name']}'}}) 
        ON CREATE SET n.id={node['id']} 
        ON MATCH SET n.id={node['id']}
        """
        queries.append(query)

    # Create relationships
    for rel in relationships:
        query = (
            f"MATCH (a {{id: {rel['source']}}}), (b {{id: {rel['target']}}}) "
            f"CREATE (a)-[:{rel['relationship']}]->(b)"
        )
        queries.append(query)

    return queries


def main():
    # Create a Neo4j driver
    driver = neo4j.GraphDatabase.driver("bolt://100.64.149.141:7687", auth=("", ""))

    # compute_bigger_embeddings_based_on_node(
    #     driver, SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # )

    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    client = AsyncOpenAI()

    # question = (
    #     "In what episode was the  Robb Stark the victim, what is the name of episode?"
    # )
    # question = "Who killed Viserys Targaryen in Game of thrones?"
    # question = "Who is Viserys Targaryen?"
    # question = "In which episode was Viserys Targaryen killed?"
    # question = "How was Viserys Targaryen killed in Game of Thrones?"
    # question = "What weapon was used to kill Viserys Targaryen in Game of Thrones?"
    # question = "Who betrayed Viserys Targaryen in Game of Thrones?"
    # question = "What was the method used to kill Viserys Targaryen in Game of Thrones?"
    # question = "To whom was Viserys Targaryen loyal to?"
    question = "Is Khal Drogo married?"

    prompt = question_prompt(question)
    response = asyncio.run(get_response(client, prompt))
    print(response)

    key_information = response.split("Key Information: ")[1].strip()

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    question_embedding = model.encode(key_information)

    node = find_most_similar_node(driver, question_embedding)
    if node:
        print("The most similar node is:")
        print(node)

    relevant_data = get_relevant_data(driver, node, hops=1)

    print("The relevant data is:")
    print(relevant_data)

    prompt = RAG_prompt(question, relevant_data)

    response = asyncio.run(get_response(client, prompt))
    print("The response is:")
    print(response)

    # Load the spaCy model
    nlp = spacy.load("en_core_web_md")

    # Sample text summary for processing
    summary = """
        Viserys Targaryen is the last living son of the former king, Aerys II Targaryen (the 'Mad King').
        As one of the last known Targaryen heirs, Viserys Targaryen is obsessed with reclaiming the Iron Throne and 
        restoring his family’s rule over Westeros. Ambitious and arrogant, he often treats his younger sister, Daenerys Targaryen, 
        as a pawn, seeing her only as a means to gain power. His ruthless ambition leads him to make a marriage alliance with 
        Khal Drogo, a powerful Dothraki warlord, hoping Khal Drogo will give him the army he needs. 
        However, Viserys Targaryen’s impatience and disrespect toward the Dothraki culture lead to his downfall;
        he is ultimately killed by Khal Drogo in a brutal display of 'a crown for a king' – having molten gold poured over his head. 
      """

    extract_entities(summary, nlp)

    # Load processed data from JSON
    json_path = Path("processed_data.json")
    with open(json_path, "r") as f:
        processed_data = json.load(f)

    # Prepare nodes and relationships
    nodes = []
    relationships = []

    # Formulate a prompt for GPT-4
    prompt = (
        "Extract entities and relationships from the following JSON data. For each entry in data['entities'], "
        "create a 'node' dictionary with fields 'id' (unique identifier), 'name' (entity text), and 'type' (entity label). "
        "For entities that have meaningful connections, define 'relationships' as dictionaries with 'source' (source node id), "
        "'target' (target node id), and 'relationship' (type of connection). Create max 30 nodes, format relationships in the format of capital letters and _ inbetween words and format the entire response in the JSON output containing only variables nodes and relationships without any text inbetween. Use following labels for nodes: Character, Title, Location, House, Death, Event, Allegiance and following relationship types: HAPPENED_IN, SIBLING_OF, PARENT_OF, MARRIED_TO, HEALED_BY, RULES, KILLED, LOYAL_TO, BETRAYED_BY. Make sure the entire JSON file fits in the output"
        "JSON data:\n"
        f"{json.dumps(processed_data)}"
    )

    response = asyncio.run(get_response(client, prompt))

    structured_data = json.loads(response)  # Assuming GPT-4 outputs structured JSON

    # Populate nodes and relationships lists
    nodes.extend(structured_data.get("nodes", []))
    relationships.extend(structured_data.get("relationships", []))

    cypher_queries = generate_cypher_queries(nodes, relationships)
    with driver.session() as session:
        for query in cypher_queries:
            try:
                session.run(query)
                print(f"Executed query: {query}")
            except Exception as e:
                print(f"Error executing query: {query}. Error: {e}")

    driver.close()


if __name__ == "__main__":
    main()
