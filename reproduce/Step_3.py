import re
import json
import asyncio
from lightrag import LightRAG, QueryParam
from tqdm import tqdm


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result, context = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result, "context": context}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    with open(output_file, "a", encoding="utf-8") as result_file, open(
        error_file, "a", encoding="utf-8"
    ) as err_file:
        result_file.write("[\n")
        first_entry = True

        for query_text in tqdm(queries, desc="Processing queries", unit="query"):
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


if __name__ == "__main__":
    cls = "agriculture"
    mode = "hybrid"
    WORKING_DIR = f"../{cls}"

    rag = LightRAG(working_dir=WORKING_DIR)
    query_param = QueryParam(mode=mode)

    queries = extract_queries(f"../datasets/questions/{cls}_questions.txt")
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{cls}_result.json", f"{cls}_errors.json"
    )
