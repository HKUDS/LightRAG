import re
import json
import argparse
import jsonlines

from openai import OpenAI


def _preview(items, limit=3):
    items = [repr(item) for item in items]
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _align_answers(queries, rows, source):
    """Return the answers of ``rows`` in the order of ``queries``.

    Step_3 leaves a failed query out of its result file, so answers must be
    matched by their saved ``query``, never by position. Every question needs
    exactly one answer; anything else raises before any request is built.
    """
    known = set(queries)
    answers = {}
    for index, row in enumerate(rows):
        query = row.get("query") if isinstance(row, dict) else None
        if query not in known:
            raise ValueError(
                f"{source}: entry {index} answers a question that is not in the "
                f"question file: {query!r}"
            )
        if query in answers:
            raise ValueError(f"{source}: question answered twice: {query!r}")
        if "result" not in row:
            raise ValueError(f"{source}: entry {index} has no 'result': {query!r}")
        answers[query] = row["result"]
    missing = [query for query in queries if query not in answers]
    if missing:
        raise ValueError(
            f"{source}: {len(missing)} of {len(queries)} questions have no answer "
            f"({_preview(missing)}); re-run the failed queries recorded in the "
            "Step_3 error file before evaluating"
        )
    return [answers[query] for query in queries]


def batch_eval(query_file, result1_file, result2_file, output_file_path):
    with open(query_file, "r") as f:
        data = f.read()

    # Strip Markdown bold exactly as Step_3.extract_queries does, so the
    # question text matches the "query" saved with each answer.
    queries = re.findall(r"- Question \d+: (.+)", data.replace("**", ""))
    if not queries:
        raise ValueError(f"{query_file}: no '- Question N: ...' lines found")
    duplicates = sorted({query for query in queries if queries.count(query) > 1})
    if duplicates:
        raise ValueError(
            f"{query_file}: questions must be unique to match answers to them; "
            f"repeated: {_preview(duplicates)}"
        )

    with open(result1_file, "r") as f:
        answers1 = json.load(f)
    answers1 = _align_answers(queries, answers1, result1_file)

    with open(result2_file, "r") as f:
        answers2 = json.load(f)
    answers2 = _align_answers(queries, answers2, result2_file)

    # Created only after validation, so a bad input never reaches the API.
    client = OpenAI()

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2)):
        sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
        """

        prompt = f"""
        You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

        Here is the question:
        {query}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """

        request_data = {
            "custom_id": f"request-{i + 1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        }

        requests.append(request_data)

    with jsonlines.open(output_file_path, mode="w") as writer:
        for request in requests:
            writer.write(request)

    print(f"Batch API requests written to {output_file_path}")

    batch_input_file = client.files.create(
        file=open(output_file_path, "rb"), purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "nightly eval job"},
    )

    print(f"Batch {batch.id} has been created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build an OpenAI Batch API request file that compares the answers of "
            "two RAG systems on the same questions."
        )
    )
    parser.add_argument(
        "-q",
        "--query_file",
        type=str,
        required=True,
        help="Questions file produced by Step_2.py.",
    )
    parser.add_argument(
        "-r1",
        "--result1_file",
        type=str,
        required=True,
        help="First system's answers, as produced by Step_3.py.",
    )
    parser.add_argument(
        "-r2",
        "--result2_file",
        type=str,
        required=True,
        help="Second system's answers, as produced by Step_3.py.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="batch_eval_requests.jsonl",
        help="Where to write the Batch API request file.",
    )

    args = parser.parse_args()

    batch_eval(args.query_file, args.result1_file, args.result2_file, args.output_file)
