import json
import logging
import re
from pathlib import Path

import jsonlines
from openai import OpenAI

logger = logging.getLogger(__name__)


def batch_eval(query_file, result1_file, result2_file, output_file_path, client: OpenAI | None = None):
    client = client or OpenAI()

    for path in (query_file, result1_file, result2_file):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Input file not found: {path}')

    try:
        with open(query_file, encoding='utf-8') as f:
            data = f.read()
    except Exception as exc:
        logger.error(f'Failed to read query file {query_file}: {exc}')
        raise

    queries = re.findall(r'- Question \d+: (.+)', data)

    try:
        with open(result1_file, encoding='utf-8') as f:
            answers1 = json.load(f)
        with open(result2_file, encoding='utf-8') as f:
            answers2 = json.load(f)
    except Exception as exc:
        logger.error(f'Failed to load result files: {exc}')
        raise

    answers1 = [i['result'] for i in answers1]
    answers2 = [i['result'] for i in answers2]

    requests = []
    for i, (query, answer1, answer2) in enumerate(zip(queries, answers1, answers2, strict=True)):
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
            'custom_id': f'request-{i + 1}',
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': 'gpt-4o-mini',
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': prompt},
                ],
            },
        }

        requests.append(request_data)

    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_file_path, mode='w') as writer:
        for request in requests:
            writer.write(request)

    logger.info(f'Batch API requests written to {output_file_path}')

    try:
        with open(output_file_path, 'rb') as f:
            batch_input_file = client.files.create(file=f, purpose='batch')
        batch_input_file_id = batch_input_file.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={'description': 'nightly eval job'},
        )
    except Exception as exc:
        logger.error(f'Error creating batch from {output_file_path}: {exc}')
        raise

    logger.info(f'Batch {batch.id} has been created.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, required=True, help='Path to file containing evaluation queries')
    parser.add_argument('--result1_file', type=str, required=True, help='Path to JSON file with first set of answers')
    parser.add_argument('--result2_file', type=str, required=True, help='Path to JSON file with second set of answers')
    parser.add_argument('--output_file_path', type=str, required=True, help='Output path for batch API requests file')
    args = parser.parse_args()

    batch_eval(args.query_file, args.result1_file, args.result2_file, args.output_file_path)
