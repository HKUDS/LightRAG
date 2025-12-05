from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = ""


def openai_complete_if_cache(
    model='gpt-4o-mini', prompt=None, system_prompt=None, history_messages=None, **kwargs
) -> str:
    if history_messages is None:
        history_messages = []
    openai_client = OpenAI()

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    response = openai_client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


if __name__ == '__main__':
    description = ''
    prompt = f"""
    Given the following description of a dataset:

    {description}

    Please identify 5 potential users who would engage with this dataset. For each user, list 5 tasks they would perform with this dataset. Then, for each (user, task) combination, generate 5 questions that require a high-level understanding of the entire dataset.

    Output the results in the following structure:
    - User 1: [user description]
        - Task 1: [task description]
            - Question 1:
            - Question 2:
            - Question 3:
            - Question 4:
            - Question 5:
        - Task 2: [task description]
            ...
        - Task 5: [task description]
    - User 2: [user description]
        ...
    - User 5: [user description]
        ...
    """

    result = openai_complete_if_cache(model='gpt-4o-mini', prompt=prompt)

    file_path = './queries.txt'
    with open(file_path, 'w') as file:
        file.write(result)

    print(f'Queries written to {file_path}')
