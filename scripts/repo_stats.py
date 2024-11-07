"""Collects stats about a repo to prepare for cost estimation"""

import os
import requests
from github import Github
import tiktoken
from dotenv import load_dotenv
import ast


def count_functions(file_content, file_extension):
    # TODO: Implement for other languages
    if file_extension == ".py":
        try:
            tree = ast.parse(file_content)
            return len(
                [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            )
        except SyntaxError:
            print("Syntax error in Python file")
            return 0
    else:
        return 0


def count_tokens(file_content):
    enc = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")
    tokens = enc.encode(file_content)
    return len(tokens)


def analyze_repo(repo_url):
    # Load environment variables from .env file
    load_dotenv()
    # Extract owner and repo name from the URL
    _, _, _, owner, repo_name = repo_url.rstrip("/").split("/")
    print(f"Owner: {owner}, Repo: {repo_name}")
    # Initialize GitHub API client using the token from .env
    g = Github(os.getenv("GITHUB_TOKEN"))
    # Get the repository
    repo = g.get_repo(f"{owner}/{repo_name}")
    # Initialize counters
    total_tokens = 0
    file_count = 0
    total_functions = 0
    total_lines = 0  # New counter for lines of code
    # Traverse through all files in the repository
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            # List of common code file extensions
            code_extensions = [
                ".py",
                ".js",
                ".ts",
                ".java",
                ".c",
                ".cpp",
                ".cs",
                ".go",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".rs",
                ".html",
                ".css",
                ".scss",
                ".sql",
            ]

            file_extension = os.path.splitext(file_content.name)[1]
            if file_extension in code_extensions:
                file_count += 1
                # Get the raw content of the file
                raw_content = requests.get(file_content.download_url).text

                # Count tokens
                total_tokens += count_tokens(raw_content)
                # Count lines of code
                total_lines += len(
                    raw_content.splitlines()
                )  # New line to count lines of code
                # Count functions for Python, JavaScript, and TypeScript files
                if file_extension in [".py", ".js", ".ts"]:
                    total_functions += count_functions(raw_content, file_extension)
    # Prepare the output dictionary
    result = {
        "number_of_files": file_count,
        "total_tokens": total_tokens,
        "total_functions": total_functions,
        "total_lines_of_code": total_lines,  # New stat in the result dictionary
    }
    return result


# Example usage
if __name__ == "__main__":
    repo_url = "https://github.com/palmier-io/palmier-vscode-extension"
    stats = analyze_repo(repo_url)
    print(stats)
