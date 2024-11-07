import os
import requests
from github import Github
from dotenv import load_dotenv


def chunk_repo(repo_url):
    # Load environment variables from .env file
    load_dotenv()
    # Extract owner and repo name from the URL
    _, _, _, owner, repo_name = repo_url.rstrip("/").split("/")
    print(f"Owner: {owner}, Repo: {repo_name}")
    # Initialize GitHub API client using the token from .env
    g = Github(os.getenv("GITHUB_TOKEN"))
    # Get the repository
    repo = g.get_repo(f"{owner}/{repo_name}")
    # Create output directory if it doesn't exist
    output_dir = "scripts/output"
    os.makedirs(output_dir, exist_ok=True)
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
    # Traverse through all files in the repository
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file_extension = os.path.splitext(file_content.name)[1]
            if file_extension in code_extensions:
                # Get the raw content of the file
                raw_content = requests.get(file_content.download_url).text

                # Create a unique filename for the output
                output_filename = (
                    f"{output_dir}/{file_content.path.replace('/', '_')}.txt"
                )

                # Write metadata and file contents to the output file
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(f"File Path: {file_content.path}\n")
                    f.write("\n--- File Contents ---\n\n")
                    f.write(raw_content)
                print(f"Processed: {file_content.path}")
    print("Repository chunking completed.")


# Example usage
if __name__ == "__main__":
    repo_url = "https://github.com/palmier-io/palmier-vscode-extension"
    chunk_repo(repo_url)
