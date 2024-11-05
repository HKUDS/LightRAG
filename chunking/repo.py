from github import Github
import os
import zipfile
from io import BytesIO
import requests

def get_github_repo(
        repo: str, 
        branch: str,
        local_path: str,
        github_token: str = ""):
    """
    Download the GitHub repository and extract it to the local directory.
    """

    # Create a directory for the repo
    repo_dir = os.path.join(local_path, repo)
    # if os.path.exists(repo_dir):
    #     print(f"Repository already exists in {repo_dir}")
    #     return repo_dir

    os.makedirs(repo_dir, exist_ok=True)

    # Create a GitHub instance
    g = Github(github_token)

    # Get the repository
    repository = g.get_repo(repo)
    print(f"Repository: {repository}")

    # Get the default branch
    if not branch:
        branch = repository.default_branch

    # Get the zip file
    zip_url = repository.get_archive_link("zipball", ref=branch)

    # Download the zip file
    if github_token:
        response = requests.get(zip_url, headers={'Authorization': f'token {github_token}'})
    else:
        response = requests.get(zip_url)

    response.raise_for_status()

    # Save and extract the zip file
    zip_file = BytesIO(response.content)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(repo_dir)

    print(f"Repository downloaded and extracted to: {repo_dir}")
    return repo_dir