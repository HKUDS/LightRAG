import os
from tree_sitter_languages import get_parser, get_language
import tiktoken
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
import zipfile
from github import Github
from io import BytesIO
import requests
from dotenv import load_dotenv

class CodeChunker:
    def __init__(self, root_dir, max_tokens=800):
        self.root_dir = root_dir
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding('cl100k_base')
        self.languages = ['python', 'javascript', 'tsx', 'java', 'cpp', 'rust', 'go', 'bash']
        self.language_node_types = {
            'python': ['function_definition', 'class_definition', 'import_statement', 'expression_statement'],
            'javascript': ['function_declaration', 'class_declaration', 'import_declaration', 'expression_statement'],
            'java': ['class_declaration', 'method_declaration', 'import_declaration'],
            'cpp': ['function_definition', 'class_specifier', 'declaration'],
            # Add more languages as needed
        }

    def get_language_from_file(self, file_path):
        """
        Given a file path, extract the extension and use pygment lexer to identify the language.
        tsx is a special case, so we handle it separately.
        https://pygments.org/languages/
        """
        extension = os.path.splitext(file_path)[1]
        if extension == ".tsx":
            return "tsx"

        try:
            lexer = get_lexer_for_filename(file_path)
            language = lexer.name.lower()
            if language in self.languages:
                return language
            else:
                return None
        except ClassNotFound:
            return None

    def traverse_directory(self):
        """
        Walk the directory and return a list of full file paths.
        """
        file_list = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    def chunk_code(self, tree, code_bytes, language_name):
        """
        Given an AST tree, recursively traverse the tree and collect chunks of code until we hit max_tokens.
        """
        code_str = code_bytes.decode('utf-8', errors='ignore')
        chunks = []
        current_chunk = ''
        current_token_count = 0

        root_node = tree.root_node
        node_types = self.language_node_types.get(language_name, [])

        # Traverse the syntax tree
        def traverse(node):
            nonlocal current_chunk, current_token_count
            if node.type in node_types:
                text = code_str[node.start_byte:node.end_byte]
                tokens = self.encoding.encode(text)
                token_count = len(tokens)
                if current_token_count + token_count > self.max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = text
                    current_token_count = token_count
                else:
                    if current_chunk:
                        current_chunk += '\n' + text
                    else:
                        current_chunk = text
                    current_token_count += token_count
            for child in node.children:
                traverse(child)

        traverse(root_node)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_files(self):
        files = self.traverse_directory()

        for file_path in files:
            language_name = self.get_language_from_file(file_path)

            if language_name is None:
                print(f"Skipping file {file_path} (unknown language)")
                continue

            print(f"Processing file {file_path} with language {language_name}")
   
            try:
                parser = get_parser(language_name)
            except LookupError:
                print(f"Parser not found for language: {language_name}")
                continue

            with open(file_path, 'rb') as f:
                code_bytes = f.read()
            tree = parser.parse(code_bytes)
            chunks = self.chunk_code(tree, code_bytes, language_name)
            for i, chunk in enumerate(chunks):
                # Process the chunk as needed (e.g., save to a file, analyze, etc.)
                print(f"Chunk {i+1} of file {file_path}:")
                print(chunk)
                print('---')

def get_github_repo(local_path, repo_url):
    # Extract owner and repo name from the URL
    _, _, _, owner, repo = repo_url.rstrip('/').split('/')

    # Create a directory for the repo
    repo_dir = os.path.join(local_path, f"{owner}_{repo}")
    if os.path.exists(repo_dir):
        print(f"Repository already exists in {repo_dir}")
        return repo_dir

    os.makedirs(repo_dir, exist_ok=True)

    # Get the GitHub token from environment variable
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("GitHub token not found in environment variables")

    # Create a GitHub instance
    g = Github(github_token)

    # Get the repository
    repository = g.get_repo(f"{owner}/{repo}")

    # Get the default branch
    default_branch = repository.default_branch

    # Get the zip file
    zip_url = repository.get_archive_link("zipball", ref=default_branch)
    
    # Download the zip file
    response = requests.get(zip_url, headers={'Authorization': f'token {github_token}'})
    response.raise_for_status()

    # Save and extract the zip file
    zip_file = BytesIO(response.content)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(repo_dir)

    print(f"Repository downloaded and extracted to: {repo_dir}")
    return repo_dir

def main():
    load_dotenv()
    repo_url = "https://github.com/palmier-io/palmier-vscode-extension"
    local_path = "/Users/harrisontin/palmierio/palmier-lightrag"
    root_dir = get_github_repo(local_path, repo_url)
    chunker = CodeChunker(root_dir)
    chunker.process_files()

if __name__ == '__main__':
    main()
