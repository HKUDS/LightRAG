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
import yaml
from typing import Dict, Any, List
from dataclasses import dataclass

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

# global config
config = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))

@dataclass
class CodeChunk:

    # The index of the chunk in the file
    index: int

    # The relative path to the file
    file_path: str

    # The content of the chunk
    content: str

    # Any metadata about the chunk
    tag: Dict[str, Any]

class CodeChunker:
    def __init__(self, root_dir, max_tokens=800):

        # Local root directory of where the repo is downloaded to
        self.root_dir = root_dir

        self.output_path = config['output_path']

        # Max tokens per chunk
        self.max_tokens = max_tokens

        # Encoding to calculate token count
        self.encoding = tiktoken.get_encoding('cl100k_base')

        # Supported Languages
        self.languages = ['python', 'javascript', 'typescript', 'tsx']
        
        # Node types to look for in the AST - TODO: Think whether we should parse generic node types instead
        self.language_node_types = {
            'python': ['function_definition', 'class_definition', 'import_statement', 'expression_statement'],
            'javascript': ['function_declaration', 'class_declaration', 'import_declaration', 'expression_statement'],
            'typescript': ['function_declaration', 'class_declaration', 'import_declaration', 'expression_statement'],
            'tsx': ['function_declaration', 'class_declaration', 'import_declaration', 'expression_statement'],
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

        # Handle special case for tsx
        extension = os.path.splitext(file_path)[1]
        if extension == ".tsx":
            return "tsx"

        try:
            lexer = get_lexer_for_filename(file_path)
            language = lexer.name.lower()
            print(language)
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

    def chunk_code(self, tree, code_bytes, language_name, file_path) -> List[CodeChunk]:
        """
        Given an AST tree, recursively traverse the tree and collect chunks of code until we hit max_tokens.
        """

        code_str = code_bytes.decode('utf-8', errors='ignore')
        chunks: List[CodeChunk] = []
        current_chunk = ''
        current_token_count = 0
        chunk_index = 0

        root_node = tree.root_node
        node_types = self.language_node_types.get(language_name, [])

        # Traverse the syntax tree
        def traverse(node):
            nonlocal current_chunk, current_token_count, chunk_index
            if node.type in node_types:
                text = code_str[node.start_byte:node.end_byte]
                tokens = self.encoding.encode(text)
                token_count = len(tokens)
                if current_token_count + token_count > self.max_tokens:
                    if current_chunk:
                        chunks.append(CodeChunk(
                            index=chunk_index,
                            file_path=file_path,
                            content=current_chunk,
                            tag={"language": language_name}
                        ))
                        chunk_index += 1
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
            chunks.append(CodeChunk(
                index=chunk_index,
                file_path=file_path,
                content=current_chunk,
                tag={"language": language_name}
            ))

        return chunks

    def process_files(self):
        files = self.traverse_directory()

        for file_path in files:
            language_name = self.get_language_from_file(file_path)

            if language_name is None:
                print(f"Skipping file {file_path} (unknown language)")
                continue
   
            try:
                parser = get_parser(language_name)
            except LookupError:
                print(f"Parser not found for language: {language_name}")
                continue

            with open(file_path, 'rb') as f:
                code_bytes = f.read()
            tree = parser.parse(code_bytes)
            chunks: List[CodeChunk] = self.chunk_code(tree, code_bytes, language_name, file_path)
            for chunk in chunks:
                # Create a sanitized file name
                sanitized_file_path = file_path.replace(self.root_dir, '').strip(os.sep).replace(os.sep, '_')
                output_file_name = f"{sanitized_file_path}_{chunk.index}.txt"
                output_file_path = os.path.join(self.output_path, output_file_name)
                
                # Write the chunk to a file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"File: {chunk.file_path}\n")
                    f.write(f"Chunk: {chunk.index + 1}\n")
                    f.write(f"Language: {chunk.tag['language']}\n")
                    f.write("\n")
                    f.write(chunk.content)
                
                print(f"Wrote chunk {chunk.index + 1} of file {chunk.file_path} to {output_file_path}")

def get_github_repo():
    """
    Download the GitHub repository and extract it to the local directory.
    """

    owner = config['github_repo']['owner']
    repo = config['github_repo']['repo']
    local_path = config['input_path']

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
    root_dir = get_github_repo()
    chunker = CodeChunker(root_dir)
    chunker.process_files()

if __name__ == '__main__':
    main()
