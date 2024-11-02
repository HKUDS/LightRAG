import os
from tree_sitter_languages import get_parser, get_language
from tree_sitter import Node
import tiktoken
import zipfile
from io import BytesIO
import requests
from dotenv import load_dotenv
import yaml
from typing import Dict, Any, List
from dataclasses import dataclass
from importlib import resources
from language_parsers import get_language_from_file
from repo import get_github_repo

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

    # The number of tokens in the chunk
    token_count: int

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
        Given an AST tree, recursively chunk the tree nodes
        """
        if file_path == "./palmier-io_palmier-vscode-extension/palmier-io-palmier-vscode-extension-c44a998ed5e8ae4254304cc36e6dec6e135374bd/frontend/src/db/cursor.ts":
            print(f"Processing file {file_path}")
        code_str = code_bytes.decode('utf-8', errors='ignore')
        current_index = 0

        def traverse(node: Node) -> List[CodeChunk]:
            nonlocal code_str, language_name, file_path, current_index
            current_token_count = 0
            current_start_byte = -1
            current_end_byte = -1
            new_chunks: List[CodeChunk] = []

            for child in node.children:
                text = code_str[child.start_byte:child.end_byte]
                tokens = self.encoding.encode(text)
                token_count = len(tokens)

                # Next child node is too big, so we need to recursively traverse the child nodes
                if token_count > self.max_tokens:
                    new_chunks.append(CodeChunk(
                        index=current_index,
                        file_path=file_path,
                        content=code_str[current_start_byte:current_end_byte],
                        token_count=current_token_count,
                        tag={"language": language_name}
                    ))
                    current_index += 1
                    current_token_count = token_count

                    current_start_byte = -1
                    current_end_byte = -1

                    new_chunks.extend(traverse(child))
                # Current chunk is too big, so we need to start a new chunk
                elif current_token_count + token_count > self.max_tokens:
                    new_chunks.append(CodeChunk(
                        index=current_index,
                        file_path=file_path,
                        content=code_str[current_start_byte:current_end_byte],
                        token_count=current_token_count,
                        tag={"language": language_name}
                    ))
                    current_index += 1
                    current_token_count = token_count

                    current_start_byte = child.start_byte
                    current_end_byte = child.end_byte
                # Otherwise, we can concatenate the current chunk with the next child node
                else:
                    if current_start_byte == -1:
                        current_start_byte = child.start_byte
                    current_end_byte = child.end_byte
                    current_token_count += token_count
            


            return new_chunks
    
        return traverse(tree.root_node)

    def process_files(self):
        files = self.traverse_directory()

        for file_path in files:
            language_name = get_language_from_file(file_path)

            if language_name is None:
                print(f"Skipping file {file_path} (unknown language)")
                continue
   
            try:
                language = get_language(language_name)
                parser = get_parser(language_name)
            except LookupError:
                print(f"Parser not found for language: {language_name}")
                continue

            # if language_name != 'python':
            #     continue
            with open(file_path, 'rb') as f:
                code_bytes = f.read()
            tree = parser.parse(code_bytes)

            # query_scm_filename = get_scm_fname(language_name)
            # if not os.path.exists(query_scm_filename):
            #     continue
            # with open(query_scm_filename, 'r') as f:
            #     query_scm = f.read()

            # query = language.query(query_scm)
            # captures = query.captures(tree.root_node)

            # captures = list(captures)
            # for node, tag in captures:
            #     start = node.start_byte
            #     end = node.end_byte
            #     print(f"Found node with tag {tag}:\n---\n {code_bytes[start:end].decode('utf-8')}\n---\n")


            chunks: List[CodeChunk] = self.chunk_code(tree, code_bytes, language_name, file_path)
            for chunk in chunks:
                # Create a sanitized file name
                sanitized_file_path = file_path.replace(self.root_dir, '').strip(os.sep).replace(os.sep, '_')
                output_file_name = f"{sanitized_file_path}_{chunk.index}.txt"
                output_file_path = os.path.join(self.output_path, output_file_name)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # Write the chunk to a file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"File: {chunk.file_path}\n")
                    f.write(f"Chunk: {chunk.index + 1}\n")
                    f.write(f"Language: {chunk.tag['language']}\n")
                    f.write(f"Tokens: {chunk.token_count}\n")
                    f.write("\n")
                    f.write(chunk.content)
                
                print(f"Wrote chunk {chunk.index + 1} of file {chunk.file_path} to {output_file_path}")

def get_scm_fname(lang):
    # Load the tags queries
    try:
        return os.path.join(os.path.dirname(__file__), "queries", f"tree-sitter-{lang}-tags.scm")
    except KeyError:
        return

if __name__ == '__main__':
    load_dotenv()

    owner = config['github_repo']['owner']
    repo = config['github_repo']['repo']
    local_path = config['input_path']
    root_dir = get_github_repo(owner, repo, local_path)

    chunker = CodeChunker(root_dir)
    chunker.process_files()
