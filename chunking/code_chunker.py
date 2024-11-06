import os
from tree_sitter_languages import get_parser
from tree_sitter import Node
import tiktoken
from dotenv import load_dotenv
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from chunking.language_parsers import get_language_from_file, SUPPORT_LANGUAGES, FILES_TO_IGNORE
from chunking.repo import get_github_repo

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

# global config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = load_config(os.path.join(project_root, 'config.yaml'))

@dataclass
class Position:

    # The line number of the position
    line: int

    # The character number within the line
    character: int

    # The byte offset of the position
    byte: int

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

    # The start position of the chunk
    start: Position

    # The end position of the chunk
    end: Position

class CodeChunker:
    def __init__(self, root_dir, max_tokens=800):

        # Local root directory of where the repo is downloaded to
        self.root_dir = root_dir

        # Path to the output directory containing the chunk files
        self.output_path = os.path.join(config['working_dir'], "input")

        # Max tokens per chunk
        self.max_tokens = max_tokens

        # Encoding to calculate token count
        self.encoding = tiktoken.get_encoding('cl100k_base')

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

        # TODO 1: Summarize both file and chunk level with prompt caching

        # TODO 2: Current greedy algorithm will often separate comments and functions since
        #         comments above the function might fit into current chunk, but the function itself does not.
        #         In this case, the comments end up being at the end of 1st chunk, and the function starts at the 2nd chunk
        #         Potential solutions:
        #            1. Overlapping chunk - set a overlapping size limit (e.g. 100 tokens). If the next node is within the limit
        #                                   keep this node at both end of current chunk and beginning of next chunk. But,
        #                                   We need to be careful because comment node is single-lined, so multi-line comments require
        #                                   some looping. Also, merging a node might duplicate the overlap content
        #            2. 

        code_str = code_bytes.decode('utf-8', errors='ignore')
        current_index = 0

        def traverse(node: Node) -> List[CodeChunk]:
            nonlocal code_str, language_name, file_path, current_index
            
            # Keep the leaf node as is, since we can't recursively split it
            # Even if it exceeds max_tokens, lightrag will chunk it as a text
            if len(node.children) == 0:
                text = code_str[node.start_byte:node.end_byte]
                tokens = self.encoding.encode(text)
                token_count = len(tokens)
                
                return [CodeChunk(
                    index=current_index,
                    file_path=file_path,
                    start=Position(line=node.start_point[0], character=node.start_point[1], byte=node.start_byte),
                    end=Position(line=node.end_point[0], character=node.end_point[1], byte=node.end_byte),
                    content=text,
                    token_count=token_count,
                    tag={"language": language_name}
                )]
            
            current_token_count = 0
            current_start_position = None
            current_end_position = None

            new_chunks: List[CodeChunk] = []

            for child in node.children:
                text = code_str[child.start_byte:child.end_byte]
                tokens = self.encoding.encode(text)
                token_count = len(tokens)

                # Next child node is too big, so we need to recursively traverse the child nodes
                if token_count > self.max_tokens:
                    # Current chunk is valid
                    if current_start_position is not None:
                        new_chunks.append(CodeChunk(
                            index=current_index,
                            file_path=file_path,
                            start=current_start_position,
                            end=current_end_position,
                            content=code_str[current_start_position.byte:current_end_position.byte],
                            token_count=current_token_count,
                            tag={"language": language_name}
                        ))
                        current_index += 1

                    # Reset Current Chunk
                    current_start_position = None
                    current_end_position = None
                    current_token_count = 0

                    new_chunks.extend(traverse(child))
                # Current chunk is too big, so we need to start a new chunk
                elif current_token_count + token_count > self.max_tokens:
                    new_chunks.append(CodeChunk(
                        index=current_index,
                        file_path=file_path,
                        start=current_start_position,
                        end=current_end_position,
                        content=code_str[current_start_position.byte:current_end_position.byte],
                        token_count=current_token_count,
                        tag={"language": language_name}
                    ))
                    current_index += 1

                    # The new current chunk will be the next child
                    current_token_count = token_count
                    current_start_position = Position(
                        line=child.start_point[0],
                        character=child.start_point[1],
                        byte=child.start_byte
                    )
                    current_end_position = Position(
                        line=child.end_point[0],
                        character=child.end_point[1],
                        byte=child.end_byte
                    )
                # Otherwise, we can concatenate the current chunk with the next child node
                else:
                    if current_start_position is None:
                        current_start_position = Position(
                            line=child.start_point[0],
                            character=child.start_point[1],
                            byte=child.start_byte
                        )
                    current_end_position = Position(
                        line=child.end_point[0],
                        character=child.end_point[1],
                        byte=child.end_byte
                    )
                    current_token_count += token_count
            
            # Add the final chunk if there's content that hasn't been added yet
            if current_start_position is not None:
                new_chunks.append(CodeChunk(
                    index=current_index,
                    file_path=file_path,
                    start=current_start_position,
                    end=current_end_position,
                    content=code_str[current_start_position.byte:current_end_position.byte],
                    token_count=current_token_count,
                    tag={"language": language_name}
                ))
                current_index += 1
            
            return new_chunks
    
        chunks = traverse(tree.root_node)

        # Merge small chunks
        merged_chunks = self.merge_chunks(chunks)
        
        return merged_chunks

    def merge_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """
        Merge small chunks together while respecting max_tokens limit,
        starting from the end of the list to preserve context.
        Returns a new list of merged chunks.
        """
        if not chunks:
            return []
        
        # Work with reversed list
        chunks = chunks[::-1]
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # If merging wouldn't exceed max_tokens, combine the chunks
            # Note: next_chunk comes before current_chunk since we're going backwards
            if current_chunk.token_count + next_chunk.token_count <= self.max_tokens:
                current_chunk = CodeChunk(
                    index=next_chunk.index,  # Use the earlier index
                    file_path=next_chunk.file_path,
                    start=next_chunk.start,
                    end=current_chunk.end,
                    content=next_chunk.content + current_chunk.content,  # Preserve order
                    token_count=current_chunk.token_count + next_chunk.token_count,
                    tag=next_chunk.tag
                )
            else:
                # Can't merge anymore, add current_chunk to results and start new chunk
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        # Don't forget to add the last chunk
        merged.append(current_chunk)
        
        # Reverse back to original order and update indices
        merged = merged[::-1]
        for i, chunk in enumerate(merged):
            chunk.index = i
            
        return merged

    def process_files(self):
        files = self.traverse_directory()

        llm_summary_enabled = config['llm_summary_enabled']
        file_summary = ""

        print(f"Chunking {len(files)} files")

        for file_path in files:

            relative_file_path = file_path.replace(self.root_dir, '')
            # Remove leading separator and split path
            relative_file_path = relative_file_path.lstrip(os.sep)
            # Remove first folder from path - this is the zip folder name downloaded from GitHub
            relative_file_path = os.sep.join(relative_file_path.split(os.sep)[1:])

            if any(relative_file_path.endswith(ext) for ext in FILES_TO_IGNORE):
                print(f"Skipping file {relative_file_path}")
                continue

            with open(file_path, 'rb') as f:
                code_bytes = f.read()
                code_str = code_bytes.decode('utf-8', errors='ignore')

            language_name = get_language_from_file(file_path)
            chunks: List[CodeChunk] = []

            if language_name is None:
                print(f"Skipping file {relative_file_path}")
                continue

            # For files that are not supported by tree-sitter, we will write the entire file
            # and let lightrag handle the chunking
            if language_name not in SUPPORT_LANGUAGES:
                tokens = self.encoding.encode(code_str)
                token_count = len(tokens)
                code_lines = code_str.split('\n')
                chunks.append(CodeChunk(
                    index=0,
                    file_path=relative_file_path,
                    start=Position(line=0, character=0, byte=0),
                    end=Position(line=len(code_lines), character=len(code_lines[-1]), byte=len(code_str)),
                    content=code_str,
                    token_count=token_count,
                    tag={"language": language_name}
                ))
            else:
                if llm_summary_enabled:
                    file_summary = generate_file_summary(code_str, relative_file_path)

                try:
                    parser = get_parser(language_name)
                except LookupError:
                    print(f"Parser not found for language: {language_name}")
                    continue

                tree = parser.parse(code_bytes)
                chunks.extend(self.chunk_code(tree, code_bytes, language_name, relative_file_path))

            for chunk in chunks:
                # Create a sanitized file name
                sanitized_file_path = file_path.replace(self.root_dir, '').strip(os.sep).replace(os.sep, '_')
                output_file_name = f"{sanitized_file_path}_{chunk.index}.txt"
                output_file_path = os.path.join(self.output_path, output_file_name)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # Write the chunk to a file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"File: {chunk.file_path}\n")
                    if file_summary:
                        f.write(f"Summary: {file_summary}\n")
                    f.write(f"Chunk: {chunk.index + 1}\n")
                    if chunk.start:
                        f.write(f"Start: {chunk.start.line}:{chunk.start.character}\n")
                    if chunk.end:
                        f.write(f"End: {chunk.end.line}:{chunk.end.character}\n")
                    f.write(f"Language: {chunk.tag['language']}\n")
                    f.write(f"Tokens: {chunk.token_count}\n")
                    f.write("\n")
                    f.write(chunk.content)
                
                # print(f"Wrote chunk {chunk.index + 1} of file {chunk.file_path} to {output_file_path}")


def generate_file_summary(code_str: str, file_path: str) -> str:
    import openai

    client = openai.OpenAI()
    model = config['llm_summary_model']

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": f"Please provide a high-level summary of the given code content in file {file_path}. Include key concepts and functionalities, mentioning relevant classes and function names along with their purposes and interactions. Take into account the file path name for context. Keep the summary concise, using no more than 100 words, and format it as a single paragraph."
            },
            {
                "role": "user", 
                "content": code_str
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    load_dotenv()

    owner = config['github_repo']['owner']
    repo = config['github_repo']['repo']
    branch = config['github_repo']['branch']
    local_path = config['working_dir']
    github_token = os.getenv('GITHUB_TOKEN')
    root_dir = get_github_repo(f"{owner}/{repo}", branch, local_path, github_token=github_token)

    chunker = CodeChunker(root_dir)
    chunker.process_files()
