# Code Chunking

This directory contains the code for chunking the code in the GitHub repository.

## High Level
1. Download Github remote repository to local file directory
2. Process each file parse the AST tree using `tree-sitter`
3. Extract the code chunks of each file and output to a `.txt` file for lightRAG consumption

## Usage
1. See `config.yaml` for the configuration
2. Set `GITHUB_TOKEN` in your environment
3. In the root directory, run `pip install -e .` to install dependencies
4. Run `python chunking/code_chunker.py` to execute the script

## Chunking Strategy
1. Given a code file, parse the AST tree
2. Starting from the root node, recursively traverse the tree
3. If the sum token of the current node and next node is within the token size limit, we append the nodes together in the same chunk
4. Repeat until either:
    a. Current chunk + next node exceeds the token limit, in which case we start a new chunk
    b. Next node itself exceeds the token limit, in which case we recursively walk the next node's children and start a new chunk
5. Once the tree is parsed into a list of chunks, we walk through the list and merge small chunks (starting backward because small chunks like funciton signature and comments should be in the beginning of the chunk)
6. If the leaf node is still too big, we won't chunk it and write it as a single chunk. This will fall back to lightRAG's default chunking strategy.
