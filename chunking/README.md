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
