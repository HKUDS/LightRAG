import os
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound

SUPPORT_LANGUAGES = [
    "bash",
    "c",
    "cpp",
    "c_sharp",
    "css",
    "dockerfile",
    "dot",
    "elisp",
    "elixir",
    "elm",
    "erlang",
    "go",
    "gomod",
    "hack",
    "haskell",
    "hcl",
    "html",
    "java",
    "javascript",
    "javascript",
    "jsdoc",
    "json",
    "julia",
    "kotlin",
    "lua",
    "make",
    "objc",
    "ocaml",
    "perl",
    "php",
    "python",
    "ql",
    "r",
    "r",
    "regex",
    "rst",
    "ruby",
    "rust",
    "scala",
    "sql",
    "sqlite",
    "tsq",
    "typescript"
]

def get_language_from_file(file_path):
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

        if language in SUPPORT_LANGUAGES:
            return language

        return None
    except ClassNotFound:
        return None