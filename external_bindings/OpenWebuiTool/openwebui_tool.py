"""
OpenWebui Lightrag Integration Tool
==================================

This tool enables the integration and use of Lightrag within the OpenWebui environment,
providing a seamless interface for RAG (Retrieval-Augmented Generation) operations.

Author: ParisNeo (parisneoai@gmail.com)
Social:
    - Twitter: @ParisNeo_AI
    - Reddit: r/lollms
    - Instagram: https://www.instagram.com/parisneo_ai/

License: Apache 2.0
Copyright (c) 2024-2025 ParisNeo

This tool is part of the LoLLMs project (Lord of Large Language and Multimodal Systems).
For more information, visit: https://github.com/ParisNeo/lollms

Requirements:
    - Python 3.8+
    - OpenWebui
    - Lightrag
"""

# Tool version
__version__ = "1.0.0"
__author__ = "ParisNeo"
__author_email__ = "parisneoai@gmail.com"
__description__ = "Lightrag integration for OpenWebui"


import requests
import json
from pydantic import BaseModel, Field
from typing import Callable, Any, Literal, Union, List, Tuple


class StatusEventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class MessageEventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, content="Some message"):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "message",
                    "data": {
                        "content": content,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        LIGHTRAG_SERVER_URL: str = Field(
            default="http://localhost:9621/query",
            description="The base URL for the LightRag server",
        )
        MODE: Literal["naive", "local", "global", "hybrid"] = Field(
            default="hybrid",
            description="The mode to use for the LightRag query. Options: naive, local, global, hybrid",
        )
        ONLY_NEED_CONTEXT: bool = Field(
            default=False,
            description="If True, only the context is needed from the LightRag response",
        )
        DEBUG_MODE: bool = Field(
            default=False,
            description="If True, debugging information will be emitted",
        )
        KEY: str = Field(
            default="",
            description="Optional Bearer Key for authentication",
        )
        MAX_ENTITIES: int = Field(
            default=5,
            description="Maximum number of entities to keep",
        )
        MAX_RELATIONSHIPS: int = Field(
            default=5,
            description="Maximum number of relationships to keep",
        )
        MAX_SOURCES: int = Field(
            default=3,
            description="Maximum number of sources to keep",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "LightRag-Tool/1.0",
        }

    async def query_lightrag(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Query the LightRag server and retrieve information.
        This function must be called before answering the user question
        :params query: The query string to send to the LightRag server.
        :return: The response from the LightRag server in Markdown format or raw response.
        """
        self.status_emitter = StatusEventEmitter(__event_emitter__)
        self.message_emitter = MessageEventEmitter(__event_emitter__)

        lightrag_url = self.valves.LIGHTRAG_SERVER_URL
        payload = {
            "query": query,
            "mode": str(self.valves.MODE),
            "stream": False,
            "only_need_context": self.valves.ONLY_NEED_CONTEXT,
        }
        await self.status_emitter.emit("Initializing Lightrag query..")

        if self.valves.DEBUG_MODE:
            await self.message_emitter.emit(
                "### Debug Mode Active\n\nDebugging information will be displayed.\n"
            )
            await self.message_emitter.emit(
                "#### Payload Sent to LightRag Server\n```json\n"
                + json.dumps(payload, indent=4)
                + "\n```\n"
            )

        # Add Bearer Key to headers if provided
        if self.valves.KEY:
            self.headers["Authorization"] = f"Bearer {self.valves.KEY}"

        try:
            await self.status_emitter.emit("Sending request to LightRag server")

            response = requests.post(
                lightrag_url, json=payload, headers=self.headers, timeout=120
            )
            response.raise_for_status()
            data = response.json()
            await self.status_emitter.emit(
                status="complete",
                description="LightRag query Succeeded",
                done=True,
            )

            # Return parsed Markdown if ONLY_NEED_CONTEXT is True, otherwise return raw response
            if self.valves.ONLY_NEED_CONTEXT:
                try:
                    if self.valves.DEBUG_MODE:
                        await self.message_emitter.emit(
                            "#### LightRag Server Response\n```json\n"
                            + data["response"]
                            + "\n```\n"
                        )
                except Exception as ex:
                    if self.valves.DEBUG_MODE:
                        await self.message_emitter.emit(
                            "#### Exception\n" + str(ex) + "\n"
                        )
                    return f"Exception: {ex}"
                return data["response"]
            else:
                if self.valves.DEBUG_MODE:
                    await self.message_emitter.emit(
                        "#### LightRag Server Response\n```json\n"
                        + data["response"]
                        + "\n```\n"
                    )
                await self.status_emitter.emit("Lightrag query success")
                return data["response"]

        except requests.exceptions.RequestException as e:
            await self.status_emitter.emit(
                status="error",
                description=f"Error during LightRag query: {str(e)}",
                done=True,
            )
            return json.dumps({"error": str(e)})

    def extract_code_blocks(
        self, text: str, return_remaining_text: bool = False
    ) -> Union[List[dict], Tuple[List[dict], str]]:
        """
        This function extracts code blocks from a given text and optionally returns the text without code blocks.

        Parameters:
        text (str): The text from which to extract code blocks. Code blocks are identified by triple backticks (```).
        return_remaining_text (bool): If True, also returns the text with code blocks removed.

        Returns:
        Union[List[dict], Tuple[List[dict], str]]:
            - If return_remaining_text is False: Returns only the list of code block dictionaries
            - If return_remaining_text is True: Returns a tuple containing:
                * List of code block dictionaries
                * String containing the text with all code blocks removed

        Each code block dictionary contains:
            - 'index' (int): The index of the code block in the text
            - 'file_name' (str): The name of the file extracted from the preceding line, if available
            - 'content' (str): The content of the code block
            - 'type' (str): The type of the code block
            - 'is_complete' (bool): True if the block has a closing tag, False otherwise
        """
        remaining = text
        bloc_index = 0
        first_index = 0
        indices = []
        text_without_blocks = text

        # Find all code block delimiters
        while len(remaining) > 0:
            try:
                index = remaining.index("```")
                indices.append(index + first_index)
                remaining = remaining[index + 3 :]
                first_index += index + 3
                bloc_index += 1
            except Exception:
                if bloc_index % 2 == 1:
                    index = len(remaining)
                    indices.append(index)
                remaining = ""

        code_blocks = []
        is_start = True

        # Process code blocks and build text without blocks if requested
        if return_remaining_text:
            text_parts = []
            last_end = 0

        for index, code_delimiter_position in enumerate(indices):
            if is_start:
                block_infos = {
                    "index": len(code_blocks),
                    "file_name": "",
                    "section": "",
                    "content": "",
                    "type": "",
                    "is_complete": False,
                }

                # Store text before code block if returning remaining text
                if return_remaining_text:
                    text_parts.append(text[last_end:code_delimiter_position].strip())

                # Check the preceding line for file name
                preceding_text = text[:code_delimiter_position].strip().splitlines()
                if preceding_text:
                    last_line = preceding_text[-1].strip()
                    if last_line.startswith("<file_name>") and last_line.endswith(
                        "</file_name>"
                    ):
                        file_name = last_line[
                            len("<file_name>") : -len("</file_name>")
                        ].strip()
                        block_infos["file_name"] = file_name
                    elif last_line.startswith("## filename:"):
                        file_name = last_line[len("## filename:") :].strip()
                        block_infos["file_name"] = file_name
                    if last_line.startswith("<section>") and last_line.endswith(
                        "</section>"
                    ):
                        section = last_line[
                            len("<section>") : -len("</section>")
                        ].strip()
                        block_infos["section"] = section

                sub_text = text[code_delimiter_position + 3 :]
                if len(sub_text) > 0:
                    try:
                        find_space = sub_text.index(" ")
                    except Exception:
                        find_space = int(1e10)
                    try:
                        find_return = sub_text.index("\n")
                    except Exception:
                        find_return = int(1e10)
                    next_index = min(find_return, find_space)
                    if "{" in sub_text[:next_index]:
                        next_index = 0
                    start_pos = next_index

                    if code_delimiter_position + 3 < len(text) and text[
                        code_delimiter_position + 3
                    ] in ["\n", " ", "\t"]:
                        block_infos["type"] = "language-specific"
                    else:
                        block_infos["type"] = sub_text[:next_index]

                    if index + 1 < len(indices):
                        next_pos = indices[index + 1] - code_delimiter_position
                        if (
                            next_pos - 3 < len(sub_text)
                            and sub_text[next_pos - 3] == "`"
                        ):
                            block_infos["content"] = sub_text[
                                start_pos : next_pos - 3
                            ].strip()
                            block_infos["is_complete"] = True
                        else:
                            block_infos["content"] = sub_text[
                                start_pos:next_pos
                            ].strip()
                            block_infos["is_complete"] = False

                        if return_remaining_text:
                            last_end = indices[index + 1] + 3
                    else:
                        block_infos["content"] = sub_text[start_pos:].strip()
                        block_infos["is_complete"] = False

                        if return_remaining_text:
                            last_end = len(text)

                    code_blocks.append(block_infos)
                is_start = False
            else:
                is_start = True

        if return_remaining_text:
            # Add any remaining text after the last code block
            if last_end < len(text):
                text_parts.append(text[last_end:].strip())
            # Join all non-code parts with newlines
            text_without_blocks = "\n".join(filter(None, text_parts))
            return code_blocks, text_without_blocks

        return code_blocks

    def clean(self, csv_content: str):
        lines = csv_content.splitlines()
        if lines:
            # Remove spaces around headers and ensure no spaces between commas
            header = ",".join([col.strip() for col in lines[0].split(",")])
            lines[0] = header  # Replace the first line with the cleaned header
            csv_content = "\n".join(lines)
        return csv_content
