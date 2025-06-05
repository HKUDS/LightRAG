"""
Complete MinerU parsing + multimodal content insertion Pipeline

This script integrates:
1. MinerU document parsing
2. Pure text content LightRAG insertion
3. Specialized processing for multimodal content (using different processors)
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
import sys

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger

# Import parser and multimodal processors
from lightrag.mineru_parser import MineruParser

# Import specialized processors
from lightrag.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
)


class RAGAnything:
    """Multimodal Document Processing Pipeline - Complete document parsing and insertion pipeline"""

    def __init__(
        self,
        lightrag: Optional[LightRAG] = None,
        llm_model_func: Optional[Callable] = None,
        vision_model_func: Optional[Callable] = None,
        embedding_func: Optional[Callable] = None,
        working_dir: str = "./rag_storage",
        embedding_dim: int = 3072,
        max_token_size: int = 8192,
    ):
        """
        Initialize Multimodal Document Processing Pipeline

        Args:
            lightrag: Optional pre-initialized LightRAG instance
            llm_model_func: LLM model function for text analysis
            vision_model_func: Vision model function for image analysis
            embedding_func: Embedding function for text vectorization
            working_dir: Working directory for storage (used when creating new RAG)
            embedding_dim: Embedding dimension (used when creating new RAG)
            max_token_size: Maximum token size for embeddings (used when creating new RAG)
        """
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size

        # Set up logging
        setup_logger("RAGAnything")
        self.logger = logging.getLogger("RAGAnything")

        # Create working directory if needed
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        # Use provided LightRAG or mark for later initialization
        self.lightrag = lightrag
        self.modal_processors = {}

        # If LightRAG is provided, initialize processors immediately
        if self.lightrag is not None:
            self._initialize_processors()

    def _initialize_processors(self):
        """Initialize multimodal processors with appropriate model functions"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG instance must be initialized before creating processors"
            )

        # Create different multimodal processors
        self.modal_processors = {
            "image": ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
            ),
            "table": TableModalProcessor(
                lightrag=self.lightrag, modal_caption_func=self.llm_model_func
            ),
            "equation": EquationModalProcessor(
                lightrag=self.lightrag, modal_caption_func=self.llm_model_func
            ),
            "generic": GenericModalProcessor(
                lightrag=self.lightrag, modal_caption_func=self.llm_model_func
            ),
        }

        self.logger.info("Multimodal processors initialized")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")

    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG instance is initialized, create if necessary"""
        if self.lightrag is not None:
            return

        # Validate required functions
        if self.llm_model_func is None:
            raise ValueError(
                "llm_model_func must be provided when LightRAG is not pre-initialized"
            )
        if self.embedding_func is None:
            raise ValueError(
                "embedding_func must be provided when LightRAG is not pre-initialized"
            )

        from lightrag.kg.shared_storage import initialize_pipeline_status

        # Create LightRAG instance with provided functions
        self.lightrag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=self.llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                max_token_size=self.max_token_size,
                func=self.embedding_func,
            ),
        )

        await self.lightrag.initialize_storages()
        await initialize_pipeline_status()

        # Initialize processors after LightRAG is ready
        self._initialize_processors()

        self.logger.info("LightRAG and multimodal processors initialized")

    def parse_document(
        self,
        file_path: str,
        output_dir: str = "./output",
        parse_method: str = "auto",
        display_stats: bool = True,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse document using MinerU

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory
            parse_method: Parse method ("auto", "ocr", "txt")
            display_stats: Whether to display content statistics

        Returns:
            (content_list, md_content): Content list and markdown text
        """
        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Choose appropriate parsing method based on file extension
        ext = file_path.suffix.lower()

        try:
            if ext in [".pdf"]:
                self.logger.info(
                    f"Detected PDF file, using PDF parser (OCR={parse_method == 'ocr'})..."
                )
                content_list, md_content = MineruParser.parse_pdf(
                    file_path, output_dir, use_ocr=(parse_method == "ocr")
                )
            elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
                self.logger.info("Detected image file, using image parser...")
                content_list, md_content = MineruParser.parse_image(
                    file_path, output_dir
                )
            elif ext in [".doc", ".docx", ".ppt", ".pptx"]:
                self.logger.info("Detected Office document, using Office parser...")
                content_list, md_content = MineruParser.parse_office_doc(
                    file_path, output_dir
                )
            else:
                # For other or unknown formats, use generic parser
                self.logger.info(
                    f"Using generic parser for {ext} file (method={parse_method})..."
                )
                content_list, md_content = MineruParser.parse_document(
                    file_path, parse_method=parse_method, output_dir=output_dir
                )

        except Exception as e:
            self.logger.error(f"Error during parsing with specific parser: {str(e)}")
            self.logger.warning("Falling back to generic parser...")
            # If specific parser fails, fall back to generic parser
            content_list, md_content = MineruParser.parse_document(
                file_path, parse_method=parse_method, output_dir=output_dir
            )

        self.logger.info(
            f"Parsing complete! Extracted {len(content_list)} content blocks"
        )
        self.logger.info(f"Markdown text length: {len(md_content)} characters")

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")
            self.logger.info(f"* Markdown content length: {len(md_content)} characters")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        return content_list, md_content

    def _separate_content(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Separate text content and multimodal content

        Args:
            content_list: Content list from MinerU parsing

        Returns:
            (text_content, multimodal_items): Pure text content and multimodal items list
        """
        text_parts = []
        multimodal_items = []

        for item in content_list:
            content_type = item.get("type", "text")

            if content_type == "text":
                # Text content
                text = item.get("text", "")
                if text.strip():
                    text_parts.append(text)
            else:
                # Multimodal content (image, table, equation, etc.)
                multimodal_items.append(item)

        # Merge all text content
        text_content = "\n\n".join(text_parts)

        self.logger.info("Content separation complete:")
        self.logger.info(f"  - Text content length: {len(text_content)} characters")
        self.logger.info(f"  - Multimodal items count: {len(multimodal_items)}")

        # Count multimodal types
        modal_types = {}
        for item in multimodal_items:
            modal_type = item.get("type", "unknown")
            modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

        if modal_types:
            self.logger.info(f"  - Multimodal type distribution: {modal_types}")

        return text_content, multimodal_items

    async def _insert_text_content(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ):
        """
        Insert pure text content into LightRAG

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        self.logger.info("Starting text content insertion into LightRAG...")

        # Use LightRAG's insert method with all parameters
        await self.lightrag.ainsert(
            input=input,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
        )

        self.logger.info("Text content insertion complete")

    async def _process_multimodal_content(
        self, multimodal_items: List[Dict[str, Any]], file_path: str
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        self.logger.info("Starting multimodal content processing...")

        file_name = os.path.basename(file_path)

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = self._get_processor_for_type(content_type)

                if processor:
                    (
                        enhanced_caption,
                        entity_info,
                    ) = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                    )
                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        self.logger.info("Multimodal content processing complete")

    def _get_processor_for_type(self, content_type: str):
        """
        Get appropriate processor based on content type

        Args:
            content_type: Content type

        Returns:
            Corresponding processor instance
        """
        # Direct mapping to corresponding processor
        if content_type == "image":
            return self.modal_processors.get("image")
        elif content_type == "table":
            return self.modal_processors.get("table")
        elif content_type == "equation":
            return self.modal_processors.get("equation")
        else:
            # For other types, use generic processor
            return self.modal_processors.get("generic")

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = "./output",
        parse_method: str = "auto",
        display_stats: bool = True,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
    ):
        """
        Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: MinerU output directory
            parse_method: Parse method
            display_stats: Whether to display content statistics
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided MD5 hash will be generated
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Starting complete document processing: {file_path}")

        # Step 1: Parse document using MinerU
        content_list, md_content = self.parse_document(
            file_path, output_dir, parse_method, display_stats
        )

        # Step 2: Separate text and multimodal content
        text_content, multimodal_items = self._separate_content(content_list)

        # Step 3: Insert pure text content with all parameters
        if text_content.strip():
            file_name = os.path.basename(file_path)
            await self._insert_text_content(
                text_content,
                file_paths=file_name,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )

        # Step 4: Process multimodal content (using specialized processors)
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path)

        self.logger.info(f"Document {file_path} processing complete!")

    async def process_folder_complete(
        self,
        folder_path: str,
        output_dir: str = "./output",
        parse_method: str = "auto",
        display_stats: bool = False,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_workers: int = 1,
    ):
        """
        Process all files in a folder in batch

        Args:
            folder_path: Path to the folder to process
            output_dir: MinerU output directory
            parse_method: Parse method
            display_stats: Whether to display content statistics for each file (recommended False for batch processing)
            split_by_character: Optional character to split text by
            split_by_character_only: If True, split only by the specified character
            file_extensions: List of file extensions to process, e.g. [".pdf", ".docx"]. If None, process all supported formats
            recursive: Whether to recursively process subfolders
            max_workers: Maximum number of concurrent workers
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(
                f"Folder does not exist or is not a valid directory: {folder_path}"
            )

        # Supported file formats
        supported_extensions = {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".doc",
            ".docx",
            ".ppt",
            ".pptx",
            ".txt",
            ".md",
        }

        # Use specified extensions or all supported formats
        if file_extensions:
            target_extensions = set(ext.lower() for ext in file_extensions)
            # Validate if all are supported formats
            unsupported = target_extensions - supported_extensions
            if unsupported:
                self.logger.warning(
                    f"The following file formats may not be fully supported: {unsupported}"
                )
        else:
            target_extensions = supported_extensions

        # Collect all files to process
        files_to_process = []

        if recursive:
            # Recursively traverse all subfolders
            for file_path in folder_path.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in target_extensions
                ):
                    files_to_process.append(file_path)
        else:
            # Process only current folder
            for file_path in folder_path.glob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in target_extensions
                ):
                    files_to_process.append(file_path)

        if not files_to_process:
            self.logger.info(f"No files to process found in {folder_path}")
            return

        self.logger.info(f"Found {len(files_to_process)} files to process")
        self.logger.info("File type distribution:")

        # Count file types
        file_type_count = {}
        for file_path in files_to_process:
            ext = file_path.suffix.lower()
            file_type_count[ext] = file_type_count.get(ext, 0) + 1

        for ext, count in sorted(file_type_count.items()):
            self.logger.info(f"  {ext}: {count} files")

        # Create progress tracking
        processed_count = 0
        failed_files = []

        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(max_workers)

        async def process_single_file(file_path: Path, index: int) -> None:
            """Process a single file"""
            async with semaphore:
                nonlocal processed_count
                try:
                    self.logger.info(
                        f"[{index}/{len(files_to_process)}] Processing: {file_path}"
                    )

                    # Create separate output directory for each file
                    file_output_dir = Path(output_dir) / file_path.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)

                    # Process file
                    await self.process_document_complete(
                        file_path=str(file_path),
                        output_dir=str(file_output_dir),
                        parse_method=parse_method,
                        display_stats=display_stats,
                        split_by_character=split_by_character,
                        split_by_character_only=split_by_character_only,
                    )

                    processed_count += 1
                    self.logger.info(
                        f"[{index}/{len(files_to_process)}] Successfully processed: {file_path}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"[{index}/{len(files_to_process)}] Failed to process: {file_path}"
                    )
                    self.logger.error(f"Error: {str(e)}")
                    failed_files.append((file_path, str(e)))

        # Create all processing tasks
        tasks = []
        for index, file_path in enumerate(files_to_process, 1):
            task = process_single_file(file_path, index)
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Output processing statistics
        self.logger.info("\n===== Batch Processing Complete =====")
        self.logger.info(f"Total files: {len(files_to_process)}")
        self.logger.info(f"Successfully processed: {processed_count}")
        self.logger.info(f"Failed: {len(failed_files)}")

        if failed_files:
            self.logger.info("\nFailed files:")
            for file_path, error in failed_files:
                self.logger.info(f"  - {file_path}: {error}")

        return {
            "total": len(files_to_process),
            "success": processed_count,
            "failed": len(failed_files),
            "failed_files": failed_files,
        }

    async def query_with_multimodal(self, query: str, mode: str = "hybrid") -> str:
        """
        Query with multimodal content support

        Args:
            query: Query content
            mode: Query mode

        Returns:
            Query result
        """
        if self.lightrag is None:
            raise ValueError(
                "No LightRAG instance available. "
                "Please either:\n"
                "1. Provide a pre-initialized LightRAG instance when creating RAGAnything, or\n"
                "2. Process documents first using process_document_complete() or process_folder_complete() "
                "to create and populate the LightRAG instance."
            )

        result = await self.lightrag.aquery(query, param=QueryParam(mode=mode))

        return result

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        if not self.modal_processors:
            return {"status": "Not initialized"}

        info = {
            "status": "Initialized",
            "processors": {},
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        for proc_type, processor in self.modal_processors.items():
            info["processors"][proc_type] = {
                "class": processor.__class__.__name__,
                "supports": self._get_processor_supports(proc_type),
            }

        return info

    def _get_processor_supports(self, proc_type: str) -> List[str]:
        """Get processor supported features"""
        supports_map = {
            "image": [
                "Image content analysis",
                "Visual understanding",
                "Image description generation",
                "Image entity extraction",
            ],
            "table": [
                "Table structure analysis",
                "Data statistics",
                "Trend identification",
                "Table entity extraction",
            ],
            "equation": [
                "Mathematical formula parsing",
                "Variable identification",
                "Formula meaning explanation",
                "Formula entity extraction",
            ],
            "generic": [
                "General content analysis",
                "Structured processing",
                "Entity extraction",
            ],
        }
        return supports_map.get(proc_type, ["Basic processing"])
