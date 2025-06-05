# type: ignore
"""
MinerU Document Parser Utility

This module provides functionality for parsing PDF, image and office documents using MinerU library,
and converts the parsing results into markdown and JSON formats
"""

from __future__ import annotations

__all__ = ["MineruParser"]

import os
import json
import argparse
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Any,
    TypeVar,
    cast,
    TYPE_CHECKING,
    ClassVar,
)

# Type stubs for magic_pdf
FileBasedDataWriter = Any
FileBasedDataReader = Any
PymuDocDataset = Any
InferResult = Any
PipeResult = Any
SupportedPdfParseMethod = Any
doc_analyze = Any
read_local_office = Any
read_local_images = Any

if TYPE_CHECKING:
    from magic_pdf.data.data_reader_writer import (
        FileBasedDataWriter,
        FileBasedDataReader,
    )
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.data.read_api import read_local_office, read_local_images
else:
    # MinerU imports
    from magic_pdf.data.data_reader_writer import (
        FileBasedDataWriter,
        FileBasedDataReader,
    )
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.data.read_api import read_local_office, read_local_images

T = TypeVar("T")


class MineruParser:
    """
    MinerU document parsing utility class

    Supports parsing PDF, image and office documents (like Word, PPT, etc.),
    converting the content into structured data and generating markdown and JSON output
    """

    __slots__: ClassVar[Tuple[str, ...]] = ()

    def __init__(self) -> None:
        """Initialize MineruParser"""
        pass

    @staticmethod
    def safe_write(
        writer: Any,
        content: Union[str, bytes, Dict[str, Any], List[Any]],
        filename: str,
    ) -> None:
        """
        Safely write content to a file, ensuring the filename is valid

        Args:
            writer: The writer object to use
            content: The content to write
            filename: The filename to write to
        """
        # Ensure the filename isn't too long
        if len(filename) > 200:  # Most filesystems have limits around 255 characters
            # Truncate the filename while keeping the extension
            base, ext = os.path.splitext(filename)
            filename = base[:190] + ext  # Leave room for the extension and some margin

        # Handle specific content types
        if isinstance(content, str):
            # Ensure str content is encoded to bytes if required
            try:
                writer.write(content, filename)
            except TypeError:
                # If the writer expects bytes, convert string to bytes
                writer.write(content.encode("utf-8"), filename)
        else:
            # For dict/list content, always encode as JSON string first
            if isinstance(content, (dict, list)):
                try:
                    writer.write(
                        json.dumps(content, ensure_ascii=False, indent=4), filename
                    )
                except TypeError:
                    # If the writer expects bytes, convert JSON string to bytes
                    writer.write(
                        json.dumps(content, ensure_ascii=False, indent=4).encode(
                            "utf-8"
                        ),
                        filename,
                    )
            else:
                # Regular content (assumed to be bytes or compatible)
                writer.write(content, filename)

    @staticmethod
    def parse_pdf(
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        use_ocr: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse PDF document

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            use_ocr: Whether to force OCR parsing

        Returns:
            Tuple[List[Dict[str, Any]], str]: Tuple containing (content list JSON, Markdown text)
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            name_without_suff = pdf_path.stem

            # Prepare output directories - ensure file name is in path
            if output_dir:
                base_output_dir = Path(output_dir)
                local_md_dir = base_output_dir / name_without_suff
            else:
                local_md_dir = pdf_path.parent / name_without_suff

            local_image_dir = local_md_dir / "images"
            image_dir = local_image_dir.name

            # Create directories
            os.makedirs(local_image_dir, exist_ok=True)
            os.makedirs(local_md_dir, exist_ok=True)

            # Initialize writers and reader
            image_writer = FileBasedDataWriter(str(local_image_dir))  # type: ignore
            md_writer = FileBasedDataWriter(str(local_md_dir))  # type: ignore
            reader = FileBasedDataReader("")  # type: ignore

            # Read PDF bytes
            pdf_bytes = reader.read(str(pdf_path))  # type: ignore

            # Create dataset instance
            ds = PymuDocDataset(pdf_bytes)  # type: ignore

            # Process based on PDF type and user preference
            if use_ocr or ds.classify() == SupportedPdfParseMethod.OCR:  # type: ignore
                infer_result = ds.apply(doc_analyze, ocr=True)  # type: ignore
                pipe_result = infer_result.pipe_ocr_mode(image_writer)  # type: ignore
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)  # type: ignore
                pipe_result = infer_result.pipe_txt_mode(image_writer)  # type: ignore

            # Draw visualizations
            try:
                infer_result.draw_model(
                    os.path.join(local_md_dir, f"{name_without_suff}_model.pdf")
                )  # type: ignore
                pipe_result.draw_layout(
                    os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf")
                )  # type: ignore
                pipe_result.draw_span(
                    os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf")
                )  # type: ignore
            except Exception as e:
                print(f"Warning: Failed to draw visualizations: {str(e)}")

            # Get data using API methods
            md_content = pipe_result.get_markdown(image_dir)  # type: ignore
            content_list = pipe_result.get_content_list(image_dir)  # type: ignore

            # Save files using dump methods (consistent with API)
            pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)  # type: ignore
            pipe_result.dump_content_list(
                md_writer, f"{name_without_suff}_content_list.json", image_dir
            )  # type: ignore
            pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")  # type: ignore

            # Save model result - convert JSON string to bytes before writing
            model_inference_result = infer_result.get_infer_res()  # type: ignore
            json_str = json.dumps(model_inference_result, ensure_ascii=False, indent=4)

            try:
                # Try to write to a file manually to avoid FileBasedDataWriter issues
                model_file_path = os.path.join(
                    local_md_dir, f"{name_without_suff}_model.json"
                )
                with open(model_file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            except Exception as e:
                print(
                    f"Warning: Failed to save model result using file write: {str(e)}"
                )
                try:
                    # If direct file write fails, try using the writer with bytes encoding
                    md_writer.write(
                        json_str.encode("utf-8"), f"{name_without_suff}_model.json"
                    )  # type: ignore
                except Exception as e2:
                    print(
                        f"Warning: Failed to save model result using writer: {str(e2)}"
                    )

            return cast(Tuple[List[Dict[str, Any]], str], (content_list, md_content))

        except Exception as e:
            print(f"Error in parse_pdf: {str(e)}")
            raise

    @staticmethod
    def parse_office_doc(
        doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse office document (Word, PPT, etc.)

        Args:
            doc_path: Path to the document file
            output_dir: Output directory path

        Returns:
            Tuple[List[Dict[str, Any]], str]: Tuple containing (content list JSON, Markdown text)
        """
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            name_without_suff = doc_path.stem

            # Prepare output directories - ensure file name is in path
            if output_dir:
                base_output_dir = Path(output_dir)
                local_md_dir = base_output_dir / name_without_suff
            else:
                local_md_dir = doc_path.parent / name_without_suff

            local_image_dir = local_md_dir / "images"
            image_dir = local_image_dir.name

            # Create directories
            os.makedirs(local_image_dir, exist_ok=True)
            os.makedirs(local_md_dir, exist_ok=True)

            # Initialize writers
            image_writer = FileBasedDataWriter(str(local_image_dir))  # type: ignore
            md_writer = FileBasedDataWriter(str(local_md_dir))  # type: ignore

            # Read office document
            ds = read_local_office(str(doc_path))[0]  # type: ignore

            # Apply chain of operations according to API documentation
            # This follows the pattern shown in MS-Office example in the API docs
            ds.apply(doc_analyze, ocr=True).pipe_txt_mode(image_writer).dump_md(
                md_writer, f"{name_without_suff}.md", image_dir
            )  # type: ignore

            # Re-execute for getting the content data
            infer_result = ds.apply(doc_analyze, ocr=True)  # type: ignore
            pipe_result = infer_result.pipe_txt_mode(image_writer)  # type: ignore

            # Get data for return values and additional outputs
            md_content = pipe_result.get_markdown(image_dir)  # type: ignore
            content_list = pipe_result.get_content_list(image_dir)  # type: ignore

            # Save additional output files
            pipe_result.dump_content_list(
                md_writer, f"{name_without_suff}_content_list.json", image_dir
            )  # type: ignore
            pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")  # type: ignore

            # Save model result - convert JSON string to bytes before writing
            model_inference_result = infer_result.get_infer_res()  # type: ignore
            json_str = json.dumps(model_inference_result, ensure_ascii=False, indent=4)

            try:
                # Try to write to a file manually to avoid FileBasedDataWriter issues
                model_file_path = os.path.join(
                    local_md_dir, f"{name_without_suff}_model.json"
                )
                with open(model_file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            except Exception as e:
                print(
                    f"Warning: Failed to save model result using file write: {str(e)}"
                )
                try:
                    # If direct file write fails, try using the writer with bytes encoding
                    md_writer.write(
                        json_str.encode("utf-8"), f"{name_without_suff}_model.json"
                    )  # type: ignore
                except Exception as e2:
                    print(
                        f"Warning: Failed to save model result using writer: {str(e2)}"
                    )

            return cast(Tuple[List[Dict[str, Any]], str], (content_list, md_content))

        except Exception as e:
            print(f"Error in parse_office_doc: {str(e)}")
            raise

    @staticmethod
    def parse_image(
        image_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse image document

        Args:
            image_path: Path to the image file
            output_dir: Output directory path

        Returns:
            Tuple[List[Dict[str, Any]], str]: Tuple containing (content list JSON, Markdown text)
        """
        try:
            # Convert to Path object for easier handling
            image_path = Path(image_path)
            name_without_suff = image_path.stem

            # Prepare output directories - ensure file name is in path
            if output_dir:
                base_output_dir = Path(output_dir)
                local_md_dir = base_output_dir / name_without_suff
            else:
                local_md_dir = image_path.parent / name_without_suff

            local_image_dir = local_md_dir / "images"
            image_dir = local_image_dir.name

            # Create directories
            os.makedirs(local_image_dir, exist_ok=True)
            os.makedirs(local_md_dir, exist_ok=True)

            # Initialize writers
            image_writer = FileBasedDataWriter(str(local_image_dir))  # type: ignore
            md_writer = FileBasedDataWriter(str(local_md_dir))  # type: ignore

            # Read image
            ds = read_local_images(str(image_path))[0]  # type: ignore

            # Apply chain of operations according to API documentation
            # This follows the pattern shown in Image example in the API docs
            ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                md_writer, f"{name_without_suff}.md", image_dir
            )  # type: ignore

            # Re-execute for getting the content data
            infer_result = ds.apply(doc_analyze, ocr=True)  # type: ignore
            pipe_result = infer_result.pipe_ocr_mode(image_writer)  # type: ignore

            # Get data for return values and additional outputs
            md_content = pipe_result.get_markdown(image_dir)  # type: ignore
            content_list = pipe_result.get_content_list(image_dir)  # type: ignore

            # Save additional output files
            pipe_result.dump_content_list(
                md_writer, f"{name_without_suff}_content_list.json", image_dir
            )  # type: ignore
            pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")  # type: ignore

            # Save model result - convert JSON string to bytes before writing
            model_inference_result = infer_result.get_infer_res()  # type: ignore
            json_str = json.dumps(model_inference_result, ensure_ascii=False, indent=4)

            try:
                # Try to write to a file manually to avoid FileBasedDataWriter issues
                model_file_path = os.path.join(
                    local_md_dir, f"{name_without_suff}_model.json"
                )
                with open(model_file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
            except Exception as e:
                print(
                    f"Warning: Failed to save model result using file write: {str(e)}"
                )
                try:
                    # If direct file write fails, try using the writer with bytes encoding
                    md_writer.write(
                        json_str.encode("utf-8"), f"{name_without_suff}_model.json"
                    )  # type: ignore
                except Exception as e2:
                    print(
                        f"Warning: Failed to save model result using writer: {str(e2)}"
                    )

            return cast(Tuple[List[Dict[str, Any]], str], (content_list, md_content))

        except Exception as e:
            print(f"Error in parse_image: {str(e)}")
            raise

    @staticmethod
    def parse_document(
        file_path: Union[str, Path],
        parse_method: str = "auto",
        output_dir: Optional[str] = None,
        save_results: bool = True,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Parse document using MinerU based on file extension

        Args:
            file_path: Path to the file to be parsed
            parse_method: Parsing method, supports "auto", "ocr", "txt", default is "auto"
            output_dir: Output directory path, if None, use the directory of the input file
            save_results: Whether to save parsing results to files

        Returns:
            Tuple[List[Dict[str, Any]], str]: Tuple containing (content list JSON, Markdown text)
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext in [".pdf"]:
            return MineruParser.parse_pdf(
                file_path, output_dir, use_ocr=(parse_method == "ocr")
            )
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
            return MineruParser.parse_image(file_path, output_dir)
        elif ext in [".doc", ".docx", ".ppt", ".pptx"]:
            return MineruParser.parse_office_doc(file_path, output_dir)
        else:
            # For unsupported file types, default to PDF parsing
            print(
                f"Warning: Unsupported file extension '{ext}', trying generic PDF parser"
            )
            return MineruParser.parse_pdf(
                file_path, output_dir, use_ocr=(parse_method == "ocr")
            )


def main():
    """
    Main function to run the MinerU parser from command line
    """
    parser = argparse.ArgumentParser(description="Parse documents using MinerU")
    parser.add_argument("file_path", help="Path to the document to parse")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "ocr", "txt"],
        default="auto",
        help="Parsing method (auto, ocr, txt)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display content statistics"
    )

    args = parser.parse_args()

    try:
        # Parse the document
        content_list, md_content = MineruParser.parse_document(
            file_path=args.file_path, parse_method=args.method, output_dir=args.output
        )

        # Display statistics if requested
        if args.stats:
            print("\nDocument Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # Count different types of content
            content_types = {}
            for item in content_list:
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

            print("\nContent Type Distribution:")
            for content_type, count in content_types.items():
                print(f"- {content_type}: {count}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
