"""
PDF Document Processor - Procesador especializado para documentos PDF.

Implementa las mejores prácticas del video AI News & Strategy Daily:
- Limpieza de "terrible header and footer pollution" común en PDFs
- Manejo especial de tablas con relaciones espaciales
- OCR para documentos escaneados
- Extracción de metadatos enriquecidos
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import tempfile
import subprocess

# PDF processing libraries
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from ..base import DocumentProcessor, ProcessedDocument


class PDFProcessor(DocumentProcessor):
    """
    Procesador especializado para documentos PDF con capacidades avanzadas.
    
    Características destacadas:
    - Extracción de texto nativo y OCR cuando sea necesario
    - Detección y limpieza automática de headers/footers repetitivos
    - Preservación de estructura de tablas y layouts
    - Extracción de metadatos PDF
    - Manejo robusto de PDFs corruptos o problemáticos
    
    Siguiendo las mejores prácticas del video AI News & Strategy Daily:
    - Los PDFs a menudo tienen "terrible header and footer pollution"
    - Las tablas requieren manejo especial para codificar relaciones espaciales
    - El OCR debe ser correcto para documentos escaneados
    """
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        use_ocr: bool = True,
        ocr_language: str = "spa+eng",  # Español + Inglés por defecto
        detect_tables: bool = True,
        remove_headers_footers: bool = True,
        min_text_confidence: float = 0.5,
        prefer_pymupdf: bool = True,
    ):
        """
        Inicializa el procesador PDF.
        
        Args:
            use_ocr: Si usar OCR para páginas sin texto extraíble
            ocr_language: Idiomas para OCR (formato tesseract)
            detect_tables: Si detectar y extraer tablas
            remove_headers_footers: Si remover headers/footers automáticamente
            min_text_confidence: Confianza mínima para texto OCR
            prefer_pymupdf: Si preferir PyMuPDF sobre PyPDF2
        """
        super().__init__(
            remove_boilerplate=remove_boilerplate,
            normalize_whitespace=normalize_whitespace,
            extract_metadata=extract_metadata,
            preserve_structure=preserve_structure,
        )
        
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        self.detect_tables = detect_tables
        self.remove_headers_footers = remove_headers_footers
        self.min_text_confidence = min_text_confidence
        self.prefer_pymupdf = prefer_pymupdf and PYMUPDF_AVAILABLE
        
        # Verificar disponibilidad de bibliotecas
        if not PYPDF_AVAILABLE and not PYMUPDF_AVAILABLE:
            raise ImportError("Se requiere pypdf o PyMuPDF para procesar PDFs")
        
        if self.use_ocr and not OCR_AVAILABLE:
            print("Warning: OCR no disponible, deshabilitando función OCR")
            self.use_ocr = False
    
    def _extract_content(self, source: Union[str, Path, bytes]) -> Tuple[str, Dict[str, Any]]:
        """
        Extrae contenido de PDF usando la mejor estrategia disponible.
        
        Prioriza texto nativo, usa OCR como fallback para páginas escaneadas.
        """
        extraction_metadata = {
            "extraction_method": [],
            "pages_processed": 0,
            "pages_with_text": 0,
            "pages_with_ocr": 0,
            "tables_found": 0,
            "text_extraction_confidence": 1.0,
        }
        
        try:
            if self.prefer_pymupdf and PYMUPDF_AVAILABLE:
                content, metadata = self._extract_with_pymupdf(source)
            else:
                content, metadata = self._extract_with_pypdf(source)
            
            extraction_metadata.update(metadata)
            
            # Post-procesamiento específico para PDFs
            if self.remove_headers_footers:
                content = self._remove_pdf_boilerplate(content, extraction_metadata)
            
            return content, extraction_metadata
            
        except Exception as e:
            # Fallback al otro método si hay error
            try:
                if self.prefer_pymupdf:
                    content, metadata = self._extract_with_pypdf(source)
                else:
                    content, metadata = self._extract_with_pymupdf(source)
                
                extraction_metadata.update(metadata)
                extraction_metadata["extraction_method"].append("fallback")
                return content, extraction_metadata
                
            except Exception as e2:
                raise Exception(f"Error extrayendo PDF con ambos métodos: {e}, {e2}")
    
    def _extract_with_pymupdf(self, source: Union[str, Path, bytes]) -> Tuple[str, Dict[str, Any]]:
        """Extrae contenido usando PyMuPDF (más robusto y rápido)."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF no disponible")
        
        metadata = {"extraction_method": ["pymupdf"]}
        
        # Abrir documento
        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(str(source))
        
        try:
            pages_content = []
            pages_metadata = []
            total_pages = len(doc)
            
            metadata.update({
                "total_pages": total_pages,
                "pages_processed": 0,
                "pages_with_text": 0,
                "pages_with_ocr": 0,
            })
            
            for page_num in range(total_pages):
                page = doc[page_num]
                page_content = ""
                page_meta = {"page_number": page_num + 1}
                
                # Intentar extraer texto nativo
                native_text = page.get_text()
                
                if native_text and len(native_text.strip()) > 50:
                    # Hay texto nativo suficiente
                    page_content = native_text
                    page_meta["extraction_method"] = "native"
                    metadata["pages_with_text"] += 1
                    
                elif self.use_ocr:
                    # Usar OCR como fallback
                    ocr_text, ocr_confidence = self._extract_page_with_ocr_pymupdf(page)
                    
                    if ocr_confidence >= self.min_text_confidence:
                        page_content = ocr_text
                        page_meta["extraction_method"] = "ocr"
                        page_meta["ocr_confidence"] = ocr_confidence
                        metadata["pages_with_ocr"] += 1
                    else:
                        page_meta["extraction_method"] = "failed"
                        page_meta["reason"] = "low_ocr_confidence"
                
                # Detectar tablas si está habilitado
                if self.detect_tables:
                    tables = self._detect_tables_pymupdf(page)
                    if tables:
                        page_meta["tables"] = tables
                        metadata["tables_found"] += len(tables)
                
                if page_content.strip():
                    pages_content.append(page_content)
                    pages_metadata.append(page_meta)
                
                metadata["pages_processed"] += 1
            
            # Extraer metadatos del documento
            doc_metadata = doc.metadata
            metadata.update({
                "pdf_metadata": doc_metadata,
                "title": doc_metadata.get("title", ""),
                "author": doc_metadata.get("author", ""),
                "subject": doc_metadata.get("subject", ""),
                "creator": doc_metadata.get("creator", ""),
                "producer": doc_metadata.get("producer", ""),
                "creation_date": doc_metadata.get("creationDate", ""),
                "modification_date": doc_metadata.get("modDate", ""),
            })
            
            # Combinar contenido de todas las páginas
            full_content = "\n\n".join(pages_content)
            metadata["pages_metadata"] = pages_metadata
            
            return full_content, metadata
            
        finally:
            doc.close()
    
    def _extract_with_pypdf(self, source: Union[str, Path, bytes]) -> Tuple[str, Dict[str, Any]]:
        """Extrae contenido usando PyPDF como fallback."""
        if not PYPDF_AVAILABLE:
            raise ImportError("PyPDF no disponible")
        
        metadata = {"extraction_method": ["pypdf"]}
        
        if isinstance(source, bytes):
            import io
            pdf_file = io.BytesIO(source)
        else:
            pdf_file = open(str(source), 'rb')
        
        try:
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            
            metadata.update({
                "total_pages": total_pages,
                "pages_processed": 0,
                "pages_with_text": 0,
            })
            
            pages_content = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    
                    if page_text and len(page_text.strip()) > 20:
                        pages_content.append(page_text)
                        metadata["pages_with_text"] += 1
                    
                    metadata["pages_processed"] += 1
                    
                except Exception as e:
                    print(f"Warning: Error extrayendo página {page_num + 1}: {e}")
                    continue
            
            # Extraer metadatos del PDF
            try:
                pdf_info = reader.metadata
                if pdf_info:
                    metadata.update({
                        "title": pdf_info.get("/Title", ""),
                        "author": pdf_info.get("/Author", ""),
                        "subject": pdf_info.get("/Subject", ""),
                        "creator": pdf_info.get("/Creator", ""),
                        "producer": pdf_info.get("/Producer", ""),
                        "creation_date": str(pdf_info.get("/CreationDate", "")),
                        "modification_date": str(pdf_info.get("/ModDate", "")),
                    })
            except Exception as e:
                print(f"Warning: Error extrayendo metadatos PDF: {e}")
            
            full_content = "\n\n".join(pages_content)
            return full_content, metadata
            
        finally:
            if not isinstance(source, bytes):
                pdf_file.close()
    
    def _extract_page_with_ocr_pymupdf(self, page) -> Tuple[str, float]:
        """
        Extrae texto de una página usando OCR con PyMuPDF.
        
        Returns:
            Tuple de (texto_extraído, confianza_promedio)
        """
        if not self.use_ocr:
            return "", 0.0
        
        try:
            # Renderizar página como imagen
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom para mejor OCR
            img_data = pix.tobytes("png")
            
            # Convertir a PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Aplicar OCR
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=self.ocr_language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extraer texto y calcular confianza
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    words.append(word)
                    conf = int(ocr_data['conf'][i])
                    if conf > 0:
                        confidences.append(conf)
            
            text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            normalized_confidence = avg_confidence / 100.0
            
            return text, normalized_confidence
            
        except Exception as e:
            print(f"Warning: Error en OCR: {e}")
            return "", 0.0
    
    def _detect_tables_pymupdf(self, page) -> List[Dict[str, Any]]:
        """
        Detecta tablas en una página usando PyMuPDF.
        
        Implementa detección básica de estructuras tabulares.
        """
        if not self.detect_tables:
            return []
        
        tables = []
        
        try:
            # Buscar tablas usando detección de líneas y texto
            tables_found = page.find_tables()
            
            for i, table in enumerate(tables_found):
                try:
                    # Extraer datos de la tabla
                    table_data = table.extract()
                    
                    if table_data and len(table_data) > 1:  # Al menos header + 1 fila
                        table_info = {
                            "table_id": i,
                            "bbox": table.bbox,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "data": table_data,
                            "extraction_method": "pymupdf_native"
                        }
                        tables.append(table_info)
                        
                except Exception as e:
                    print(f"Warning: Error extrayendo tabla {i}: {e}")
                    continue
        
        except Exception as e:
            # Fallback: detección manual básica
            print(f"Warning: Error en detección automática de tablas: {e}")
            tables = self._detect_tables_manual(page)
        
        return tables
    
    def _detect_tables_manual(self, page) -> List[Dict[str, Any]]:
        """
        Detección manual de tablas basada en patrones de texto.
        
        Fallback cuando la detección automática no está disponible.
        """
        tables = []
        
        try:
            text = page.get_text()
            lines = text.split('\n')
            
            # Buscar patrones de tabla (líneas con múltiples columnas separadas)
            potential_tables = []
            current_table_lines = []
            
            for line in lines:
                # Detectar si la línea parece ser parte de una tabla
                # (múltiples "palabras" separadas por espacios grandes)
                if self._looks_like_table_row(line):
                    current_table_lines.append(line)
                else:
                    if len(current_table_lines) >= 3:  # Mínimo 3 filas para considerar tabla
                        potential_tables.append(current_table_lines[:])
                    current_table_lines = []
            
            # Procesar tablas encontradas
            for i, table_lines in enumerate(potential_tables):
                table_data = []
                for line in table_lines:
                    # Dividir por múltiples espacios
                    columns = re.split(r'\s{2,}', line.strip())
                    if columns:
                        table_data.append(columns)
                
                if table_data:
                    table_info = {
                        "table_id": i,
                        "rows": len(table_data),
                        "columns": max(len(row) for row in table_data) if table_data else 0,
                        "data": table_data,
                        "extraction_method": "manual_pattern"
                    }
                    tables.append(table_info)
        
        except Exception as e:
            print(f"Warning: Error en detección manual de tablas: {e}")
        
        return tables
    
    def _looks_like_table_row(self, line: str) -> bool:
        """
        Determina si una línea parece ser parte de una tabla.
        
        Criterios:
        - Múltiples "columnas" separadas por espacios
        - Longitud razonable
        - No parece ser texto corrido
        """
        if not line.strip() or len(line.strip()) < 10:
            return False
        
        # Contar "columnas" (grupos de texto separados por 2+ espacios)
        columns = re.split(r'\s{2,}', line.strip())
        
        # Debe tener al menos 2 columnas
        if len(columns) < 2:
            return False
        
        # Las columnas no deben ser muy largas (no texto corrido)
        avg_column_length = sum(len(col.split()) for col in columns) / len(columns)
        if avg_column_length > 10:  # Promedio de más de 10 palabras por columna
            return False
        
        return True
    
    def _remove_pdf_boilerplate(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Remueve headers y footers repetitivos específicos de PDFs.
        
        Implementa la recomendación del video sobre "terrible header and footer pollution".
        """
        if not self.remove_headers_footers:
            return content
        
        lines = content.split('\n')
        if len(lines) < 10:  # No procesar documentos muy cortos
            return content
        
        # Detectar patrones repetitivos que aparecen en múltiples páginas
        line_frequency = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:  # Solo considerar líneas cortas
                line_frequency[stripped] = line_frequency.get(stripped, 0) + 1
        
        # Identificar líneas que aparecen frecuentemente (posibles headers/footers)
        total_pages = metadata.get("total_pages", 1)
        min_frequency = max(2, total_pages // 3)  # Debe aparecer en al menos 1/3 de las páginas
        
        boilerplate_lines = set()
        for line, freq in line_frequency.items():
            if freq >= min_frequency:
                # Verificar patrones comunes de headers/footers
                if (self._looks_like_header_footer(line) or 
                    freq > total_pages * 0.8):  # Aparece en >80% de páginas
                    boilerplate_lines.add(line)
        
        # Filtrar líneas boilerplate
        cleaned_lines = []
        for line in lines:
            if line.strip() not in boilerplate_lines:
                cleaned_lines.append(line)
        
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Añadir información sobre limpieza a metadatos
        if boilerplate_lines:
            metadata["boilerplate_removed"] = list(boilerplate_lines)
            metadata["boilerplate_lines_count"] = len(boilerplate_lines)
        
        return cleaned_content
    
    def _looks_like_header_footer(self, line: str) -> bool:
        """
        Determina si una línea parece ser un header o footer.
        
        Patrones comunes:
        - Números de página
        - Fechas
        - Nombres de documentos/capítulos cortos
        - URLs o información de copyright
        """
        line_lower = line.lower().strip()
        
        # Patrones de headers/footers
        patterns = [
            r'^\d+$',                           # Solo número de página
            r'^page\s*\d+',                     # "Page N"
            r'^\d+\s*/\s*\d+$',                # "1 / 10"
            r'\d{1,2}/\d{1,2}/\d{2,4}',        # Fechas
            r'\d{4}-\d{2}-\d{2}',              # Fechas ISO
            r'^\s*chapter\s+\d+',               # "Chapter N"  
            r'^\s*capítulo\s+\d+',             # "Capítulo N"
            r'^©\s*\d{4}',                      # Copyright
            r'www\.|\.com|\.org|\.net',         # URLs
            r'confidential|internal|draft',     # Marcas de estado
        ]
        
        for pattern in patterns:
            if re.search(pattern, line_lower):
                return True
        
        # También considerar líneas muy cortas con pocas palabras
        words = line.split()
        if len(words) <= 3 and len(line) < 50:
            return True
        
        return False
    
    def supports_type(self, file_path: Union[str, Path]) -> bool:
        """Verifica si el archivo es un PDF."""
        path = Path(file_path)
        return path.suffix.lower() == '.pdf'
    
    def get_supported_extensions(self) -> List[str]:
        """Retorna extensiones PDF soportadas."""
        return ['.pdf']