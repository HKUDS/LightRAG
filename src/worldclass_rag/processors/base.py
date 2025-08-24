"""
Base classes for document processors.

Siguiendo las mejores prácticas del video AI News & Strategy Daily:
- Limpieza de boilerplate (encabezados y pies de página)
- Normalización del espacio en blanco
- Extracción de metadatos enriquecidos
- Manejo robusto de errores
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import mimetypes

@dataclass
class ProcessedDocument:
    """
    Representa un documento procesado con contenido limpio y metadatos.
    
    Incluye toda la información necesaria para chunking e indexación efectivos.
    """
    content: str
    metadata: Dict[str, Any]
    
    # Información del documento original
    source_path: Optional[str] = None
    source_type: Optional[str] = None
    file_size: Optional[int] = None
    
    # Procesamiento
    processed_at: datetime = None
    processor_used: str = ""
    processing_time_ms: Optional[float] = None
    
    # Calidad del procesamiento
    extraction_quality: float = 1.0
    warnings: List[str] = None
    
    # Contenido estructurado
    sections: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización automática de campos."""
        if self.processed_at is None:
            self.processed_at = datetime.now()
            
        if self.warnings is None:
            self.warnings = []
            
        if self.sections is None:
            self.sections = []
            
        if self.tables is None:
            self.tables = []
            
        if self.images is None:
            self.images = []
            
        # Calcular hash del contenido para deduplicación
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        self.metadata["content_hash"] = content_hash
        self.metadata["content_length"] = len(self.content)
        
    def add_warning(self, warning: str) -> None:
        """Añade una advertencia al procesamiento."""
        self.warnings.append(warning)
        
    def add_section(self, title: str, content: str, level: int = 1, **kwargs) -> None:
        """Añade una sección estructurada."""
        section = {
            "title": title,
            "content": content,
            "level": level,
            "start_position": kwargs.get("start_position"),
            "end_position": kwargs.get("end_position"),
            **kwargs
        }
        self.sections.append(section)
    
    def add_table(self, data: List[List[str]], caption: str = "", **kwargs) -> None:
        """Añade una tabla extraída."""
        table = {
            "data": data,
            "caption": caption,
            "rows": len(data),
            "columns": len(data[0]) if data else 0,
            **kwargs
        }
        self.tables.append(table)
    
    def add_image(self, path: str, caption: str = "", **kwargs) -> None:
        """Añade información de imagen."""
        image = {
            "path": path,
            "caption": caption,
            **kwargs
        }
        self.images.append(image)


class DocumentProcessor(ABC):
    """
    Clase base para todos los procesadores de documentos.
    
    Implementa el patrón Template Method para procesamiento consistente.
    """
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
    ):
        """
        Inicializa el procesador base.
        
        Args:
            remove_boilerplate: Si remover encabezados/pies repetitivos
            normalize_whitespace: Si normalizar espacios y saltos de línea
            extract_metadata: Si extraer metadatos del documento
            preserve_structure: Si preservar información estructural
        """
        self.remove_boilerplate = remove_boilerplate
        self.normalize_whitespace = normalize_whitespace  
        self.extract_metadata = extract_metadata
        self.preserve_structure = preserve_structure
        
    def process(
        self, 
        source: Union[str, Path, bytes], 
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Procesa un documento completo siguiendo el pipeline estándar.
        
        Template method que orchestrates el proceso completo:
        1. Validación y preparación
        2. Extracción de contenido específico del tipo
        3. Limpieza y normalización
        4. Extracción de metadatos
        5. Validación de calidad
        """
        start_time = datetime.now()
        
        try:
            # 1. Validación de entrada
            self._validate_input(source)
            
            # 2. Extracción de contenido (implementado por subclases)
            raw_content, extraction_metadata = self._extract_content(source)
            
            # 3. Limpieza del contenido
            cleaned_content = self._clean_content(raw_content)
            
            # 4. Extracción de metadatos
            metadata = self._extract_metadata(source, source_metadata or {})
            metadata.update(extraction_metadata)
            
            # 5. Crear documento procesado
            processed_doc = ProcessedDocument(
                content=cleaned_content,
                metadata=metadata,
                source_path=str(source) if isinstance(source, (str, Path)) else None,
                source_type=self._get_source_type(source),
                processor_used=self.__class__.__name__,
            )
            
            # 6. Validación de calidad
            self._validate_quality(processed_doc)
            
            # 7. Calcular tiempo de procesamiento
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            processed_doc.processing_time_ms = processing_time
            
            return processed_doc
            
        except Exception as e:
            # Manejo robusto de errores
            error_doc = ProcessedDocument(
                content="",
                metadata={"error": str(e), "processor": self.__class__.__name__},
                source_path=str(source) if isinstance(source, (str, Path)) else None,
                extraction_quality=0.0,
                processor_used=self.__class__.__name__,
            )
            error_doc.add_warning(f"Error procesando documento: {str(e)}")
            return error_doc
    
    @abstractmethod
    def _extract_content(self, source: Union[str, Path, bytes]) -> tuple[str, Dict[str, Any]]:
        """
        Extrae el contenido específico del tipo de documento.
        
        Must be implemented by subclasses.
        
        Returns:
            Tuple of (extracted_text, extraction_metadata)
        """
        pass
    
    def _validate_input(self, source: Union[str, Path, bytes]) -> None:
        """Valida la entrada del procesador."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {path}")
            if not path.is_file():
                raise ValueError(f"La ruta no es un archivo: {path}")
        elif isinstance(source, bytes):
            if len(source) == 0:
                raise ValueError("Los datos bytes están vacíos")
        else:
            raise TypeError(f"Tipo de fuente no soportado: {type(source)}")
    
    def _clean_content(self, raw_content: str) -> str:
        """
        Limpia el contenido siguiendo las mejores prácticas.
        
        Implementa limpieza estándar que puede ser extendida por subclases.
        """
        if not raw_content:
            return ""
            
        content = raw_content
        
        # Remover boilerplate si está habilitado
        if self.remove_boilerplate:
            content = self._remove_boilerplate(content)
        
        # Normalizar espacios en blanco
        if self.normalize_whitespace:
            content = self._normalize_whitespace(content)
        
        return content
    
    def _remove_boilerplate(self, content: str) -> str:
        """
        Remueve contenido boilerplate común.
        
        Implementación base que puede ser sobrescrita por procesadores específicos.
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        # Detectar y remover encabezados/pies repetitivos
        line_frequency = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_frequency[stripped] = line_frequency.get(stripped, 0) + 1
        
        # Considerar como boilerplate líneas que aparecen más de 3 veces
        # y son cortas (probables encabezados/pies)
        boilerplate_lines = {
            line for line, freq in line_frequency.items() 
            if freq > 3 and len(line) < 100
        }
        
        # Filtrar líneas boilerplate
        for line in lines:
            if line.strip() not in boilerplate_lines:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_whitespace(self, content: str) -> str:
        """
        Normaliza espacios en blanco siguiendo mejores prácticas.
        
        - Convierte múltiples espacios en uno solo
        - Normaliza saltos de línea
        - Remueve espacios al inicio/final de líneas
        """
        import re
        
        # Normalizar saltos de línea (convertir \r\n y \r a \n)
        content = re.sub(r'\r\n|\r', '\n', content)
        
        # Remover espacios al final de líneas
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # Convertir múltiples espacios consecutivos en uno solo
        content = re.sub(r' {2,}', ' ', content)
        
        # Limitar saltos de línea consecutivos a máximo 2
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remover espacios al principio y final del documento
        content = content.strip()
        
        return content
    
    def _extract_metadata(
        self, 
        source: Union[str, Path, bytes], 
        source_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extrae metadatos básicos del documento.
        
        Puede ser extendido por procesadores específicos.
        """
        metadata = source_metadata.copy()
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            
            # Información del archivo
            metadata.update({
                "filename": path.name,
                "file_extension": path.suffix.lower(),
                "file_size": path.stat().st_size,
                "mime_type": mimetypes.guess_type(str(path))[0],
                "created_at": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            })
        
        # Metadatos de procesamiento
        metadata.update({
            "processed_by": self.__class__.__name__,
            "processed_at": datetime.now().isoformat(),
            "remove_boilerplate": self.remove_boilerplate,
            "normalize_whitespace": self.normalize_whitespace,
        })
        
        return metadata
    
    def _get_source_type(self, source: Union[str, Path, bytes]) -> str:
        """Determina el tipo de fuente del documento."""
        if isinstance(source, bytes):
            return "bytes"
        
        path = Path(source)
        extension = path.suffix.lower()
        
        type_mapping = {
            '.txt': 'text',
            '.md': 'markdown', 
            '.pdf': 'pdf',
            '.docx': 'word',
            '.doc': 'word',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.pptx': 'powerpoint',
            '.ppt': 'powerpoint',
            '.html': 'html',
            '.htm': 'html',
            '.xml': 'xml',
            '.json': 'json',
            '.csv': 'csv',
        }
        
        return type_mapping.get(extension, 'unknown')
    
    def _validate_quality(self, processed_doc: ProcessedDocument) -> None:
        """
        Valida la calidad del documento procesado.
        
        Añade advertencias y ajusta quality score según problemas detectados.
        """
        quality_score = 1.0
        
        # Verificar longitud mínima
        if len(processed_doc.content) < 50:
            processed_doc.add_warning("Contenido muy corto después del procesamiento")
            quality_score *= 0.5
        
        # Verificar ratio de caracteres válidos
        text_chars = sum(1 for c in processed_doc.content if c.isprintable() or c.isspace())
        if len(processed_doc.content) > 0:
            valid_ratio = text_chars / len(processed_doc.content)
            if valid_ratio < 0.8:
                processed_doc.add_warning("Alto porcentaje de caracteres no válidos")
                quality_score *= 0.7
        
        # Verificar estructura básica (presencia de oraciones)
        sentences = processed_doc.content.count('.') + processed_doc.content.count('!') + processed_doc.content.count('?')
        if sentences == 0 and len(processed_doc.content) > 100:
            processed_doc.add_warning("No se detectaron oraciones en el contenido")
            quality_score *= 0.6
        
        processed_doc.extraction_quality = quality_score
    
    def supports_type(self, file_path: Union[str, Path]) -> bool:
        """
        Verifica si este procesador soporta el tipo de archivo.
        
        Debe ser implementado por subclases.
        """
        return False
    
    def get_supported_extensions(self) -> List[str]:
        """
        Retorna lista de extensiones soportadas por este procesador.
        
        Debe ser implementado por subclases.
        """
        return []