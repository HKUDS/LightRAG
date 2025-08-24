"""
Text Document Processor - Procesador para documentos de texto plano.

Maneja archivos de texto, markdown, y otros formatos basados en texto,
implementando las mejores prácticas del video AI News & Strategy Daily.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chardet

from ..base import DocumentProcessor


class TextProcessor(DocumentProcessor):
    """
    Procesador especializado para documentos de texto plano.
    
    Características:
    - Detección automática de encoding
    - Preservación de estructura markdown/rst
    - Limpieza inteligente de formateo
    - Extracción de metadatos textuales
    - Manejo robusto de diferentes encodings
    """
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        detect_encoding: bool = True,
        preserve_markdown: bool = True,
        extract_links: bool = True,
        extract_headers: bool = True,
    ):
        """
        Inicializa el procesador de texto.
        
        Args:
            detect_encoding: Si detectar automáticamente el encoding
            preserve_markdown: Si preservar estructura markdown
            extract_links: Si extraer y catalogar enlaces
            extract_headers: Si extraer jerarquía de encabezados
        """
        super().__init__(
            remove_boilerplate=remove_boilerplate,
            normalize_whitespace=normalize_whitespace,
            extract_metadata=extract_metadata,
            preserve_structure=preserve_structure,
        )
        
        self.detect_encoding = detect_encoding
        self.preserve_markdown = preserve_markdown
        self.extract_links = extract_links
        self.extract_headers = extract_headers
        
        # Extensiones soportadas
        self.supported_extensions = {
            '.txt', '.text', '.md', '.markdown', '.rst', '.asciidoc', 
            '.org', '.wiki', '.log', '.cfg', '.conf', '.ini', '.yaml', 
            '.yml', '.json', '.xml', '.html', '.htm', '.css', '.js',
            '.py', '.java', '.cpp', '.c', '.h', '.sql', '.sh', '.bash'
        }
    
    def _extract_content(self, source: Union[str, Path, bytes]) -> tuple[str, Dict[str, Any]]:
        """
        Extrae contenido de archivos de texto con detección de encoding.
        
        Maneja robustamente diferentes encodings y formatos.
        """
        extraction_metadata = {}
        
        if isinstance(source, bytes):
            # Detectar encoding de datos bytes
            encoding_info = chardet.detect(source) if self.detect_encoding else {'encoding': 'utf-8'}
            encoding = encoding_info.get('encoding', 'utf-8')
            confidence = encoding_info.get('confidence', 1.0)
            
            try:
                content = source.decode(encoding)
                extraction_metadata.update({
                    "encoding": encoding,
                    "encoding_confidence": confidence,
                    "source_type": "bytes"
                })
            except UnicodeDecodeError:
                # Fallback a utf-8 con errores ignorados
                content = source.decode('utf-8', errors='ignore')
                extraction_metadata.update({
                    "encoding": "utf-8",
                    "encoding_confidence": 0.5,
                    "encoding_fallback": True,
                    "source_type": "bytes"
                })
        
        else:
            # Leer archivo con detección de encoding
            path = Path(source)
            
            if self.detect_encoding:
                # Leer muestra para detectar encoding
                with open(path, 'rb') as f:
                    raw_data = f.read(10000)  # Leer primeros 10KB
                
                encoding_info = chardet.detect(raw_data)
                encoding = encoding_info.get('encoding', 'utf-8')
                confidence = encoding_info.get('confidence', 1.0)
            else:
                encoding = 'utf-8'
                confidence = 1.0
            
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                extraction_metadata.update({
                    "encoding": encoding,
                    "encoding_confidence": confidence,
                    "source_type": "file",
                    "file_size": path.stat().st_size
                })
                
            except UnicodeDecodeError:
                # Fallback con manejo de errores
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                extraction_metadata.update({
                    "encoding": "utf-8",
                    "encoding_confidence": 0.5,
                    "encoding_fallback": True,
                    "source_type": "file"
                })
        
        # Análisis básico del contenido
        extraction_metadata.update(self._analyze_text_content(content))
        
        return content, extraction_metadata
    
    def _analyze_text_content(self, content: str) -> Dict[str, Any]:
        """
        Analiza el contenido textual para extraer metadatos útiles.
        """
        analysis = {
            "content_type": "text",
            "character_count": len(content),
            "line_count": content.count('\n') + 1,
            "word_count": len(content.split()),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
        }
        
        # Detectar tipo de formato
        format_type = self._detect_text_format(content)
        analysis["format_type"] = format_type
        
        # Análisis específico según formato
        if format_type == "markdown":
            analysis.update(self._analyze_markdown(content))
        elif format_type == "restructuredtext":
            analysis.update(self._analyze_rst(content))
        elif format_type == "html":
            analysis.update(self._analyze_html(content))
        elif format_type == "code":
            analysis.update(self._analyze_code(content))
        
        return analysis
    
    def _detect_text_format(self, content: str) -> str:
        """
        Detecta el formato del texto basándose en patrones.
        """
        # Contar indicadores de diferentes formatos
        indicators = {
            "markdown": 0,
            "restructuredtext": 0,
            "html": 0,
            "json": 0,
            "xml": 0,
            "code": 0,
            "log": 0,
        }
        
        # Markdown indicators
        md_patterns = [
            r'^#{1,6}\s+',      # Headers
            r'^\*{1,2}[^*]+\*{1,2}',  # Bold/italic
            r'^\- |\* |\+ ',    # Lists
            r'^\d+\. ',         # Numbered lists
            r'```',             # Code blocks
            r'\[.*?\]\(.*?\)',  # Links
        ]
        
        for pattern in md_patterns:
            indicators["markdown"] += len(re.findall(pattern, content, re.MULTILINE))
        
        # HTML indicators
        if '<' in content and '>' in content:
            html_tags = len(re.findall(r'<[^>]+>', content))
            indicators["html"] = html_tags
        
        # JSON indicators
        if content.strip().startswith('{') and content.strip().endswith('}'):
            indicators["json"] = content.count('"') + content.count(':')
        
        # XML indicators
        if re.match(r'^\s*<\?xml', content):
            indicators["xml"] = len(re.findall(r'<[^>]+>', content))
        
        # Code indicators (multiple patterns)
        code_patterns = [
            r'def\s+\w+\s*\(',      # Python functions
            r'function\s+\w+\s*\(', # JavaScript functions
            r'class\s+\w+\s*[:{]',  # Classes
            r'import\s+\w+',        # Imports
            r'#include\s*<',        # C includes
        ]
        
        for pattern in code_patterns:
            indicators["code"] += len(re.findall(pattern, content))
        
        # Log indicators
        log_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # Dates
            r'\d{2}:\d{2}:\d{2}',           # Times
            r'\b(ERROR|WARN|INFO|DEBUG)\b', # Log levels
        ]
        
        for pattern in log_patterns:
            indicators["log"] += len(re.findall(pattern, content))
        
        # Determinar formato más probable
        max_score = max(indicators.values())
        if max_score == 0:
            return "plain_text"
        
        return max(indicators, key=indicators.get)
    
    def _analyze_markdown(self, content: str) -> Dict[str, Any]:
        """Analiza contenido Markdown específicamente."""
        analysis = {
            "markdown_analysis": True,
            "headers": [],
            "links": [],
            "images": [],
            "code_blocks": [],
        }
        
        # Extraer encabezados
        if self.extract_headers:
            header_pattern = r'^(#{1,6})\s+(.+)$'
            for match in re.finditer(header_pattern, content, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2).strip()
                analysis["headers"].append({
                    "level": level,
                    "title": title,
                    "position": match.start()
                })
        
        # Extraer enlaces
        if self.extract_links:
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            for match in re.finditer(link_pattern, content):
                text = match.group(1)
                url = match.group(2)
                analysis["links"].append({
                    "text": text,
                    "url": url,
                    "position": match.start()
                })
        
        # Extraer imágenes
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(image_pattern, content):
            alt_text = match.group(1)
            src = match.group(2)
            analysis["images"].append({
                "alt_text": alt_text,
                "src": src,
                "position": match.start()
            })
        
        # Extraer bloques de código
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            language = match.group(1) or "unknown"
            code = match.group(2)
            analysis["code_blocks"].append({
                "language": language,
                "code": code[:200] + "..." if len(code) > 200 else code,
                "length": len(code),
                "position": match.start()
            })
        
        return analysis
    
    def _analyze_rst(self, content: str) -> Dict[str, Any]:
        """Analiza contenido reStructuredText."""
        analysis = {
            "rst_analysis": True,
            "headers": [],
            "directives": [],
        }
        
        # Encabezados RST (líneas subrayadas)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                if next_line and all(c in '=-~^"' for c in next_line.strip()):
                    if len(next_line.strip()) >= len(line.strip()) * 0.8:
                        char = next_line.strip()[0]
                        level = {'=': 1, '-': 2, '~': 3, '^': 4, '"': 5}.get(char, 6)
                        analysis["headers"].append({
                            "level": level,
                            "title": line.strip(),
                            "line": i
                        })
        
        # Directivas RST
        directive_pattern = r'^\.\. (\w+)::'
        for match in re.finditer(directive_pattern, content, re.MULTILINE):
            directive = match.group(1)
            analysis["directives"].append({
                "directive": directive,
                "position": match.start()
            })
        
        return analysis
    
    def _analyze_html(self, content: str) -> Dict[str, Any]:
        """Analiza contenido HTML básico."""
        analysis = {
            "html_analysis": True,
            "tags": [],
            "links": [],
        }
        
        # Contar tags HTML
        tag_pattern = r'<(\w+)[^>]*>'
        tag_counts = {}
        for match in re.finditer(tag_pattern, content):
            tag = match.group(1).lower()
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        analysis["tags"] = tag_counts
        
        # Extraer enlaces
        if self.extract_links:
            link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>'
            for match in re.finditer(link_pattern, content):
                url = match.group(1)
                text = match.group(2)
                analysis["links"].append({
                    "url": url,
                    "text": text,
                    "position": match.start()
                })
        
        return analysis
    
    def _analyze_code(self, content: str) -> Dict[str, Any]:
        """Analiza contenido de código."""
        analysis = {
            "code_analysis": True,
            "estimated_language": "unknown",
            "functions": [],
            "classes": [],
            "imports": [],
        }
        
        # Detectar lenguaje de programación
        language_indicators = {
            "python": [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'class\s+\w+'],
            "javascript": [r'function\s+\w+', r'var\s+\w+', r'let\s+\w+', r'const\s+\w+'],
            "java": [r'public\s+class', r'private\s+\w+', r'public\s+static\s+void'],
            "c": [r'#include\s*<', r'int\s+main', r'printf\s*\('],
            "sql": [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
        }
        
        max_score = 0
        detected_lang = "unknown"
        
        for lang, patterns in language_indicators.items():
            score = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in patterns)
            if score > max_score:
                max_score = score
                detected_lang = lang
        
        analysis["estimated_language"] = detected_lang
        
        # Extraer funciones (patrón general)
        function_patterns = [
            r'def\s+(\w+)\s*\(',          # Python
            r'function\s+(\w+)\s*\(',     # JavaScript
            r'(\w+)\s*\([^)]*\)\s*\{',   # C/Java style
        ]
        
        for pattern in function_patterns:
            for match in re.finditer(pattern, content):
                function_name = match.group(1)
                analysis["functions"].append({
                    "name": function_name,
                    "position": match.start()
                })
        
        return analysis
    
    def _clean_content(self, raw_content: str) -> str:
        """
        Limpieza específica para texto, preservando estructura cuando sea necesario.
        """
        content = super()._clean_content(raw_content)
        
        # Preservar estructura markdown si está habilitado
        if self.preserve_markdown and self._detect_text_format(content) == "markdown":
            content = self._clean_markdown_preserving_structure(content)
        
        return content
    
    def _clean_markdown_preserving_structure(self, content: str) -> str:
        """
        Limpia markdown preservando elementos estructurales importantes.
        """
        # Preservar encabezados, listas, enlaces, pero limpiar formato excesivo
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preservar líneas estructurales importantes
            if (line.strip().startswith('#') or          # Headers
                re.match(r'^\s*[-*+]\s', line) or        # Lists
                re.match(r'^\s*\d+\.\s', line) or        # Numbered lists
                line.strip().startswith('```') or         # Code blocks
                '[' in line and '](' in line):          # Links
                cleaned_lines.append(line)
            else:
                # Limpiar líneas normales pero preservar saltos importantes
                cleaned_line = line.strip()
                if cleaned_line or (cleaned_lines and cleaned_lines[-1].strip()):
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def supports_type(self, file_path: Union[str, Path]) -> bool:
        """Verifica si el archivo es un tipo de texto soportado."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Retorna extensiones de texto soportadas."""
        return sorted(list(self.supported_extensions))