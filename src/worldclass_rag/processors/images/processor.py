"""
Image Document Processor - Procesador para imágenes con OCR y análisis.

Extrae texto de imágenes usando OCR optimizado, siguiendo las mejores prácticas
del video AI News & Strategy Daily.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import tempfile
import base64

# Image processing libraries
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from ..base import DocumentProcessor


class ImageProcessor(DocumentProcessor):
    """
    Procesador especializado para imágenes con capacidades avanzadas de OCR.
    
    Características:
    - OCR multi-idioma optimizado
    - Pre-procesamiento de imagen para mejor reconocimiento
    - Detección de orientación y corrección automática
    - Extracción de metadatos de imagen
    - Manejo de diferentes formatos de imagen
    
    Siguiendo las mejores prácticas:
    - "optical character recognition is correct" es fundamental
    - Optimización de imagen antes del OCR
    - Manejo robusto de diferentes calidades de imagen
    """
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        ocr_language: str = "spa+eng",
        enhance_image: bool = True,
        auto_rotate: bool = True,
        min_confidence: float = 30.0,
        denoise: bool = True,
    ):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            ocr_language: Idiomas para OCR (formato tesseract)
            enhance_image: Si mejorar imagen antes del OCR
            auto_rotate: Si detectar y corregir orientación
            min_confidence: Confianza mínima para aceptar texto OCR
            denoise: Si aplicar reducción de ruido
        """
        super().__init__(
            remove_boilerplate=remove_boilerplate,
            normalize_whitespace=normalize_whitespace,
            extract_metadata=extract_metadata,
            preserve_structure=preserve_structure,
        )
        
        self.ocr_language = ocr_language
        self.enhance_image = enhance_image
        self.auto_rotate = auto_rotate
        self.min_confidence = min_confidence
        self.denoise = denoise
        
        # Verificar dependencias
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow es requerido para procesar imágenes")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract es requerido para OCR")
        
        # Extensiones soportadas
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
            '.webp', '.gif', '.svg', '.ico'
        }
    
    def _extract_content(self, source: Union[str, Path, bytes]) -> Tuple[str, Dict[str, Any]]:
        """
        Extrae texto de imagen usando OCR optimizado.
        """
        extraction_metadata = {
            "extraction_method": "ocr",
            "ocr_language": self.ocr_language,
            "image_enhancements": [],
            "ocr_confidence": 0.0,
        }
        
        try:
            # Cargar imagen
            if isinstance(source, bytes):
                import io
                image = Image.open(io.BytesIO(source))
                extraction_metadata["source_type"] = "bytes"
            else:
                image = Image.open(str(source))
                path = Path(source)
                extraction_metadata.update({
                    "source_type": "file",
                    "file_size": path.stat().st_size,
                    "original_format": image.format,
                })
            
            # Extraer metadatos de imagen
            img_metadata = self._extract_image_metadata(image)
            extraction_metadata.update(img_metadata)
            
            # Pre-procesar imagen para mejor OCR
            processed_image = self._preprocess_image(image, extraction_metadata)
            
            # Aplicar OCR
            text, ocr_confidence = self._apply_ocr(processed_image)
            
            extraction_metadata["ocr_confidence"] = ocr_confidence
            
            # Validar calidad del resultado
            if ocr_confidence < self.min_confidence:
                extraction_metadata["warning"] = f"Baja confianza OCR: {ocr_confidence:.1f}%"
            
            return text, extraction_metadata
            
        except Exception as e:
            raise Exception(f"Error procesando imagen: {e}")
    
    def _extract_image_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extrae metadatos comprehensivos de la imagen."""
        metadata = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
        }
        
        # EXIF data si está disponible
        try:
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    # Extraer campos EXIF importantes
                    exif_fields = {
                        'DateTime': 306,
                        'Software': 305,
                        'Artist': 315,
                        'Copyright': 33432,
                        'Orientation': 274,
                    }
                    
                    extracted_exif = {}
                    for field_name, field_code in exif_fields.items():
                        if field_code in exif:
                            extracted_exif[field_name] = str(exif[field_code])
                    
                    if extracted_exif:
                        metadata["exif"] = extracted_exif
        except Exception:
            pass  # EXIF opcional
        
        # Calcular estadísticas de imagen
        if image.mode in ('L', 'RGB'):
            try:
                # Convertir a array numpy si OpenCV está disponible
                if OPENCV_AVAILABLE:
                    img_array = np.array(image)
                    metadata.update({
                        "mean_brightness": float(np.mean(img_array)),
                        "std_brightness": float(np.std(img_array)),
                    })
            except Exception:
                pass
        
        # Determinar si parece ser un documento escaneado
        aspect_ratio = image.width / image.height
        metadata["aspect_ratio"] = aspect_ratio
        metadata["likely_document"] = (
            0.7 <= aspect_ratio <= 1.5 and  # Proporción de documento típica
            image.width >= 300 and          # Resolución mínima razonable
            image.height >= 300
        )
        
        return metadata
    
    def _preprocess_image(self, image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
        """
        Pre-procesa imagen para optimizar OCR.
        
        Aplica mejoras según las características detectadas.
        """
        processed = image.copy()
        enhancements = []
        
        # Convertir a RGB si es necesario
        if processed.mode != 'RGB':
            processed = processed.convert('RGB')
            enhancements.append("converted_to_rgb")
        
        # Detectar y corregir orientación si está habilitado
        if self.auto_rotate:
            try:
                # Buscar información de orientación en EXIF
                exif = metadata.get("exif", {})
                orientation = exif.get("Orientation")
                
                if orientation:
                    # Aplicar rotación según EXIF
                    rotation_map = {
                        '3': 180, '6': 270, '8': 90
                    }
                    if orientation in rotation_map:
                        processed = processed.rotate(rotation_map[orientation], expand=True)
                        enhancements.append(f"rotated_{rotation_map[orientation]}_degrees")
                
                # Si no hay EXIF, intentar detección automática
                elif TESSERACT_AVAILABLE:
                    osd = pytesseract.image_to_osd(processed, output_type=pytesseract.Output.DICT)
                    rotation = osd.get('rotate', 0)
                    if abs(rotation) > 1:  # Solo rotar si la rotación es significativa
                        processed = processed.rotate(-rotation, expand=True)
                        enhancements.append(f"auto_rotated_{-rotation}_degrees")
                        
            except Exception:
                pass  # Rotación es opcional
        
        # Mejoras de imagen si está habilitado
        if self.enhance_image:
            processed = self._enhance_image_for_ocr(processed, enhancements)
        
        metadata["image_enhancements"] = enhancements
        return processed
    
    def _enhance_image_for_ocr(self, image: Image.Image, enhancements: List[str]) -> Image.Image:
        """
        Aplica mejoras específicas para optimizar OCR.
        """
        enhanced = image
        
        try:
            # Aumentar contraste si la imagen es muy clara/oscura
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            enhancements.append("contrast_enhanced")
            
            # Ajustar nitidez
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            enhancements.append("sharpness_enhanced")
            
            # Reducir ruido si está habilitado y OpenCV disponible
            if self.denoise and OPENCV_AVAILABLE:
                # Convertir PIL a OpenCV
                cv_image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
                
                # Aplicar filtro de reducción de ruido
                denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
                
                # Convertir de vuelta a PIL
                enhanced = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                enhancements.append("denoised")
            
            # Redimensionar si es muy pequeña (para mejor OCR)
            width, height = enhanced.size
            if width < 1000 or height < 1000:
                # Escalar manteniendo proporción, mínimo 1000px en lado más corto
                scale_factor = 1000 / min(width, height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
                enhancements.append(f"upscaled_{scale_factor:.1f}x")
        
        except Exception as e:
            print(f"Warning: Error en mejora de imagen: {e}")
            return image  # Retornar original si hay error
        
        return enhanced
    
    def _apply_ocr(self, image: Image.Image) -> Tuple[str, float]:
        """
        Aplica OCR a la imagen procesada.
        
        Returns:
            Tuple de (texto_extraído, confianza_promedio)
        """
        try:
            # Configuración optimizada de tesseract
            custom_config = r'--oem 3 --psm 6'  # PSM 6: single uniform block of text
            
            # Extraer texto con información detallada
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.ocr_language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Procesar resultados OCR
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    confidence = int(ocr_data['conf'][i])
                    if confidence > 0:  # Filtrar palabras sin confianza
                        words.append(word)
                        confidences.append(confidence)
            
            # Calcular texto final y confianza promedio
            text = ' '.join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text, avg_confidence
            
        except Exception as e:
            print(f"Error en OCR: {e}")
            return "", 0.0
    
    def _clean_content(self, raw_content: str) -> str:
        """
        Limpieza específica para texto extraído por OCR.
        
        El texto OCR a menudo tiene artefactos específicos que necesitan limpieza.
        """
        content = super()._clean_content(raw_content)
        
        if not content:
            return content
        
        # Limpieza específica para OCR
        content = self._clean_ocr_artifacts(content)
        
        return content
    
    def _clean_ocr_artifacts(self, text: str) -> str:
        """
        Limpia artefactos comunes del OCR.
        """
        import re
        
        # Corregir espaciado incorrecto en números y fechas
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
        
        # Corregir puntuación separada incorrectamente
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remover caracteres extraños comunes en OCR
        # Mantener solo caracteres imprimibles y espacios
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Corregir palabras partidas por líneas (común en OCR de documentos)
        # Buscar patrones como "pala- bra" y convertir a "palabra"
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Limpiar múltiples espacios consecutivos
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Limpiar líneas que son solo caracteres extraños
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Filtrar líneas que son principalmente caracteres no-alfabéticos
            alpha_chars = sum(1 for c in line if c.isalpha())
            total_chars = len(line.strip())
            
            if total_chars == 0:
                continue
            elif total_chars < 3:
                continue
            elif alpha_chars / total_chars >= 0.3:  # Al menos 30% caracteres alfabéticos
                cleaned_lines.append(line)
            elif any(word.isdigit() for word in line.split()):  # Contiene números válidos
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def enhance_image_quality(self, image_path: Union[str, Path]) -> Optional[str]:
        """
        Crea una versión mejorada de la imagen optimizada para OCR.
        
        Returns:
            Ruta al archivo temporal con la imagen mejorada, o None si hay error.
        """
        try:
            image = Image.open(str(image_path))
            
            # Aplicar mejoras
            enhanced = self._enhance_image_for_ocr(image, [])
            
            # Guardar en archivo temporal
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            enhanced.save(temp_file.name, 'PNG')
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error mejorando imagen: {e}")
            return None
    
    def extract_text_with_coordinates(self, source: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Extrae texto con información de coordenadas para preservar layout.
        
        Útil para documentos donde la posición del texto es importante.
        """
        try:
            # Cargar imagen
            if isinstance(source, bytes):
                import io
                image = Image.open(io.BytesIO(source))
            else:
                image = Image.open(str(source))
            
            # Pre-procesar
            processed = self._preprocess_image(image, {})
            
            # OCR con coordenadas
            ocr_data = pytesseract.image_to_data(
                processed,
                lang=self.ocr_language,
                output_type=pytesseract.Output.DICT
            )
            
            # Organizar datos por coordenadas
            text_blocks = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip() and int(ocr_data['conf'][i]) > self.min_confidence:
                    text_blocks.append({
                        'text': word,
                        'confidence': int(ocr_data['conf'][i]),
                        'bbox': {
                            'x': int(ocr_data['left'][i]),
                            'y': int(ocr_data['top'][i]), 
                            'width': int(ocr_data['width'][i]),
                            'height': int(ocr_data['height'][i])
                        },
                        'block_num': int(ocr_data['block_num'][i]),
                        'par_num': int(ocr_data['par_num'][i]),
                        'line_num': int(ocr_data['line_num'][i]),
                        'word_num': int(ocr_data['word_num'][i])
                    })
            
            return {
                'text_blocks': text_blocks,
                'full_text': ' '.join(block['text'] for block in text_blocks),
                'image_dimensions': {'width': processed.width, 'height': processed.height}
            }
            
        except Exception as e:
            raise Exception(f"Error extrayendo texto con coordenadas: {e}")
    
    def supports_type(self, file_path: Union[str, Path]) -> bool:
        """Verifica si el archivo es una imagen soportada."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Retorna extensiones de imagen soportadas."""
        return sorted(list(self.supported_extensions))