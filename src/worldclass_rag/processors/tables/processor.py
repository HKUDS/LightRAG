"""
Table Processor - Manejo especializado de tablas con codificación de relaciones espaciales.

Implementa las mejores prácticas del video AI News & Strategy Daily:
"Tablas requieren manejo especial porque hay que codificar relaciones espaciales"
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import io

# Table processing libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False

from ..base import DocumentProcessor


class TableProcessor(DocumentProcessor):
    """
    Procesador especializado para archivos de tabla (CSV, Excel, etc.).
    
    Características clave siguiendo mejores prácticas:
    - Codificación de relaciones espaciales entre celdas
    - Preservación de estructura jerárquica de headers
    - Detección automática de tipos de datos
    - Conversión inteligente a formato textual para RAG
    - Manejo robusto de formatos Excel complejos
    
    El desafío principal es convertir datos tabulares 2D en representación
    textual que preserve las relaciones semánticas para recuperación efectiva.
    """
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        extract_metadata: bool = True,
        preserve_structure: bool = True,
        max_rows_preview: int = 100,
        include_column_types: bool = True,
        create_searchable_text: bool = True,
        preserve_formulas: bool = False,
        handle_merged_cells: bool = True,
    ):
        """
        Inicializa el procesador de tablas.
        
        Args:
            max_rows_preview: Máximo filas para preview en metadatos
            include_column_types: Si incluir información de tipos de datos
            create_searchable_text: Si crear representación textual optimizada para búsqueda
            preserve_formulas: Si preservar fórmulas Excel (requiere openpyxl)
            handle_merged_cells: Si manejar celdas combinadas correctamente
        """
        super().__init__(
            remove_boilerplate=remove_boilerplate,
            normalize_whitespace=normalize_whitespace,
            extract_metadata=extract_metadata,
            preserve_structure=preserve_structure,
        )
        
        self.max_rows_preview = max_rows_preview
        self.include_column_types = include_column_types
        self.create_searchable_text = create_searchable_text
        self.preserve_formulas = preserve_formulas
        self.handle_merged_cells = handle_merged_cells
        
        # Verificar dependencias
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas es requerido para procesar tablas")
        
        # Extensiones soportadas
        self.supported_extensions = {'.csv', '.xlsx', '.xls', '.tsv', '.ods'}
        
        # Configurar encodings comunes para CSV
        self.csv_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def _extract_content(self, source: Union[str, Path, bytes]) -> Tuple[str, Dict[str, Any]]:
        """
        Extrae contenido de archivos de tabla con preservación de estructura.
        """
        extraction_metadata = {
            "table_type": "unknown",
            "sheets": [],
            "total_rows": 0,
            "total_columns": 0,
            "column_info": {},
            "has_headers": False,
        }
        
        try:
            # Determinar tipo de archivo y procesarlo
            if isinstance(source, bytes):
                # Intentar detectar formato de bytes
                tables_data = self._process_bytes_data(source, extraction_metadata)
            else:
                path = Path(source)
                extension = path.suffix.lower()
                
                if extension == '.csv':
                    tables_data = self._process_csv(path, extraction_metadata)
                elif extension in ['.xlsx', '.xls']:
                    tables_data = self._process_excel(path, extraction_metadata)
                elif extension == '.tsv':
                    tables_data = self._process_tsv(path, extraction_metadata)
                elif extension == '.ods':
                    tables_data = self._process_ods(path, extraction_metadata)
                else:
                    raise ValueError(f"Formato de archivo no soportado: {extension}")
            
            # Convertir datos tabulares a representación textual
            text_content = self._convert_tables_to_text(tables_data, extraction_metadata)
            
            return text_content, extraction_metadata
            
        except Exception as e:
            raise Exception(f"Error procesando tabla: {e}")
    
    def _process_csv(self, file_path: Path, metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """Procesa archivo CSV con detección automática de encoding y delimitador."""
        metadata["table_type"] = "csv"
        
        # Intentar diferentes encodings
        for encoding in self.csv_encodings:
            try:
                # Detectar delimitador
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(1024)
                
                # Probar delimitadores comunes
                for delimiter in [',', ';', '\t', '|']:
                    if sample.count(delimiter) > sample.count(','):
                        break
                else:
                    delimiter = ','
                
                # Leer CSV
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    dtype=str,  # Leer todo como string inicialmente
                    na_filter=False  # Preservar strings vacíos
                )
                
                metadata.update({
                    "encoding": encoding,
                    "delimiter": delimiter,
                    "rows": len(df),
                    "columns": len(df.columns),
                })
                
                return [df]
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == self.csv_encodings[-1]:  # Último intento
                    raise e
                continue
        
        raise Exception("No se pudo leer el archivo CSV con ningún encoding probado")
    
    def _process_excel(self, file_path: Path, metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """Procesa archivo Excel con manejo de múltiples hojas."""
        metadata["table_type"] = "excel"
        
        # Leer información de hojas
        if file_path.suffix.lower() == '.xlsx' and OPENPYXL_AVAILABLE:
            engine = 'openpyxl'
        else:
            engine = 'xlrd' if XLRD_AVAILABLE else None
        
        if not engine:
            raise ImportError("Se requiere openpyxl o xlrd para leer archivos Excel")
        
        # Leer todas las hojas
        excel_file = pd.ExcelFile(file_path, engine=engine)
        sheets_data = []
        sheets_metadata = []
        
        for sheet_name in excel_file.sheet_names:
            try:
                # Leer hoja
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    dtype=str,
                    na_filter=False
                )
                
                if not df.empty:
                    sheets_data.append(df)
                    
                    sheet_meta = {
                        "name": sheet_name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "has_data": True
                    }
                    
                    # Detectar si tiene headers
                    if self._detect_headers(df):
                        sheet_meta["has_headers"] = True
                        metadata["has_headers"] = True
                    
                    sheets_metadata.append(sheet_meta)
                    
            except Exception as e:
                print(f"Warning: Error leyendo hoja '{sheet_name}': {e}")
                sheets_metadata.append({
                    "name": sheet_name,
                    "error": str(e),
                    "has_data": False
                })
        
        metadata["sheets"] = sheets_metadata
        metadata["total_sheets"] = len(excel_file.sheet_names)
        
        return sheets_data
    
    def _process_tsv(self, file_path: Path, metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """Procesa archivo TSV (Tab-Separated Values)."""
        metadata["table_type"] = "tsv"
        
        for encoding in self.csv_encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter='\t',
                    dtype=str,
                    na_filter=False
                )
                
                metadata.update({
                    "encoding": encoding,
                    "delimiter": "\\t",
                    "rows": len(df),
                    "columns": len(df.columns),
                })
                
                return [df]
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == self.csv_encodings[-1]:
                    raise e
                continue
        
        raise Exception("No se pudo leer el archivo TSV")
    
    def _process_ods(self, file_path: Path, metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """Procesa archivo ODS (OpenDocument Spreadsheet)."""
        metadata["table_type"] = "ods"
        
        try:
            # pandas puede leer ODS con odfpy
            df = pd.read_excel(file_path, engine='odf', dtype=str, na_filter=False)
            
            metadata.update({
                "rows": len(df),
                "columns": len(df.columns),
            })
            
            return [df]
            
        except ImportError:
            raise ImportError("Se requiere odfpy para leer archivos ODS")
        except Exception as e:
            raise Exception(f"Error leyendo archivo ODS: {e}")
    
    def _process_bytes_data(self, data: bytes, metadata: Dict[str, Any]) -> List[pd.DataFrame]:
        """Procesa datos de tabla desde bytes."""
        # Intentar detectar formato
        data_str = data.decode('utf-8', errors='ignore')[:1000]
        
        if data_str.startswith('PK'):  # Posible Excel
            # Crear archivo temporal para Excel
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                return self._process_excel(Path(tmp.name), metadata)
        else:
            # Asumir CSV
            try:
                df = pd.read_csv(io.BytesIO(data), dtype=str, na_filter=False)
                metadata.update({
                    "table_type": "csv_bytes",
                    "rows": len(df),
                    "columns": len(df.columns),
                })
                return [df]
            except Exception as e:
                raise Exception(f"Error procesando datos como CSV: {e}")
    
    def _detect_headers(self, df: pd.DataFrame) -> bool:
        """
        Detecta si la primera fila contiene headers.
        
        Heurísticas:
        - Primera fila tiene tipos diferentes al resto
        - Nombres descriptivos vs datos
        - Patrones de texto vs números
        """
        if len(df) < 2:
            return False
        
        first_row = df.iloc[0]
        second_row = df.iloc[1]
        
        # Contar tipos de datos diferentes
        first_row_numeric = sum(1 for val in first_row if str(val).replace('.', '').replace('-', '').isdigit())
        second_row_numeric = sum(1 for val in second_row if str(val).replace('.', '').replace('-', '').isdigit())
        
        # Si la primera fila tiene menos números que la segunda, probablemente son headers
        if first_row_numeric < second_row_numeric and second_row_numeric > len(df.columns) * 0.3:
            return True
        
        # Verificar longitud promedio (headers suelen ser más cortos)
        first_row_avg_len = sum(len(str(val)) for val in first_row) / len(first_row)
        rest_avg_len = sum(len(str(val)) for val in df.iloc[1:].values.flatten()) / (len(df) - 1) / len(df.columns)
        
        if first_row_avg_len < rest_avg_len * 0.7:
            return True
        
        return False
    
    def _convert_tables_to_text(self, tables: List[pd.DataFrame], metadata: Dict[str, Any]) -> str:
        """
        Convierte datos tabulares a representación textual optimizada para RAG.
        
        Implementa la codificación de relaciones espaciales mencionada en el video.
        """
        text_parts = []
        
        for i, df in enumerate(tables):
            if df.empty:
                continue
            
            # Información de la tabla
            table_info = []
            
            # Metadatos de la tabla
            sheet_name = ""
            if metadata.get("sheets") and i < len(metadata["sheets"]):
                sheet_name = metadata["sheets"][i].get("name", f"Tabla_{i+1}")
                table_info.append(f"=== {sheet_name} ===")
            else:
                table_info.append(f"=== Tabla {i+1} ===")
            
            table_info.append(f"Dimensiones: {len(df)} filas x {len(df.columns)} columnas")
            
            # Analizar tipos de columnas
            if self.include_column_types:
                column_analysis = self._analyze_columns(df)
                metadata["column_info"][sheet_name or f"table_{i}"] = column_analysis
                
                # Añadir información de columnas al texto
                col_descriptions = []
                for col, info in column_analysis.items():
                    col_desc = f"{col} ({info['type']}"
                    if info.get('unique_values') and info['unique_values'] < 20:
                        col_desc += f", valores: {', '.join(map(str, info['sample_values'][:5]))}"
                    col_desc += ")"
                    col_descriptions.append(col_desc)
                
                table_info.append("Columnas: " + " | ".join(col_descriptions))
            
            # Crear representación textual de la tabla
            if self.create_searchable_text:
                table_text = self._create_searchable_table_text(df, sheet_name)
                table_info.append("\nContenido:")
                table_info.append(table_text)
            else:
                # Representación simple de la tabla
                table_info.append("\nDatos:")
                table_info.append(df.to_string(max_rows=self.max_rows_preview, max_cols=None))
            
            text_parts.append("\n".join(table_info))
        
        # Actualizar metadatos globales
        if tables:
            metadata["total_rows"] = sum(len(df) for df in tables)
            metadata["total_columns"] = max(len(df.columns) for df in tables if not df.empty)
        
        return "\n\n" + "="*50 + "\n\n".join(text_parts)
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analiza cada columna para determinar tipo de datos y características.
        """
        column_analysis = {}
        
        for col in df.columns:
            series = df[col].dropna()
            
            if series.empty:
                column_analysis[col] = {"type": "empty", "null_count": len(df)}
                continue
            
            analysis = {
                "null_count": df[col].isnull().sum(),
                "unique_values": series.nunique(),
                "sample_values": series.unique()[:5].tolist(),
            }
            
            # Detectar tipo de datos
            # Intentar convertir a numérico
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isnull().all():
                if numeric_series.dtype == 'int64' or (numeric_series == numeric_series.astype(int)).all():
                    analysis["type"] = "integer"
                else:
                    analysis["type"] = "float"
                analysis["min"] = numeric_series.min()
                analysis["max"] = numeric_series.max()
                analysis["mean"] = numeric_series.mean()
            
            # Verificar si es fecha
            elif self._is_date_column(series):
                analysis["type"] = "date"
                try:
                    date_series = pd.to_datetime(series, errors='coerce')
                    analysis["date_range"] = {
                        "start": date_series.min().isoformat() if pd.notna(date_series.min()) else None,
                        "end": date_series.max().isoformat() if pd.notna(date_series.max()) else None
                    }
                except:
                    pass
            
            # Verificar si es categórico
            elif analysis["unique_values"] < len(series) * 0.5 and analysis["unique_values"] < 50:
                analysis["type"] = "categorical"
                value_counts = series.value_counts().head(10).to_dict()
                analysis["value_counts"] = value_counts
            
            # Por defecto, texto
            else:
                analysis["type"] = "text"
                analysis["avg_length"] = series.str.len().mean()
                analysis["max_length"] = series.str.len().max()
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Detecta si una columna contiene fechas."""
        # Probar convertir a fecha una muestra
        sample = series.head(min(100, len(series)))
        
        try:
            date_series = pd.to_datetime(sample, errors='coerce')
            valid_dates = date_series.dropna()
            
            # Si más del 70% se convierte exitosamente, es probablemente fecha
            return len(valid_dates) / len(sample) > 0.7
        except:
            return False
    
    def _create_searchable_table_text(self, df: pd.DataFrame, table_name: str = "") -> str:
        """
        Crea representación textual optimizada para búsqueda semántica.
        
        Codifica relaciones espaciales de manera que sean comprensibles
        para modelos de lenguaje y búsqueda vectorial.
        """
        searchable_parts = []
        
        # Headers como contexto
        if self._detect_headers(df):
            headers = df.columns.tolist()
            searchable_parts.append(f"Esta tabla contiene las siguientes columnas: {', '.join(headers)}")
        
        # Descripción por filas con contexto relacional
        for idx, row in df.iterrows():
            if idx >= self.max_rows_preview:
                searchable_parts.append(f"... (tabla continúa con {len(df) - idx} filas más)")
                break
            
            # Crear descripción natural de la fila
            row_description = []
            
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    # Formato natural: "columna: valor"
                    row_description.append(f"{col}: {value}")
            
            if row_description:
                # Crear oración descriptiva
                row_text = f"Fila {idx + 1} - " + " | ".join(row_description)
                searchable_parts.append(row_text)
        
        # Añadir resumen estadístico si es útil
        if self.include_column_types:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                summary_parts = []
                for col in numeric_cols:
                    col_stats = df[col].describe()
                    summary_parts.append(f"{col}: promedio {col_stats['mean']:.2f}, rango {col_stats['min']:.2f}-{col_stats['max']:.2f}")
                
                if summary_parts:
                    searchable_parts.append(f"Resumen estadístico: {' | '.join(summary_parts)}")
        
        return "\n".join(searchable_parts)
    
    def supports_type(self, file_path: Union[str, Path]) -> bool:
        """Verifica si el archivo es una tabla soportada."""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def get_supported_extensions(self) -> List[str]:
        """Retorna extensiones de tabla soportadas."""
        return sorted(list(self.supported_extensions))