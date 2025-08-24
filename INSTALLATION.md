# WorldClass RAG - Guía de Instalación

Esta guía te ayudará a instalar y configurar WorldClass RAG, un módulo de clase mundial para Retrieval-Augmented Generation.

## 🎯 Requisitos del Sistema

### Mínimos
- Python 3.9 o superior
- 4GB RAM disponible
- 2GB espacio en disco
- Acceso a internet (para descargar modelos)

### Recomendados para Producción
- Python 3.11 o superior
- 16GB+ RAM
- SSD con 100GB+ espacio libre
- GPU (opcional, para modelos de embeddings personalizados)

## 🚀 Instalación Rápida

### 1. Clonar o Descargar el Proyecto

```bash
# Si usas git
git clone <repository-url>
cd worldclass-rag

# O descarga y extrae el archivo ZIP
```

### 2. Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 3. Instalar Dependencias Base

```bash
# Instalar dependencias core
pip install -r requirements.txt

# O instalar manualmente
pip install pydantic sentence-transformers chromadb numpy scikit-learn
```

## 📦 Instalación por Componentes

### Componente Base (Siempre Requerido)
```bash
pip install pydantic sentence-transformers chromadb numpy
```

### Procesamiento de PDFs
```bash
pip install pypdf
# Para PDFs más complejos (opcional):
pip install PyMuPDF
```

### Procesamiento de Imágenes con OCR
```bash
# Instalar dependencias Python
pip install Pillow pytesseract

# Instalar Tesseract (sistema)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-spa

# macOS:
brew install tesseract tesseract-lang

# Windows: Descargar desde https://github.com/UB-Mannheim/tesseract/wiki
```

### Procesamiento de Documentos Office
```bash
pip install python-docx python-pptx openpyxl pandas
```

### Modelos de Lenguaje (OpenAI)
```bash
pip install openai
export OPENAI_API_KEY="tu-api-key-aqui"
```

### Procesamiento Avanzado de Lenguaje
```bash
pip install spacy
python -m spacy download es_core_news_sm  # Español
python -m spacy download en_core_web_sm   # Inglés
```

### API REST
```bash
pip install fastapi uvicorn httpx
```

### Seguridad Empresarial
```bash
pip install presidio-analyzer presidio-anonymizer cryptography
```

## 🔧 Configuración

### 1. Variables de Entorno (Opcional)

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Configuración de embeddings
RAG_EMBEDDING_PROVIDER=sentence_transformers
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Para usar OpenAI (opcional)
OPENAI_API_KEY=tu-api-key-aqui
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small

# Configuración de seguridad
RAG_ENABLE_PII_DETECTION=true
RAG_GDPR_COMPLIANCE=true

# Configuración de rendimiento
RAG_ENABLE_CACHE=true
RAG_CACHE_TYPE=memory

# Entorno
RAG_ENVIRONMENT=development
RAG_DEBUG=true
RAG_LOG_LEVEL=INFO
```

### 2. Verificar Instalación

```python
# Ejecutar ejemplo básico
python examples/basic_usage.py
```

## 🧪 Verificación de Instalación

### Test Rápido
```python
import sys
sys.path.append('src')

from worldclass_rag import RAGEngine
from worldclass_rag.core.embeddings import SentenceTransformerEmbeddings

# Crear modelo de embeddings
embeddings = SentenceTransformerEmbeddings()
print(f"✅ Embeddings funcionando: {embeddings.model_name}")

# Crear motor RAG
engine = RAGEngine()
print("✅ Motor RAG inicializado correctamente")
```

### Test Completo
```bash
# Ejecutar tests unitarios (después de instalar pytest)
pip install pytest
pytest tests/ -v
```

## 🚨 Solución de Problemas Comunes

### Error: "sentence-transformers not found"
```bash
pip install sentence-transformers
# Si persiste el error:
pip install torch torchvision transformers
```

### Error: "pytesseract not found"
```bash
# Instalar tesseract primero (ver arriba)
# Luego instalar Python package:
pip install pytesseract
```

### Error: "spacy model not found"
```bash
# Descargar modelos de spaCy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

### Error: "No module named 'worldclass_rag'"
```bash
# Asegurar que estás en el directorio correcto
cd worldclass-rag
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# O usar instalación en modo desarrollo
pip install -e .
```

### Problemas de Memoria
```bash
# Para sistemas con poca RAM, usar modelos más pequeños
export RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2  # 80MB
# En lugar de all-mpnet-base-v2 (420MB)
```

## 🔧 Instalación en Diferentes Entornos

### Docker (Recomendado para Producción)
```dockerfile
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY examples/ ./examples/

CMD ["python", "examples/basic_usage.py"]
```

### Google Colab
```python
# Celda 1: Instalar dependencias
!pip install sentence-transformers chromadb pypdf

# Celda 2: Clonar repositorio
!git clone <repository-url>
%cd worldclass-rag

# Celda 3: Ejecutar ejemplo
!python examples/basic_usage.py
```

### Jupyter Notebook
```python
# Instalar en notebook
import sys
!{sys.executable} -m pip install sentence-transformers chromadb

# Configurar path
import sys
sys.path.append('src')

# Usar normalmente
from worldclass_rag import RAGEngine
```

## 📈 Configuración para Producción

### 1. Optimización de Rendimiento
```python
from worldclass_rag.config import RAGConfig

config = RAGConfig()
config.performance.enable_cache = True
config.performance.batch_embeddings = True
config.performance.vector_db_sharding = True
```

### 2. Seguridad Empresarial
```python
config.security.enable_pii_detection = True
config.security.gdpr_compliance = True
config.security.audit_logging = True
config.security.encrypt_at_rest = True
```

### 3. Monitoreo
```python
config.monitoring.enable_metrics = True
config.monitoring.enable_alerting = True
config.monitoring.latency_threshold_ms = 2000
```

## 🆘 Soporte

Si encuentras problemas durante la instalación:

1. **Revisa los logs**: Los errores suelen dar pistas específicas
2. **Verifica versiones**: `python --version`, `pip --version`
3. **Limpia caché de pip**: `pip cache purge`
4. **Reinstala en entorno limpio**: Crea un nuevo virtual environment
5. **Revisa dependencias del sistema**: Especialmente tesseract para OCR

## 🎉 ¡Listo!

Una vez completada la instalación, puedes:

1. Ejecutar `python examples/basic_usage.py` para ver una demostración completa
2. Explorar la documentación en `docs/`
3. Revisar más ejemplos en `examples/`
4. Comenzar a integrar WorldClass RAG en tu proyecto

¡WorldClass RAG está listo para potenciar tus aplicaciones con RAG de clase mundial! 🚀