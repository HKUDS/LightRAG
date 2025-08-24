#!/usr/bin/env python3
"""
Ejemplo básico de uso del módulo WorldClass RAG.

Demuestra:
1. Inicialización del motor RAG
2. Procesamiento de documentos
3. Chunking inteligente
4. Búsqueda híbrida
5. Evaluación de rendimiento

Basado en las mejores prácticas del video AI News & Strategy Daily.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from worldclass_rag import (
    RAGEngine,
    SemanticChunker,
    RecursiveChunker,
    SentenceChunker,
    HybridRetriever,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    RAGEvaluator
)

from worldclass_rag.processors import TextProcessor, PDFProcessor
from worldclass_rag.core.retrieval import SemanticRetriever, KeywordRetriever
from worldclass_rag.core.evaluation import RelevanceEvaluator


def main():
    """Demostración completa del sistema WorldClass RAG."""
    
    print("🌟 WorldClass RAG - Demostración Básica")
    print("=" * 50)
    
    # 1. Configurar modelos de embeddings
    print("\n1. Configurando modelos de embeddings...")
    
    try:
        # Intentar usar OpenAI (requiere API key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        if not embeddings.is_available():
            raise Exception("OpenAI no disponible")
        print("✅ Usando OpenAI embeddings")
    except:
        # Fallback a modelo local
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Usando Sentence Transformers embeddings (local)")
    
    # 2. Crear estrategias de chunking
    print("\n2. Configurando estrategias de chunking...")
    
    # Estrategia semántica (agrupa por significado)
    semantic_chunker = SemanticChunker(
        chunk_size=800,
        chunk_overlap=150,
        similarity_threshold=0.5
    )
    
    # Estrategia recursiva (respeta estructura)
    recursive_chunker = RecursiveChunker(
        chunk_size=1000,
        chunk_overlap=200,
        preserve_sentence_boundaries=True
    )
    
    # Estrategia por oraciones (nunca rompe oraciones)
    sentence_chunker = SentenceChunker(
        chunk_size=900,
        chunk_overlap=180,
        language="es"
    )
    
    print("✅ Configuradas 3 estrategias de chunking")
    
    # 3. Crear retrievers
    print("\n3. Configurando sistema de retrieval híbrido...")
    
    semantic_retriever = SemanticRetriever(
        embedding_model=embeddings,
        similarity_threshold=0.3
    )
    
    keyword_retriever = KeywordRetriever(
        scoring_method="bm25",
        language="es"
    )
    
    # Configuración híbrida optimizada
    from worldclass_rag.core.retrieval.hybrid_retriever import HybridSearchConfig
    
    hybrid_config = HybridSearchConfig(
        semantic_weight=0.7,    # Priorizar búsqueda semántica
        keyword_weight=0.3,     # Complementar con keywords
        fusion_method="rrf",    # Reciprocal Rank Fusion
        rerank=True,            # Aplicar re-ranking
        rerank_top_k=20,
        final_top_k=5
    )
    
    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        config=hybrid_config
    )
    
    print("✅ Sistema híbrido configurado (semántica + keywords + re-ranking)")
    
    # 4. Crear documentos de ejemplo
    print("\n4. Preparando documentos de ejemplo...")
    
    sample_documents = [
        {
            "content": """
            La Recuperación Aumentada por Generación (RAG) es una técnica revolucionaria en IA 
            que combina la potencia de los Large Language Models con bases de conocimiento externas. 
            RAG permite que los modelos accedan a información actualizada y específica del dominio, 
            eliminando las limitaciones de fechas de corte de conocimiento.
            
            El proceso RAG consta de tres fases principales:
            1. Recuperación: Buscar información relevante en la base de conocimiento
            2. Aumento: Combinar la consulta con los datos recuperados  
            3. Generación: Producir una respuesta fundamentada en evidencia real
            """,
            "metadata": {
                "source": "manual_rag.pdf",
                "section": "Introducción",
                "author": "WorldClass AI Team",
                "topic": "RAG Fundamentals"
            }
        },
        {
            "content": """
            Los embeddings son representaciones vectoriales de texto en espacios de alta dimensión,
            típicamente 1,536 dimensiones según las mejores prácticas actuales. Estos vectores
            capturan el significado semántico del texto, permitiendo que significados similares
            se agrupen matemáticamente mediante similitud coseno.
            
            La elección del modelo de embeddings es crítica para el éxito de RAG:
            - OpenAI text-embedding-3-large: 3,072 dimensiones, máxima calidad
            - OpenAI text-embedding-3-small: 1,536 dimensiones, balance calidad-costo  
            - Sentence Transformers: Modelos locales, privacidad garantizada
            """,
            "metadata": {
                "source": "manual_embeddings.pdf", 
                "section": "Embeddings",
                "author": "WorldClass AI Team",
                "topic": "Vector Representations"
            }
        },
        {
            "content": """
            El chunking o segmentación de documentos es un arte y ciencia críticos en RAG.
            Una mala segmentación puede arruinar completamente un proyecto RAG.
            
            Estrategias de chunking recomendadas:
            - Chunking semántico: Agrupa por significado, preserva coherencia temática
            - Chunking recursivo: Respeta jerarquías estructurales del documento
            - Chunking por oraciones: Nunca rompe oraciones, preserva gramática
            
            Principio fundamental: SIEMPRE incluir superposición entre chunks para
            preservar contexto y evitar pérdida de información crítica.
            """,
            "metadata": {
                "source": "manual_chunking.pdf",
                "section": "Chunking Strategies", 
                "author": "WorldClass AI Team",
                "topic": "Document Processing"
            }
        },
        {
            "content": """
            La búsqueda híbrida representa el nivel 2 de RAG según las mejores prácticas.
            Combina búsqueda semántica (por significado) con búsqueda por palabras clave,
            ofreciendo mayor precisión y potencial de velocidad mejorada.
            
            Componentes de búsqueda híbrida:
            - Búsqueda vectorial: Encuentra documentos semánticamente relacionados
            - Búsqueda BM25: Captura coincidencias exactas de términos importantes  
            - Re-ranking: Mejora significativamente la precisión para propósitos comerciales
            - Fusión de resultados: Combina ambas estrategias de forma inteligente
            """,
            "metadata": {
                "source": "manual_hybrid.pdf",
                "section": "Hybrid Search",
                "author": "WorldClass AI Team", 
                "topic": "Advanced Retrieval"
            }
        },
        {
            "content": """
            Las métricas de evaluación de RAG son fundamentales para el éxito en producción.
            Según AI News & Strategy Daily, existen 4 métricas clave:
            
            1. Relevancia: ¿Se recuperan los chunks correctos?
            2. Fidelidad: ¿La respuesta se basa en fuentes reales?
            3. Calidad: ¿Un humano la calificaría como correcta?
            4. Latencia: ¿Es suficientemente rápido (menos de 2 segundos)?
            
            Un sistema RAG debe ser evaluado continuamente usando estas métricas
            para identificar problemas y oportunidades de mejora sistemática.
            """,
            "metadata": {
                "source": "manual_evaluation.pdf",
                "section": "RAG Metrics",
                "author": "WorldClass AI Team",
                "topic": "Evaluation Framework"
            }
        }
    ]
    
    print(f"✅ Preparados {len(sample_documents)} documentos de ejemplo")
    
    # 5. Procesar documentos con diferentes estrategias de chunking
    print("\n5. Procesando documentos con chunking inteligente...")
    
    all_chunks = []
    
    for i, doc in enumerate(sample_documents):
        print(f"   Procesando documento {i+1}/5...")
        
        # Usar diferentes estrategias según el contenido
        if "RAG" in doc["content"]:
            chunker = semantic_chunker  # Usar chunking semántico para contenido conceptual
        elif "embeddings" in doc["content"].lower():
            chunker = recursive_chunker  # Usar chunking recursivo para contenido técnico
        else:
            chunker = sentence_chunker  # Usar chunking por oraciones por defecto
        
        # Procesar documento
        chunks = chunker.chunk_document(
            text=doc["content"],
            source=doc["metadata"]["source"],
            section=doc["metadata"]["section"],
            additional_metadata=doc["metadata"]
        )
        
        # Añadir identificadores únicos
        for j, chunk in enumerate(chunks):
            chunk.chunk_id = f"doc_{i}_chunk_{j}"
            chunk.add_metadata("document_index", i)
        
        all_chunks.extend(chunks)
    
    print(f"✅ Generados {len(all_chunks)} chunks usando estrategias inteligentes")
    
    # 6. Indexar chunks en el sistema híbrido
    print("\n6. Indexando chunks en sistema híbrido...")
    
    # Preparar chunks para indexación
    chunks_for_indexing = []
    for chunk in all_chunks:
        chunk_data = {
            "content": chunk.content,
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.get_metadata("source", "unknown"),
            "metadata": chunk.metadata,
            "source_file": chunk.get_metadata("source"),
            "source_section": chunk.get_metadata("section"),
        }
        chunks_for_indexing.append(chunk_data)
    
    # Indexar en retriever híbrido
    hybrid_retriever.add_chunks(chunks_for_indexing)
    
    print("✅ Chunks indexados en ambos sistemas (semántico + keywords)")
    
    # 7. Realizar consultas de prueba
    print("\n7. Realizando consultas de prueba...")
    
    test_queries = [
        "¿Qué es RAG y cómo funciona?",
        "¿Cuáles son las mejores prácticas para chunking?", 
        "¿Cómo funcionan los embeddings en RAG?",
        "¿Qué métricas usar para evaluar RAG?",
        "¿Qué ventajas tiene la búsqueda híbrida?"
    ]
    
    results_summary = []
    
    for i, query in enumerate(test_queries):
        print(f"\n   Consulta {i+1}: {query}")
        
        # Realizar búsqueda híbrida
        results = hybrid_retriever.retrieve(query, top_k=3)
        
        # Mostrar resultados
        if results:
            print(f"   📊 Encontrados {len(results)} resultados relevantes:")
            for j, result in enumerate(results):
                print(f"      {j+1}. Score: {result.score:.3f} | Fuente: {result.source_id}")
                print(f"         Chunk: {result.content[:100]}...")
                
                # Información sobre el método de retrieval
                fusion_method = result.get_metadata("fusion_method", "unknown")
                print(f"         Método: {fusion_method}, Reranked: {result.get_metadata('reranked', False)}")
        else:
            print("   ❌ No se encontraron resultados")
        
        results_summary.append({
            "query": query,
            "results_count": len(results),
            "top_score": results[0].score if results else 0.0,
            "results": results
        })
    
    # 8. Evaluación de relevancia
    print("\n8. Evaluando relevancia de resultados...")
    
    # Crear evaluador de relevancia
    relevance_evaluator = RelevanceEvaluator(
        use_semantic_similarity=True,
        use_keyword_overlap=True,
        relevance_threshold=0.3
    )
    
    evaluation_results = []
    
    for query_result in results_summary:
        query = query_result["query"]
        results = query_result["results"]
        
        if results:
            # Simular respuesta generada (en implementación real vendría del LLM)
            simulated_response = f"Basándome en la información recuperada: {results[0].content[:200]}..."
            
            # Evaluar relevancia
            eval_result = relevance_evaluator.evaluate(
                query=query,
                response=simulated_response,
                retrieved_chunks=results
            )
            
            evaluation_results.append(eval_result)
            
            print(f"   📈 Relevancia para '{query[:50]}...': {eval_result.metrics.relevance:.3f}")
    
    # 9. Análisis de rendimiento
    print("\n9. Análisis de rendimiento del sistema...")
    
    # Estadísticas del retriever híbrido
    hybrid_stats = hybrid_retriever.get_stats()
    
    print("   📊 Estadísticas del sistema:")
    print(f"      - Chunks indexados: {hybrid_stats['indexed_chunks']}")
    print(f"      - Búsquedas realizadas: {hybrid_stats['search_performance']['total_searches']}")
    print(f"      - Tiempo promedio total: {hybrid_stats['search_performance']['avg_total_time']:.1f}ms")
    print(f"      - Tiempo semántico: {hybrid_stats['search_performance']['avg_semantic_time']:.1f}ms")
    print(f"      - Tiempo keywords: {hybrid_stats['search_performance']['avg_keyword_time']:.1f}ms")
    
    # Estadísticas de evaluación
    if evaluation_results:
        avg_relevance = sum(r.metrics.relevance for r in evaluation_results) / len(evaluation_results)
        print(f"      - Relevancia promedio: {avg_relevance:.3f}")
        
        passing_evals = sum(1 for r in evaluation_results if r.metrics.relevance >= 0.7)
        pass_rate = passing_evals / len(evaluation_results)
        print(f"      - Tasa de éxito (>0.7): {pass_rate:.1%}")
    
    # 10. Recomendaciones
    print("\n10. Recomendaciones del sistema...")
    
    recommendations = []
    
    if hybrid_stats['search_performance']['avg_total_time'] > 2000:  # > 2 segundos
        recommendations.append("🐌 Latencia alta detectada. Considerar optimizar embeddings o usar caché")
    
    if evaluation_results and avg_relevance < 0.7:
        recommendations.append("📉 Relevancia baja. Considerar ajustar estrategia de chunking o umbral de similitud")
    
    if not recommendations:
        recommendations.append("✅ Sistema funcionando óptimamente según mejores prácticas")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n" + "=" * 50)
    print("🎉 Demostración completada exitosamente!")
    print("\nEl sistema WorldClass RAG está listo para:")
    print("- ✅ Procesar documentos con chunking inteligente")
    print("- ✅ Realizar búsqueda híbrida (semántica + keywords)")
    print("- ✅ Aplicar re-ranking para mayor precisión")
    print("- ✅ Evaluar rendimiento con métricas clave")
    print("- ✅ Escalar a producción empresarial")


if __name__ == "__main__":
    main()