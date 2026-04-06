from unittest.mock import MagicMock
from lightrag.kg.mongo_index_manager import MongoIndexManager
from lightrag.kg.mongo_migration import MongoMigration
from lightrag.kg.mongo_vector_search import MongoVectorSearch

# ---------------------------------------------------------------------------
# Tests for MongoIndexManager
# ---------------------------------------------------------------------------

def test_mongo_index_manager_ensure_indexes():
    mock_db = MagicMock()
    mock_chunks = MagicMock()
    mock_meta = MagicMock()
    
    mock_db.__getitem__.side_effect = lambda x: mock_chunks if x == "chunks" else mock_meta
    
    manager = MongoIndexManager(mock_db)
    manager.ensure_indexes()
    
    # Verify text index creation
    mock_chunks.create_index.assert_any_call(
        [("content", "text")],
        name="text_index"
    )
    
    # Verify vector index creation
    mock_chunks.create_index.assert_any_call(
        [("embedding", "vector")],
        name="vector_index",
        vectorOptions={
            "dimensions": 768,
            "similarity": "cosine"
        }
    )
    
    # Verify version update in meta collection
    mock_meta.update_one.assert_called_once_with(
        {"_id": "index_version"},
        {"$set": {"version": MongoIndexManager.VERSION}},
        upsert=True
    )

# ---------------------------------------------------------------------------
# Tests for MongoMigration
# ---------------------------------------------------------------------------

def test_mongo_migration_run_needed_new_install():
    mock_db = MagicMock()
    mock_manager = MagicMock()
    mock_manager.VERSION = 1
    
    # Simulate no version info in database
    mock_db["meta"].find_one.return_value = None
    
    migration = MongoMigration(mock_db, mock_manager)
    migration.run()
    
    # Should call ensure_indexes
    mock_manager.ensure_indexes.assert_called_once()

def test_mongo_migration_run_needed_upgrade():
    mock_db = MagicMock()
    mock_manager = MagicMock()
    mock_manager.VERSION = 2
    
    # Simulate older version in database
    mock_db["meta"].find_one.return_value = {"_id": "index_version", "version": 1}
    
    migration = MongoMigration(mock_db, mock_manager)
    migration.run()
    
    # Should call ensure_indexes
    mock_manager.ensure_indexes.assert_called_once()

def test_mongo_migration_run_not_needed():
    mock_db = MagicMock()
    mock_manager = MagicMock()
    mock_manager.VERSION = 1
    
    # Simulate same version in database
    mock_db["meta"].find_one.return_value = {"_id": "index_version", "version": 1}
    
    migration = MongoMigration(mock_db, mock_manager)
    migration.run()
    
    # Should NOT call ensure_indexes
    mock_manager.ensure_indexes.assert_not_called()

# ---------------------------------------------------------------------------
# Tests for MongoVectorSearch
# ---------------------------------------------------------------------------

def test_mongo_vector_search_vector_only():
    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [{"_id": "res1"}]
    
    searcher = MongoVectorSearch(mock_collection)
    embedding = [0.1] * 768
    results = searcher.vector_search(embedding, k=5)
    
    assert len(results) == 1
    assert results[0]["_id"] == "res1"
    
    # Verify the aggregate pipeline
    args, _ = mock_collection.aggregate.call_args
    pipeline = args[0]
    assert len(pipeline) == 1
    assert "$vectorSearch" in pipeline[0]
    assert pipeline[0]["$vectorSearch"]["index"] == "vector_index"
    assert pipeline[0]["$vectorSearch"]["queryVector"] == embedding
    assert pipeline[0]["$vectorSearch"]["limit"] == 5

def test_mongo_vector_search_hybrid():
    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [{"_id": "hybrid_res"}]
    
    searcher = MongoVectorSearch(mock_collection)
    query = "find something"
    embedding = [0.2] * 768
    results = searcher.hybrid_search(query, embedding, k=3)
    
    assert len(results) == 1
    
    # Verify the aggregate pipeline has both stages
    args, _ = mock_collection.aggregate.call_args
    pipeline = args[0]
    assert len(pipeline) == 2
    assert "$search" in pipeline[0]
    assert "$vectorSearch" in pipeline[1]
    
    # Verify search params
    assert pipeline[0]["$search"]["text"]["query"] == query
    assert pipeline[0]["$search"]["index"] == "text_index"
    
    # Verify vector params
    assert pipeline[1]["$vectorSearch"]["queryVector"] == embedding
    assert pipeline[1]["$vectorSearch"]["limit"] == 3
