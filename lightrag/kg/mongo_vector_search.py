
class MongoVectorSearch:
    def __init__(self, collection):
        self.collection = collection

    def vector_search(self, embedding, k=10):
        return list(self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": k
                }
            }
        ]))

    def hybrid_search(self, query, embedding, k=10):
        return list(self.collection.aggregate([
            {
                "$search": {
                    "index": "text_index",
                    "text": {"query": query, "path": "content"}
                }
            },
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": k
                }
            }
        ]))
