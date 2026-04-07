class MongoVectorSearch:
    def __init__(self, collection):
        self.collection = collection

    def vector_search(self, embedding, k=10):
        return list(
            self.collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": embedding,
                            "numCandidates": 100,
                            "limit": k,
                        }
                    }
                ]
            )
        )

    def hybrid_search(self, query, embedding, k=10):
        return list(
            self.collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": embedding,
                            "numCandidates": 100,
                            "limit": k * 2,
                        }
                    },
                    {
                        "$match": {
                            "$or": [
                                {"content": {"$regex": query, "$options": "i"}},
                                {"_id": {"$regex": query, "$options": "i"}},
                            ]
                        }
                    },
                    {"$limit": k},
                ]
            )
        )
