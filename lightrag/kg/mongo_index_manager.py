class MongoIndexManager:
    VERSION = 1

    def __init__(self, db):
        self.db = db

    def ensure_indexes(self):
        col = self.db["chunks"]

        col.create_index(
            [("content", "text")],
            name="text_index"
        )

        col.create_index(
            [("embedding", "vector")],
            name="vector_index",
            vectorOptions={
                "dimensions": 768,
                "similarity": "cosine"
            }
        )

        self.db["meta"].update_one(
            {"_id": "index_version"},
            {"$set": {"version": self.VERSION}},
            upsert=True
        )
