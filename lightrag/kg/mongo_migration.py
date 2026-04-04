class MongoMigration:
    def __init__(self, db, manager):
        self.db = db
        self.manager = manager

    def run(self):
        meta = self.db["meta"].find_one({"_id": "index_version"})
        if not meta or meta["version"] < self.manager.VERSION:
            print("Running index migration...")
            self.manager.ensure_indexes()
