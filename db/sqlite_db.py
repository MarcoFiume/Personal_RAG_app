class SQLiteVectorDatabase:
    def __init__(self, index_name, vector_dim, db_path):
        pass

    def _create_tables(self):
        pass

    @property
    def total_images(self):
       return 0

    @property
    def total_memory_usage(self):
        return 0

    def exists_batch(self, paths):
        pass

    def store(self, path, embedding):
        pass

    def store_batch(self, paths, embeddings):
        pass

    def search(self, query_embedding, top_k, page, page_size):
        pass

    def flush(self):
        pass