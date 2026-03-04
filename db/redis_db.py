import redis
import numpy as np
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

SEARCH_ALGORITHM = 'HNSW'
DISTANCE_METRIC = 'COSINE'
DTYPE = 'FLOAT32'

class RedisVectorDatabase:
    def __init__(self, index_name, vector_dim, host, port):
        self.r = redis.Redis(host=host, port=port, decode_responses=False)
        self.index_name = index_name
        self.vector_dim = vector_dim
        self._create_index()

    @property
    def total_images(self):
        info = self.r.ft(self.index_name).info()
        return info['num_docs']

    @property
    def total_memory_usage(self):
        return self.r.info()['used_memory_rss_human']

    def _create_index(self):
        """Create a vector index if it doesn't already exist."""
        try:
            self.r.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            schema = [
                VectorField(
                    'embedding',
                    SEARCH_ALGORITHM,
                    {
                        'TYPE': DTYPE,
                        'DIM': self.vector_dim,
                        'DISTANCE_METRIC': DISTANCE_METRIC,
                    },
                )
            ]
            self.r.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=['img:'], index_type=IndexType.HASH),
            )

    def exists_batch(self, paths):
        """Return the subset of paths already stored in the DB."""
        pipe = self.r.pipeline()
        for path in paths:
            pipe.exists(f'img:{path}')
        results = pipe.execute()
        return {path for path, exists in zip(paths, results) if exists}

    def store(self, path, embedding):
        key = f'img:{path}'
        self.r.hset(key, mapping={
            'embedding': embedding.astype(np.float32).tobytes()
        })

    def store_batch(self, paths, embeddings):
        pipe = self.r.pipeline()
        for path, embedding in zip(paths, embeddings):
            key = f'img:{path}'
            pipe.hset(key, mapping={'embedding': embedding.tobytes()})
        pipe.execute()

    def search(self, query_embedding, top_k = 5, page = 0, page_size = 10):
        """Find the top_k most similar documents to the query embedding."""
        query_vector = query_embedding.tobytes()

        query = (
            Query(f'*=>[KNN {top_k} @embedding $vector AS score]')
            .return_fields('score')
            .sort_by('score')
            .paging(page * page_size, page_size)
            .dialect(2)
        )

        results = self.r.ft(self.index_name).search(query, {'vector': query_vector})
        return [
            {'path': img.id.removeprefix('img:'), 'score': float(img.score)}
            for img in results.docs
        ]

    def flush(self) -> None:
        self.r.flushdb()
        self._create_index()
