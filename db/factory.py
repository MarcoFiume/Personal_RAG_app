from .redis_db import RedisVectorDatabase
from .sqlite_db import SQLiteVectorDatabase

def get_vector_db(settings, vector_dim):
    backend = settings['vector_db_backend'].lower()
    index_name = settings['index_name']
    
    if backend == 'redis':
        return RedisVectorDatabase(
            index_name=index_name,
            vector_dim=vector_dim,
            host=settings['redis_host'],
            port=settings['redis_port']
        )
    elif backend == 'sqlite':
        return SQLiteVectorDatabase(
            index_name=index_name,
            vector_dim=vector_dim,
            db_path=settings['sqlite_db_path']
        )
    else:
        raise ValueError(f'Unsupported vector database backend: {backend}')
