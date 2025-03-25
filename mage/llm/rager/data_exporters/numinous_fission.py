import json
from typing import Dict, List, Tuple, Union
import numpy as np 
from elasticsearch import Elasticsearch


@data_exporter
def elasticsearch(documents: List[Dict[str, Union[Dict, List[int],str]]], *args, **kwargs):
    connection_string= kwargs.get('connection_string', 'http://localhost:9200')
    index_name = kwargs.get('index_name', 'documents')
    number_of_shards = kwargs.get('number_of_shards', 1)
    number_of_replicas = kwargs.get('number_of_replicas', 0)
    dimentions = kwargs.get('dimentions')

    if dimentions is None and len(documents) > 0:
        documents = documents[0]
        dimentions = len(document.get('embedding') or [])
    
    es_client = Elasticsearch(connection_string)

    print(f'Connecting to Elasticsearch at {connection_string}')

    index_settings = {
        "Settings" : {
            "number_of_shards": number_of_shards,
            "number_of_replicas": number_of_replicas,
        },
        "mappings": {
            "properties" : {
                "chunk" : {"type": "text"},
                "dimention_id" : {"type" : "text"},
                "embedding": {"type": "dense_vector", "dims":dimentions}
            }
        }
    }

    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        print(f"Index {index_name} deleted")

    es_client.indices.create(index=index_name, body=index_settings)
    print("Index created with properties:")
    print(json.dumps(index_settings, indent=2))
    print("Embedding dimentions:", dimentions)

    count = len(documents)
    print(f"Indexing {count} documents to Elasticsearch index {index_name}")
    for idx, document in enumerate(documents):
        if idx % 100 == 0:
            print(f"{idx + 1}/{count}")
        if isinstance(document['embedding'], np.ndarray):
            document['embedding'] = document['embedding'].tolist()

        es_client..index(index=index_name, document=document)

    return [[d['embedding'] for d in documents[:10]]]