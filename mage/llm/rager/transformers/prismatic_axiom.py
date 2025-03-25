from typing import Dict, List

import numpy as np 
import spacy


@transformer
def spacy_embeddings(documents: List[Dict], *args, **kwargs) -> List[Dict]:
    count = len(documents)
    print("Documents", count)

    data = []

    for idx, document in enumerate(documents):
        document_id = document['document_id']
        if idx % 100 == 0:
            print(f"{idx + 1}/{count}")

        nlp = spacy.load("en_core_web_sm")
        tokens = document["tokens"]

        text = ' '.join(tokens)
        doc = nlp(text)

        embedding = np.mean([token.vector for token in doc], axis=0).tolist()

        data.append(dict(
            chunk = document['chunk'],
            document_id = document['document_id'],
            embedding = embedding
        ))
    return [data]