import re 
from typing import Any, Dict, List


@transformer
def chunk_documents(data: List[Dict[str, Any]], *args, **kwargs):
    documents = []

    for idx,document in enumerate(data):
        course = item['course']

        for info in item['documents']:
            section = info['section']
            question = info['question']
            answer = info['text']

        document_id = ':'.join([re.sub(r'\w', '_', part) for part in [course, section, question]]).lower()

        # Format the document string
        chunk = '\n'.join([
            f'course:\n{course}\n',
            f'section:\n{section}\n',
            f'question:\n{question}\n',
            f'answer:\n{answer}\n',
        ])

        documents.append(dict(
            chunk=chunk,
            document=info,
                document_id=document_id
        ))

    print(f"Documents:", len(documents))

    return [documents]