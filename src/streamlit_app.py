import streamlit as st
import time

from elasticsearch import Elasticsearch
from openai import OpenAI

client =  OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
)

es_client = Elasticsearch("http://localhost:9200")

def elastic_search(query, index_name = "course_faq_questions"):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index= index_name, body= search_query)

    hits_results = []
    for hit in response['hits']['hits']:
        hits_results.append(hit['_source'])

    return hits_results

def build_prompt(query, search_results):
    prompt_template = """
You are an expert machine learning and mlops engineering helping a junior engineer as an assitant and guide.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering. DO NOT USE OTHER CONTENT OTHER THAN GIVEN CONTEXT!
if the CONTEXT does not contain the answer, Output "Not FOUND in the context given" and explain your answer with reasons.

QUESTION: {question}

CONTEXT: {context}
""".strip()

    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm_call(prompt):

    response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt,}],
    model="gemma2:2b",
)
    return response.choices[0].message.content

def rag(query):

    search_results = elastic_search(query)
    Prompt = build_prompt(query, search_results)
    answer = llm_call(Prompt)
    return answer


def main():
    st.title("RAG Function Invocation")

    user_input = st.text_input("Enter your input:")

    if st.button("Ask"):
        with st.spinner('Processing...'):
            output = rag(user_input)
            st.success("Completed!")
            st.write(output)

if __name__ == "__main__":
    main()
