# seaarch-engine-ws

## implementing in Memory search Engine
- used codespace for contained virtual env
- started all the way from sparse bag of words for exact word match while using cosine similarity score, then to using SVD for reduced dim -> dense vector embeddings
- Using the cosine similarity score created an indices for the ranked documents

## LLM -
- Using Groq to serve Llama 3.1- 8b
- Serving with Ollama -- phi3, gemma2:2b

## Serving UI
- Streamlit UI

## containrize and docker
- docker-compose
## Elastic search
- practical use case...

## Using Codspaces
- as this project uses codespaces the library dependencies in the requirements.txt is a long list, most are pre-configured by codespace
- Docer run for elastic search within codespaces
"""
docker run -it -d \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.9.0
"""
