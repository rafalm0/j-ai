from together import Together
from typing import List
from keys import api_key
import numpy as np
from time import sleep
from tqdm import tqdm
import json
import os


def vector_retreival(client, query: str, top_k: int = 5, vector_index: np.ndarray = None) -> List[int]:
    """
    Retrieve the top-k most similar items from an index based on a query.
    Args:
        client : client api object
        query (str): The query string to search for.
        top_k (int, optional): The number of top similar items to retrieve. Defaults to 5.
        vector_index (np.ndarray, optional): The index array containing embeddings to search against. Defaults to None.
    Returns:
        List[int]: A list of indices corresponding to the top-k most similar items in the index.
    """

    query_embedding = np.array(generate_embeddings(client, [query], 'BAAI/bge-large-en-v1.5')[0])

    similarity_scores = np.dot(query_embedding, vector_index.T)

    return list(np.argsort(-similarity_scores)[:top_k])


def rerank(client, query: str, chunks: List[str], top_k=3) -> List[int]:
    response = client.rerank.create(
        model="Salesforce/Llama-Rank-V1",
        query=query,
        documents=chunks,
        top_n=top_k
    )

    return [result.index for result in response.results]


def create_chunks(document, chunk_size=300, overlap=50):
    return [document[i: i + chunk_size] for i in range(0, len(document), chunk_size - overlap)]


def generate_embeddings(client, input_texts: List[str], model_api_string: str) -> np.ndarray:
    """Generate embeddings from Together python library.

    Args:
        client : client api object
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    outputs = client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return np.array([x.embedding for x in outputs.data])


def load_and_embed_jsonl(paths: list[str], embedding_model="BAAI/bge-large-en-v1.5"):
    enriched = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                json_line = json.loads(line)
                embedding = generate_embeddings(client,[json_line["chunk"]], embedding_model)[0]
                json_line["embedding"] = embedding.tolist()
                enriched.append(json_line)

    return enriched


def save_embedded_jsonl(path: str, enriched_data: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in enriched_data:
            f.write(json.dumps(item) + "\n")


def load_embeddings(path):
    jsonl = []
    with open(path, "r", encoding="utf-8") as f:
        for line_ in tqdm(f.readlines()):
            data_ = json.loads(line_)
            jsonl.append(data_)
        return jsonl


# EXAMPLE USAGE

if __name__ == "__main__":
    year = '2024'

    client = Together(api_key=api_key)

    if year == '2024':
        subject = 'the arrival of the generative AI and the effects on journalist'
    else:
        subject = 'the arrival of the internet and the affects on journalist'
    jsonl_path = [f"RAG-processed/nyt_{year}_full_clean.jsonl", f"RAG-processed/nyt_{year}_full_clean-2.jsonl"]
    embedding_output = f"RAG-embeddings/nyt_{year}_embedded.jsonl"
    query = f"What did people think of the {subject} in {year}?"

    if not os.path.exists(embedding_output):
        data = load_and_embed_jsonl(jsonl_path)
        save_embedded_jsonl(embedding_output, data)
    else:
        data = load_embeddings(embedding_output)

    print("---------------- testing the RAG -----------------------------------------")
    chunks = [d["chunk"] for d in data]
    embeddings = np.array([d["embedding"] for d in data])

    top_k_indices = vector_retreival(query=query, top_k=6, vector_index=embeddings, client=client)
    top_k_chunks = [chunks[i] for i in top_k_indices]

    reranked_indices = rerank(query=query, chunks=top_k_chunks, top_k=4, client=client)
    reranked_chunks = "\n\n".join([top_k_chunks[i] for i in reranked_indices])

    chosen_articles = [data[i] for i in reranked_indices]

    print("RERANKED CHUNKS:")
    print(reranked_chunks)

    # You can also chat with the context
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user",
             "content": f"Answer the question: {query}. Use this info as extra source from interview of the new york times:\n\n{reranked_chunks}"},
        ],
    )
    print("\nRESPONSE:")
    print(response.choices[0].message.content)
