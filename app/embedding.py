import os
import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))

MODEL_NAME = "embed-multilingual-v3.0"

def get_embedding(text: str):
    response = co.embed(
        texts=[text],
        model=MODEL_NAME,
        input_type="search_document"
    )
    return response.embeddings[0]
