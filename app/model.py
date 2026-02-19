from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")

_model = None


def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = load_model()

    # E5 requires prefix
    formatted_texts = [f"query: {text}" for text in texts]

    embeddings = model.encode(
        formatted_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()
