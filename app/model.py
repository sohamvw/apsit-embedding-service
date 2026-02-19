from sentence_transformers import SentenceTransformer
import os
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-base")

_model = None


def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )
        torch.set_grad_enabled(False)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = load_model()

    formatted_texts = [f"query: {text}" for text in texts]

    embeddings = model.encode(
        formatted_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8
    )

    return embeddings.tolist()
