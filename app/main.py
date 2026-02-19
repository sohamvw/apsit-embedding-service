from fastapi import FastAPI, HTTPException
from app.schemas import EmbedRequest, EmbedResponse
from app.model import embed_texts
import os

app = FastAPI(title="APSIT Embedding Service")


@app.on_event("startup")
def startup_event():
    # Force model load at startup
    from app.model import load_model
    load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    embeddings = embed_texts(request.texts)

    # Dimension validation (production safety)
    for vector in embeddings:
        if len(vector) != 768:
            raise HTTPException(
                status_code=500,
                detail="Embedding dimension mismatch"
            )

    return {"embeddings": embeddings}
