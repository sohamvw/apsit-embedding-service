from fastapi import FastAPI, HTTPException
from app.schemas import EmbedRequest, EmbedResponse
from app.embedding import get_embedding

app = FastAPI(title="APSIT Embedding Service (Cohere)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    embeddings = []

    for text in request.texts:
        vector = get_embedding(text)

        # Validate dimension (Cohere multilingual = 1024)
        if len(vector) != 1024:
            raise HTTPException(
                status_code=500,
                detail="Embedding dimension mismatch (expected 1024)"
            )

        embeddings.append(vector)

    return {"embeddings": embeddings}
