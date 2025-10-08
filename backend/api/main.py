from fastapi import FastAPI
from .routes import chat

app = FastAPI(title="Adaptive RAG API")
app.include_router(chat.router, prefix="/api")

@app.get("/health")
def health():
    return {"status":"ok"}