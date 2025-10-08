# core/retrieval.py
from core.vectorstore import get_retriever

def retrieve(state: dict):
    """
    Retrieve relevant documents from Pinecone.
    """
    retriever = get_retriever(k=5)
    question = state.get("question", "")

    docs = retriever.get_relevant_documents(question)
    state["documents"] = docs
    return state
