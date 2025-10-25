# core/vectorstore.py
import os
from pinecone import Pinecone, ServerlessSpec
# from langchain_community.embeddings import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from utils.config import settings
# from cohere import Cohere
def get_retriever(k: int = 3):
    # Init Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    
    index_name = settings.PINECONE_INDEX_NAME
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"Index '{index_name}' not found in Pinecone project.")

    # Embeddings
    # cohere_client = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=settings.COHERE_API_KEY)
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load vectorstore
    vectorstore = LC_Pinecone.from_existing_index(index_name=index_name, embedding=hf_embeddings)
    
    # Return retriever
    return vectorstore.as_retriever(search_kwargs={"k": k})
