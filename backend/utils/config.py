import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME","adaptive-rag-index")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

settings = Settings() 