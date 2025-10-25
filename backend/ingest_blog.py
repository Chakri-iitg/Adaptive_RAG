# ingest_blog.py
import requests
from bs4 import BeautifulSoup
from utils.config import settings
from pinecone import Pinecone, ServerlessSpec
# from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
# Blog links you want to ingest
# BLOG_URLS = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
# ]

# def fetch_content(url: str) -> str:
#     """Fetch and clean text from a blog URL."""
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#     # Remove scripts and styles
#     for script in soup(["script", "style"]):
#         script.extract()
#     text = soup.get_text(separator=" ")
#     return text.strip()

def get_pdf_texts(path):
    reader = PdfReader(path)

    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    pdf_texts = [text for text in pdf_texts if text]

    
    docs = [Document(page_content=text, metadata={"source": path}) for text in pdf_texts]

    return docs

def ingest():
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Cohere embeddings
    # embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=settings.COHERE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Fetch blogs
    docs = get_pdf_texts(settings.PDF_PATH)

    # for url in BLOG_URLS:
    #     content = fetch_content(url)
    #     docs.append(Document(page_content=content, metadata={"source": url}))
    char_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ". ", " ",""],
        chunk_size = 1500,
        chunk_overlap = 0,
    
    )
    # splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
    docs = char_splitter.split_documents(docs)
  

    # trimmed_docs = []
    # for doc in docs:
    #     doc.metadata = {
    #         "source": doc.metadata.get
    #     }
    # Store in Pinecone
    LC_Pinecone.from_documents(docs, embedding=embeddings, index_name=index_name)
    print(f"Ingested {len(docs)} documents into Pinecone index '{index_name}'")

if __name__ == "__main__":
    ingest()
