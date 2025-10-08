# 🚀 Adaptive RAG – Intelligent Retrieval-Augmented Generation System  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangChain](https://img.shields.io/badge/LangChain-AI%20Orchestration-orange)
![Cohere](https://img.shields.io/badge/Cohere-Embeddings-yellow)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A production-grade Retrieval-Augmented Generation (RAG) backend pipeline that adaptively improves its responses through feedback loops and dynamic query transformations.

---

## 🧠 Overview  

**Adaptive RAG** is an intelligent document-questioning system built with **LangChain**, **LangGraph**, and **FastAPI**.  
Unlike static RAG systems, this project uses **adaptive retrieval and grading agents** that evaluate whether the retrieved context is relevant — and reformulate queries automatically to improve accuracy.  

The pipeline ensures **fewer hallucinations**, **smarter retrieval**, and **better factual consistency** for question-answering tasks.  

---

## 🏗️ Architecture  
```
    User Query
        │
        ▼
┌────────────────────────────┐
│ Router Agent (LLM)         │   ← decides "vectorstore" or "web_search"
│ (LangChain prompt)         │
└────────────────────────────┘
    │                    │
    │ vectorstore        │ web_search
    ▼                    ▼
┌───────────────────┐   ┌────────────────────┐
│ RAG Subgraph      │   │ Web Search Agent   │
│ (LangGraph)       │   │ (Tavily search)    │
└───────────────────┘   └────────────────────┘
      │                        │
      ▼                        ▼
  [Retrieve]               [Return web docs]
  (Pinecone via Cohere)        │
      │                        │
      ▼                        └──────┐
┌───────────────────┐                 │
│ Retrieval Grader  │ <─(LLM) evaluates relevance
│ (LLM)             │                 ▼
└───────────────────┘            ┌─────────────────┐
      │ if no relevant docs      │ Generate Answer │
      │ ───────────────────────► │ (Cohere RAG)    │
      ▼                          └─────────────────┘
┌───────────────────┐                   │
│ Query Rewriter    │ ──(LLM rewrites)──┘
│ (LLM)             │
└───────────────────┘
      │
      └─(loop)──► Retrieve ─► Grade ─► (once good) ► Generate
                                                
After generation:
┌────────────────────────────────────────────────────────┐
│ Generation Graders (LLM):                              │
│  - Hallucination Grader (is output grounded in docs?)  │
│  - Answer Grader (does it answer the question?)        │
└────────────────────────────────────────────────────────┘
      │
      ▼
  If useful → Return response (FastAPI → Streamlit)
  Else → Transform query (or retry generation) and loop back
```


---

## ⚙️ Tech Stack  

| Component | Technology |
|------------|-------------|
| **Backend** | FastAPI |
| **Orchestration** | LangChain + LangGraph |
| **Embeddings** | Cohere API |
| **Vector Database** | Pinecone |
| **Document Parsing** | LangChain Document Loaders |
| **Language Model** | Cohere Command-R / other LLMs |

---

## ✨ Key Features  

✅ **Adaptive Retrieval Loop** – The system dynamically refines queries when context relevance is low.  
✅ **Multi-Agent Pipeline** – Separate agents for grading, transforming, and generating responses.  
✅ **Vector Search** – Uses Pinecone for efficient semantic document retrieval.  
✅ **Scalable API** – Built with FastAPI for production-level deployment.  
✅ **Modular Design** – Each component (retriever, grader, generator) is pluggable.  

---

## 📁 Project Structure  
```
Adaptive-RAG/
│
├── backend/
│ ├── api/ # FastAPI endpoints
│ ├── agents/ # LangChain-based agents
│ ├── core/ # Graph building logic
│ ├── utils/ # Config and helper functions
│ ├── ingest_blog.py # Script to ingest text into vector DB
│ ├── main.py # API entry point
│ ├── requirements.txt
│
├── .env.example # Template for environment variables
├── README.md
└── .gitignore
```




## 🔧 Setup & Installation  
```
1️⃣ Clone the repository
bash
git clone https://github.com/<your-username>/Adaptive-RAG.git
cd Adaptive-RAG/backend

2️⃣ Create a virtual environment

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Add environment variables

Create a .env file in the backend folder:

COHERE_API_KEY=your_cohere_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
5️⃣ Ingest documents

Run the ingestion script to embed data into Pinecone:

python ingest_blog.py

6️⃣ Run the backend
uvicorn main:app --reload
```