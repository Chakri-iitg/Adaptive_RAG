from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from core.llm_client import generate_text
def web_search(state: dict):
    question = state["question"]

    tool = TavilySearchResults(k=3)

    res = tool.invoke({"query":question})

    contents = [r.get("content", r.get("snippet", "")) for r in res]

    doc = Document(page_content="\n\n".join(contents))

    prompt = (
        "You are an AI assistant. Use the context below to answer the question truthfully and concisely.\n\n"
        f"Context:\n{doc}\n\nQuestion: {question}\n\nAnswer:"
    )
    gen = generate_text(prompt)

    # generation = "\n\n".join(contents)

    return {"question":question, "documents":[doc], "generation": gen}