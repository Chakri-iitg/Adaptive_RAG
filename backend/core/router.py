from core.llm_client import generate_text

def route_question(state: dict):
    question = state["question"]

    prompt = (
        "You are an expert at routing a user question to a vectorstore or web search.\n"
        "The vectorstore contains documents related to Gen AI Tools, agents, prompt engineering, and adversarial attacks.\n\n"
        "Use the vectorstore for questions on these topics. Otherwise, use web-search."
        f"Question: {question}\n\n Choice:"
    )

    out = generate_text(prompt,max_tokens=20)

    out = out.lower()

    if "web" in out:
        return "web_search"
    
    return "vectorstore"



