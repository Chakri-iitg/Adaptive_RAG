from core.llm_client import generate_text

def rewrite_query(state: dict):
    question = state["question"]

    prompt = (
        "Rewrite the following user question to improve retrieval in vector search. keep meaning the same, be concise.\n\n"
        f"Original: {question}\n\n Rewritten:"
    )

    rewritten = generate_text(prompt, max_tokens=80)

    return {"question": rewritten.strip(), "documents": state.get("documents", [])}
