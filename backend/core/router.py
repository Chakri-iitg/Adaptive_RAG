from core.llm_client import generate_text

def route_question(state: dict):
    question = state["question"]

    prompt = (
        "You are an expert at routing a user question to a vectorstore or web search.\n"
        ''' You have access to two knowledge sources:
                1. Encyclopedia of Medicine (retrieved from a local vector database)
                2. Web Search Tool (for questions not covered in the Encyclopedia)

            Your goal:
                - Always prioritize giving accurate, factual, and helpful answers.
                - Use the Encyclopedia content if the user’s question relates to diseases, symptoms, treatments, anatomy, or other medical topics covered in the book.
                - If the user's question is unrelated, or the vector database returns low similarity results, use the Web Search Tool.
                - Never combine data from both sources unless explicitly needed.
                - When using web data, mention: “According to recent web information…” to distinguish it.
                - Keep your tone professional, concise, and clear.'''
        "Use the vectorstore for questions on these topics. Otherwise, use web-search."
        f"Question: {question}\n\n Choice:"
    )

    out = generate_text(prompt,max_tokens=20)

    out = out.lower()

    if "web" in out:
        return "web_search"
    
    return "vectorstore"



