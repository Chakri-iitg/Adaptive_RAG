from core.llm_client import generate_text

def grade_document_relevance(question: str, document_text: str) -> bool:

    prompt = (
         "You are a grader assessing relevance of a retrieved document to a user question. \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
   " It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n"
    " Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        f"Question: {question}\n\n Document: {document_text}\n\n Relevant?"
    )

    out = generate_text(prompt, max_tokens=40)

    return "yes" in out.lower()

def check_hallucination(documents: list, generation: str) -> bool:

    docs_text = "\n\n".join([d.page_content if hasattr(d, "page_content") else str(d) for d in documents])
    prompt = (
        "You are a grader assessing whether the LLM generation is grounded in the facts provided below.\n\n"
        f"Facts:\n{docs_text}\n\nGeneration:\n{generation}\n\n"
        "Is the generation supported by the facts? Answer 'yes' or 'no'."
    )

    out = generate_text(prompt, max_tokens = 40)

    return "yes" in out.lower()

def check_answer_addresses(question: str, generation: str) -> bool:
    prompt = (
        "You are a grader. Does the answer below resolve the user's question? Respond 'yes' or 'no'.\n\n"
        f"Question:{question}\n\n Answer: {generation}\n\n"
    )

    out = generate_text(prompt, max_tokens=40)

    return "yes" in out.lower()
