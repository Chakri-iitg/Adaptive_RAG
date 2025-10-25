from langgraph.graph import StateGraph, START, END
# from langchain.schema import Document
from langchain_core.documents import Document
from typing import Any, Dict, List
from core.retrieval import retrieve as core_retrieve
from core.graders import (
    grade_document_relevance,
    check_hallucination,
    check_answer_addresses
)
from core.query_rewriter import rewrite_query
from core.llm_client import generate_text
from core.web_search import web_search as core_web_search

GraphState = Dict[str,Any]

def node_retrieve(state: GraphState) -> GraphState:
    print("---NODE: retrieve---")

    return core_retrieve(state)

def node_grade_documents(state: GraphState) -> GraphState:
    print("--NODE: grade_documents---")
    question = state["question"]
    documents = state.get("documents",[])
    filtered = []

    for d in documents:
        text = d.page_content if hasattr(d,"page_content") else str(d)
        ok = grade_document_relevance(question,text)
        print(f"graded doc -> {ok}")

        if ok:
            filtered.append(d)
        
    return {"documents":filtered, "question": question}

def node_transform_query(state: GraphState) -> GraphState:
    print("---NODE: transform_query---")
    return rewrite_query(state)

def node_web_search(state: GraphState) -> GraphState:
    print("---NODE: web_search---")
    return core_web_search(state)

def node_generate(state: GraphState) -> GraphState:
    print("---NODE: generate---")
    q = state["question"]
    docs = state.get("documents", [])
    context = "\n\n".join([d.page_content if hasattr(d, "page_content") else str(d) for d in docs])
    prompt = (
        "You are an AI assistant. Use the context below to answer the question truthfully and concisely.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
    )
    gen = generate_text(prompt)
    return {"question":q, "documents":docs, "generation": gen}

def node_grade_generation(state: GraphState) -> GraphState:
    print("---NODE: grade_generation---")
    question = state["question"]
    documents = state.get("documents",[])
    generation = state.get("generation","")

    grounded = check_hallucination(documents, generation)
    addresses = check_answer_addresses(question, generation)

    print(f" grounded = {grounded}, addresses = {addresses}")

    if not grounded:
        state["decision"] = "not_supported"
    elif not addresses:
        state["decision"] = "not_useful"
    else:
        state["decision"] = "useful"
    
    return state

def build_rag_graph():
    workflow = StateGraph(dict)

    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents",node_grade_documents)
    workflow.add_node("transform_query",node_transform_query)
    workflow.add_node("web_search",node_web_search)
    workflow.add_node("generate",node_generate)
    workflow.add_node("grade_generation",node_grade_generation)

    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve","grade_documents")

    def decide_after_grade(state: GraphState):

        print("---EDGE: decide_after_grade---")

        filtered = state.get("documents",[])

        if not filtered:
            return "transform_query"
        
        return "generate"

    workflow.add_conditional_edges("grade_documents", decide_after_grade, {"transform_query": "transform_query", "generate":"generate"})
    
    workflow.add_edge("transform_query","retrieve")
    workflow.add_edge("generate","grade_generation")

    def decide_after_gen(state: GraphState):
        dec = state.get("decision","")
        print(f"---EDGE: decide_after_gen -> {dec}---")

        if dec =="useful":
            return END 
        if dec =="not_useful":
            return "transform_query"
        if dec == "not_supported":
            return "generate"
        
        return "transform_query"

    workflow.add_conditional_edges("grade_generation", decide_after_gen, {
        "generate": "generate",
        "transform_query": "transform_query",
         END : END
    })

    compiled = workflow.compile()

    return compiled 

compiled_rag = build_rag_graph()

def run_rag_graph(question: str):
    print(f"Running RAG Graph for: {question}")
    inputs = {"question": question}
    last_state = None 
    for output in compiled_rag.stream(inputs):

        for node, state in output.items():
            print(f"Node '{node}' executed. State keys: {list(state.keys())}")
            last_state = state 
        print("----")
    
    return last_state