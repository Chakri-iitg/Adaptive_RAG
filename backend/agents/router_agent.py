from core.router import route_question
from agents.rag_agent_langgraph import run_rag_graph
from core.web_search import web_search

def router_agent(question:str):
    state = {"question":question}

    decision = route_question(state)

    if decision == "web_search":
        return web_search(state)
    
    return run_rag_graph(question)