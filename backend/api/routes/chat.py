from fastapi import APIRouter
from agents.router_agent import router_agent
from pydantic import BaseModel

router = APIRouter()

class Query(BaseModel):
    question: str
    
@router.post("/chat")
async def chat(q: Query):

    out = router_agent(q.question)

    docs = []

    for d in out.get("documents",[]):
        if hasattr(d, "page_content"):
            docs.append(d.page_content)
        else:
            docs.append(str(d))
    
    return {
        "question": out.get("question", q.question),
        "answer": out.get("generation",""),
        "documents":docs
    }