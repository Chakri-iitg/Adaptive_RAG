from langchain_community.chat_models import ChatCohere
from utils.config import settings

def get_chat_llm(model: str ="command-a-03-2025", temperature:float = 0.0):

    return ChatCohere(model=model, temperature=temperature, cohere_api_key=settings.COHERE_API_KEY)

def generate_text(prompt: str, model: str = "command-a-03-2025", max_tokens: int = 256):

    llm = get_chat_llm(model = model)

    resp = llm.invoke(prompt)

    if hasattr(resp,"content"):
        return resp.content 
    if hasattr(resp,"generations"):
        return resp.generations[0].text
    
    return str(resp)