from fastapi import FastAPI
from pydantic import BaseModel
from .chatbot import *

app = FastAPI()
chatbot = LlamaIndexChatbot()

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    bot_response: str

@app.get("/")
def healthy_check():
    return {'status': 'healthy'}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = chatbot.get_response(request.user_input)
    return ChatResponse(bot_response=response)
