from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import LlamaIndexChatbot

app = FastAPI()
chatbot = LlamaIndexChatbot()

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    bot_response: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = chatbot.get_response(request.user_input)
    return ChatResponse(bot_response=response)
