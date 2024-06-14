from fastapi import FastAPI
from pydantic import BaseModel
from backend.app.chatbot import LlamaIndexChatbot
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

app = FastAPI()
chatbot = LlamaIndexChatbot(input_files=["transformers.pdf"])

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    bot_response: str

class DocMetadata(BaseModel):
    doc_metadata: str

@app.get("/")
def healthy_check() -> dict:
    """
    Health check endpoint.

    :return: Dictionary indicating the health status
    """
    return {'status': 'healthy'}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint to get a response from the chatbot.

    :param request: ChatRequest containing the user input
    :return: ChatResponse containing the bot's response
    """
    response = chatbot.get_response(request.user_input)
    return ChatResponse(bot_response=response)


# @app.post("/metadata", response_model=DocMetadata)
# def chat(request: ChatRequest) -> DocMetadata:
#     """
#     Chat endpoint to get a metadata of the document(s).

#     :param request: DocMetadata containing the user input
#     :return: DocMetadata containing the document's metadata
#     """
#     response = chatbot.get_metadata(request.user_input)
#     return DocMetadata(bot_response=response)