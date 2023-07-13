from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chatgpt.chat import Chat
from app.agent.agent import Agent

from langchain.agents.conversational_chat.base import ConversationalChatAgent

ConversationalChatAgent._validate_tools = lambda *_, **__: None 

load_dotenv()

app = FastAPI()
chat = Chat()
agent = Agent()

class Message(BaseModel):
    message: str

@app.get("/chat/models")
async def get_models():
    return chat.getModels()

@app.post("/chat/message")
def call_agent(message: Message):
    def generate_text(message: str):
        for i in agent.conversation_chat(message):
            yield i.encode("utf-8")

    response = generate_text(message.message)
    
    return StreamingResponse(response, media_type="text/plain")

