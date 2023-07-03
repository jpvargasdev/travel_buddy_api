import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.chatgpt.chat import Chat
from src.agent.agent import Agent

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
def analyze_text(message: Message):
    def generate_text(message: str):
        for i in agent.agent_chat(message):
            yield i.encode("utf-8")

    response = generate_text(message.message)

    return StreamingResponse(response, media_type="text/plain")


 
