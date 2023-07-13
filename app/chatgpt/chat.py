import os
import openai
import requests

URL_MODELS = "https://api.openai.com/v1/models"
URL_COMPLETIONS = "https://api.openai.com/v1/chat/completions"

class Chat:
    def __init__(self):
        GPT_KEY = os.getenv("OPENAI_API_KEY")
        self.HEADERS = {"Authorization": f"Bearer {GPT_KEY}"}


    def getModels(self) -> str:
        return requests.get(URL_MODELS, headers=self.HEADERS).json()
 
    def message(self, message: str) -> str:
        body = {
            "model": "gpt-3.5",
            "messages": [{"role": "system","content": "You are a helpful assistant."},{"role": "user","content": message}]
        }

        return requests.post(URL_COMPLETIONS, headers=self.HEADERS, json=body).json()
