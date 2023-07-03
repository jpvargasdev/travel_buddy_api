import os
import json

from typing import Dict, List, Any, Tuple

from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, ConversationChain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import FinalStreamingStdOutCallbackHandler

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

LLM_COMPLETION = "text-davinci-003"
LLM_CHAT = "gpt-3.5-turbo"

class CustomStreamingStdOutCallbackHandler(FinalStreamingStdOutCallbackHandler):
    buffer: List[Tuple[str, float]] = []
    stop_token = "#!stop!#"

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        return super().on_llm_start(serialized, prompts, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        return super().on_llm_new_token(token, **kwargs)
    

stream_handler = CustomStreamingStdOutCallbackHandler()


class Agent:
    def __init__(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="input", output_key="output")

        self.llm = OpenAI(temperature=0, model=LLM_COMPLETION, client="", streaming=True, callbacks=[stream_handler])
        self.llm_chat = ChatOpenAI(temperature=0, model=LLM_CHAT, client="", streaming=True, callbacks=[stream_handler])
        self.conversation = ConversationChain(llm=self.llm, verbose=True)

        tools = load_tools(["serpapi", "llm-math", "openweathermap-api", "human"], llm=self.llm)

        self.agent_conversation = initialize_agent(
                tools,
                self.llm_chat, 
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                )
        self.agent = initialize_agent(
                tools, 
                self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                verbose=True, 
                memory=memory,
                return_intermediate_steps=True,
                )
       
    def chat(self, message):
        response = self.llm_chat.predict_messages([HumanMessage(content=message)])
        return response

    def predict(self, message):
        response = self.llm.predict(message)
        return response

    def agent_chat(self, message):
        values = self.agent({
            "input": message
            })

        print(values)
        response = json.dumps(values["output"], indent=2)

        return response


    def conversation_chat(self, message):
        values = self.agent_conversation({
            "input": message
            })

        print(values)
        response = json.dumps(values["output"], indent=2)

        return response
