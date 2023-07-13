import os
import json

from typing import Dict, List, Any, Tuple

from langchain import PromptTemplate
from langchain.tools import BaseTool

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools 
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

from app.agent.customtools.files_handler_tool import tool_add_row_to_csv, tool_create_csv_file, tool_delete_csv_file, tools_default_file_management, tool_get_current_date

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
        template = """The following is a friendly conversation between a human and an AI travel assistant. The AI is talkative and provides lots of specific details from its context.
            If the AI does not know the answer to a question, it truthfully says it does not know. 


            {chat_history}
            Current conversation
            {chat_history}

            Human: {input}
            AI trael assistant:"""

        prompt = PromptTemplate(
                input_variables=["chat_history", "input"],
                template=template,
                )

        memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
                )

        # LLM CHATGPT chat
        self.llm_chat = ChatOpenAI(
                temperature=0,
                model=LLM_CHAT, client="",
                streaming=True,
                callbacks=[stream_handler]
                )
        # Agent initialization
        tools = self._init_tools()

        # CHATGPT Agent
        self.agent_adv = initialize_agent(
                tools,
                self.llm_chat, 
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                prompt=prompt,
                )

   
    def _init_tools(self) -> List[BaseTool]:

        default_tools = [
                "serpapi",
                "llm-math",
                "openweathermap-api",
                ]

        tools = load_tools(
                default_tools,
                llm=self.llm_chat
                )

        custom_tools = [
                Tool(
                    name="Create CSV",
                    func=tool_create_csv_file.run,
                    description="""
                        Useful when you need to create a csv file,
                        the tool receives two parameters, the first one is the name of the file and the second one is an array of strings,
                        both are mandatory, and both parameters as a single string.
                        """
                    ),
                Tool(
                    name="Add row to CSV",
                    func=tool_add_row_to_csv.run,
                    description="""
                        Useful when you need to edit a csv file,
                        the tool receives two parameters,
                        the first one is the name of the file and the second one is an array of strings or just empty value,
                        which are the items of the row, both are mandatory, and both parameters as single string. At the end of the second paramenter
                        you will add the current date using the current date function. 
                        """
                    ),
                Tool(
                    name="Delete CSV",
                    func=tool_delete_csv_file.run,
                    description="""Useful when you need to delete a csv file, one parameter, the csvfile name"""
                    ),
                Tool(
                    name="Get current Date",
                    func=tool_get_current_date.run,
                    description="""Useful when u need to get the current date"""
                    )
                ]

        tools = tools + custom_tools + tools_default_file_management
        
        return tools

    def conversation_chat(self, message):
        values = self.agent_adv({
            "input": message
            })
        return self.format_response(values)

    def format_response(self, values) -> str:
        print("values:", values)
        result = json.dumps(values["output"], indent=2)
        if len(result) > 0:
            return json.dumps(values["output"], indent=2)
        else:
            return result


