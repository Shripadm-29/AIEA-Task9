# utils/langchain_llm.py

from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LangChainLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def query(self, prompt):
        return self.llm.predict(prompt)
