#langchain_llm.py

from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file into the program
load_dotenv()

# Define a helper class for using LangChain's LLM easily
class LangChainLLM:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Create an instance of the ChatOpenAI model
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    # Send a prompt to the LLM and return its prediction
    def query(self, prompt):
        return self.llm.predict(prompt)  # Calls LangChain's predict method
