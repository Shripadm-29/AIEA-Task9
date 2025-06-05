# models/llm_baseline.py

from langchain_llm import LangChainLLM

class LLMBaselineModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = LangChainLLM(model_name)

    def answer_question(self, question):
        prompt = f"Answer the following logical reasoning question:\n\n{question}"
        return self.llm.query(prompt)
