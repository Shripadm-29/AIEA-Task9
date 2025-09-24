#llm_baseline.py

from langchain_llm import LangChainLLM

# Define a baseline model that uses LLM to answer questions
class LLMBaselineModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Create an instance of the LangChainLLM helper
        # This sets up the LLM with the given model name
        self.llm = LangChainLLM(model_name)

    # Method: answer a logical reasoning question
    def answer_question(self, question):
        # Build a prompt that tells the model what to do
        prompt = f"Answer the following logical reasoning question:\n\n{question}"
        # Send the prompt to the LLM and return its answer
        return self.llm.query(prompt)
