# models/logic_lm.py

from langchain_llm import LangChainLLM
from logic_utils import LogicSolver, check_logic_validity

class LogicLMModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = LangChainLLM(model_name)
        self.solver = LogicSolver()

    def logic_translate(self, question):
        prompt = (
            "Translate the following description into symbolic Prolog-style logic facts and rules."
            " Only output facts and rules."
            " Do NOT include any queries, answers, or explanations."
            " Ensure correct syntax with a period at the end of each fact and rule."
            " Do NOT use markdown formatting or code blocks."
            "\n\n"
            f"{question}"
        )
        return self.llm.query(prompt)

    def refine_logic(self, broken_logic, errors):
        error_message = "\n".join(errors)
        prompt = (
            f"The following Prolog-like logic contains syntax errors:\n"
            f"{error_message}\n\n"
            "Please correct only the syntax errors while keeping ALL the original facts and rules."
            " Do NOT add, remove, or answer anything."
            " Only fix the syntax. Keep facts and rules exactly as they are."
            "\n\n"
            f"{broken_logic}"
        )
        return self.llm.query(prompt)


    def solve(self, question):
        # Step 1: Translate
        logic = self.logic_translate(question)

        print("\nGenerated Logic Output:\n")
        print(logic)

        # Step 2: Validate
        errors = check_logic_validity(logic)
        if errors:
            print("\nErrors Detected in Logic:")
            for error in errors:
                print(error)
            print("\nRefining Logic...\n")
            logic = self.refine_logic(logic, errors)

            print("\nRefined Logic Output:\n")
            print(logic)

        # Step 3: Solve
        solution = self.solver.solve_logic(logic)
        return solution
