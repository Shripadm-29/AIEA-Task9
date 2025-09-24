#logic_lm.py

from langchain_llm import LangChainLLM
from logic_utils import LogicSolver, check_logic_validity

# Define a model that uses an LLM + logic solver
class LogicLMModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Initialize the LLM helper (for generating/refining logic)
        self.llm = LangChainLLM(model_name)
        # Initialize the custom solver (for executing logic rules)
        self.solver = LogicSolver()

    # Step 1: Translate natural language into Prolog-style logic
    def logic_translate(self, question):
        # Prompt tells the LLM exactly how to format the output
        prompt = (
            "Translate the following description into symbolic Prolog-style logic facts and rules."
            " Only output facts and rules."
            " Do NOT include any queries, answers, or explanations."
            " Ensure correct syntax with a period at the end of each fact and rule."
            " Do NOT use markdown formatting or code blocks."
            "\n\n"
            f"{question}"
        )
        # Send prompt to LLM and return result
        return self.llm.query(prompt)

    # Step 2 (if needed): Refine logic if errors are found
    def refine_logic(self, broken_logic, errors):
        # Combine all error messages into one string
        error_message = "\n".join(errors)
        # Prompt tells the LLM to ONLY fix syntax, nothing else
        prompt = (
            f"The following Prolog-like logic contains syntax errors:\n"
            f"{error_message}\n\n"
            "Please correct only the syntax errors while keeping ALL the original facts and rules."
            " Do NOT add, remove, or answer anything."
            " Only fix the syntax. Keep facts and rules exactly as they are."
            "\n\n"
            f"{broken_logic}"
        )
        # Return refined logic from LLM
        return self.llm.query(prompt)

    # Step 3: Full pipeline — translate, validate, refine, solve
    def solve(self, question):
        # Step 1: Translate natural language to logic
        logic = self.logic_translate(question)

        print("\nGenerated Logic Output:\n")
        print(logic)

        # Step 2: Validate syntax of the generated logic
        errors = check_logic_validity(logic)
        if errors:
            print("\nErrors Detected in Logic:")
            for error in errors:
                print(error)   # Print each error for debugging
            print("\nRefining Logic...\n")

            # Ask the LLM to fix the logic
            logic = self.refine_logic(logic, errors)

            print("\nRefined Logic Output:\n")
            print(logic)

        # Step 3: Solve the corrected logic with LogicSolver
        solution = self.solver.solve_logic(logic)
        return solution
