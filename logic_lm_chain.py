# logic_lm_chain.py

from langchain_llm import LangChainLLM
from logic_utils import LogicSolver, check_logic_validity
from retriever import create_retriever_from_kb
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

class LogicLMChain:
    def __init__(self, kb_path, model_name="gpt-3.5-turbo"):
        self.llm = LangChainLLM(model_name)
        self.solver = LogicSolver()
        self.retriever = create_retriever_from_kb(kb_path)
        self.kb_path = kb_path  # <--- Save the KB path
        self.prompt = PromptTemplate(
            input_variables=["context", "description"],
            template=(
                "You are given some background knowledge:\n\n{context}\n\n"
                "Translate the following description into general Prolog-style logic rules."
                " Only output general rules. Do NOT output specific facts, queries, or answers."
                " The rules should work for any entity, not just a particular example."
                " Use the convention that sibling(X, Y) means X is a sibling of Y, and sibling relationships are symmetric."
                " Make sure to define uncle(X, Y) as: uncle(X, Y) :- parent(Z, Y), sibling(X, Z)."
                " Ensure each rule ends with a period.\n\n"
                "{description}"
            )
        )

        self.chain = self.prompt | self.llm.llm

    def logic_translate(self, context, description):
        full_prompt = self.prompt.format(context=context, description=description)
        return self.llm.query(full_prompt)

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

    def load_kb_facts(self):
        with open(self.kb_path, 'r') as f:
            lines = f.readlines()
        facts = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('%') and ':-' not in line:  # Only facts, skip comments and rules
                facts.append(line)
        return "\n".join(facts)

    def solve(self, description):
        # Step 1: Retrieve relevant facts (optional for context)
        docs = self.retriever.invoke(description)
        context = "\n".join([doc.page_content for doc in docs])

        # Step 2: LLM logic generation
        logic_rule = self.logic_translate(context, description)

        print("\nGenerated Logic Output:\n")
        print(logic_rule)

        # Step 3: Validate logic
        errors = check_logic_validity(logic_rule)
        if errors:
            print("\nErrors Detected in Logic:")
            for error in errors:
                print(error)
            print("\nRefining Logic...\n")
            logic_rule = self.refine_logic(logic_rule, errors)
            print("\nRefined Logic Output:\n")
            print(logic_rule)

        # Step 4: Load full KB facts
        kb_facts = self.load_kb_facts()

        # Merge KB facts + LLM rule
        full_logic = kb_facts + "\n" + logic_rule

        # Step 5: Symbolic solving
        solution = self.solver.solve_logic(full_logic)
        return solution
