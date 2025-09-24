# logic_lm_chain.py

from langchain_llm import LangChainLLM
from logic_utils import LogicSolver, check_logic_validity
from retriever import create_retriever_from_kb
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# LogicLMChain = LLM + KB retriever + LogicSolver pipeline
class LogicLMChain:
    def __init__(self, kb_path, model_name="gpt-3.5-turbo"):
        # Initialize LLM wrapper
        self.llm = LangChainLLM(model_name)
        # Initialize symbolic solver
        self.solver = LogicSolver()
        # Build retriever over the knowledge base (KB)
        self.retriever = create_retriever_from_kb(kb_path)
        # Save path to KB file
        self.kb_path = kb_path  

        # Define the base prompt template for generating rules
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

        # Create a simple chain: prompt → LLM
        self.chain = self.prompt | self.llm.llm

    # Step 1: Translate description + context into logic rules
    def logic_translate(self, context, description):
        # Fill the prompt with retrieved facts + description
        full_prompt = self.prompt.format(context=context, description=description)
        # Run it through the LLM
        return self.llm.query(full_prompt)

    # Step 2 (if errors): Refine broken logic rules
    def refine_logic(self, broken_logic, errors):
        # Combine error messages
        error_message = "\n".join(errors)
        # Ask the LLM to only fix syntax errors
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

    # Utility: Load pure KB facts (ignores comments and rules)
    def load_kb_facts(self):
        with open(self.kb_path, 'r') as f:
            lines = f.readlines()
        facts = []
        for line in lines:
            line = line.strip()
            # Keep only plain facts (skip comments "%" and rules ":-")
            if line and not line.startswith('%') and ':-' not in line:
                facts.append(line)
        # Join into one big string
        return "\n".join(facts)

    # Step 3: Full pipeline = Retrieve → Generate → Validate → Solve
    def solve(self, description):
        # Step 1: Retrieve relevant KB snippets (context for LLM)
        docs = self.retriever.invoke(description)
        context = "\n".join([doc.page_content for doc in docs])

        # Step 2: Generate logic rules from description + context
        logic_rule = self.logic_translate(context, description)

        print("\nGenerated Logic Output:\n")
        print(logic_rule)

        # Step 3: Validate syntax
        errors = check_logic_validity(logic_rule)
        if errors:
            print("\nErrors Detected in Logic:")
            for error in errors:
                print(error)
            print("\nRefining Logic...\n")
            # Ask LLM to fix rules
            logic_rule = self.refine_logic(logic_rule, errors)
            print("\nRefined Logic Output:\n")
            print(logic_rule)

        # Step 4: Load complete KB facts
        kb_facts = self.load_kb_facts()

        # Merge KB facts with generated logic
        full_logic = kb_facts + "\n" + logic_rule

        # Step 5: Solve with symbolic LogicSolver
        solution = self.solver.solve_logic(full_logic)
        return solution
