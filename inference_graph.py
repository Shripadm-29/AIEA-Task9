# inference_graph.py

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from logic_utils import LogicSolver
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class InferenceState:
    question: str
    relevant_facts: Optional[str] = None
    logic_rule: Optional[str] = None
    final_solution: Optional[List] = None

def load_kb_node(state):
    with open("kb.txt", "r") as f:
        kb_facts = f.read()
    state.relevant_facts = kb_facts
    return state

def logic_generate_node(state):
    prompt = PromptTemplate(
        input_variables=["context", "description"],
        template=(
            "You are given some background knowledge in Prolog format:\n\n{context}\n\n"
            "Translate the following description into Prolog-style logic rules."
            " Only output logic rules. No explanations, no extra text."
            " Each rule must end with a period.\n\n"
            "{description}"
        )
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    full_prompt = prompt.format(
        context=state.relevant_facts,
        description="""Define the following relations:
- Uncle: uncle(X, Y) :- parent(Z, Y), sibling(X, Z).
- Aunt: aunt(X, Y) :- parent(Z, Y), sibling(X, Z).
- Cousin: cousin(X, Y) :- parent(Z, X), parent(W, Y), sibling(Z, W).
- Grandparent: grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
- Great-grandparent: greatgrandparent(X, Y) :- parent(X, Z), parent(Z, W), parent(W, Y)."""
    )

    response = llm.invoke(full_prompt)
    state.logic_rule = response.content  # ✅ Extract text content
    return state

def solve_node(state):
    logic_solver = LogicSolver()
    full_logic = state.relevant_facts + "\n" + state.logic_rule
    solution = logic_solver.solve_logic(full_logic)
    state.final_solution = solution
    return state

graph = StateGraph(InferenceState)
graph.add_node("LoadKB", load_kb_node)
graph.add_node("GenerateLogic", logic_generate_node)
graph.add_node("Solve", solve_node)

graph.set_entry_point("LoadKB")
graph.add_edge("LoadKB", "GenerateLogic")
graph.add_edge("GenerateLogic", "Solve")

compiled_graph = graph.compile()
