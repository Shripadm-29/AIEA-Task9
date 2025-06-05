# inference_graph.py (UPGRADED with CoT + Self-Refinement)

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from logic_utils import LogicSolver, check_logic_validity
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class InferenceState:
    question: str
    relevant_facts: Optional[str] = None
    logic_rule: Optional[str] = None
    final_solution: Optional[List] = None
    retry_count: int = 0  # Track number of refinements
    errors: Optional[List[str]] = None  # To store syntax errors
    _next: Optional[str] = None  # To store next node decision

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
    state.logic_rule = response.content
    return state

def check_validity_node(state):
    errors = check_logic_validity(state.logic_rule)
    if not errors:
        state._next = "Solve"
    else:
        state.errors = errors
        state._next = "SelfRefine"
    return state

def self_refine_node(state):
    if state.retry_count >= 3:
        print("❌ Maximum refinement attempts reached.")
        return state

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(
        input_variables=["broken_logic", "errors"],
        template=(
            "The following Prolog logic has syntax errors:\n\n{errors}\n\n"
            "Please fix the syntax errors while keeping all the original facts and rules.\n"
            "Do NOT add, remove, or answer anything.\n\n"
            "{broken_logic}"
        )
    )
    full_prompt = prompt.format(
        broken_logic=state.logic_rule,
        errors="\n".join(state.errors)
    )

    response = llm.invoke(full_prompt)
    state.logic_rule = response.content
    state.retry_count += 1
    return state

def solve_node(state):
    logic_solver = LogicSolver()
    full_logic = state.relevant_facts + "\n" + state.logic_rule
    solution = logic_solver.solve_logic(full_logic)
    state.final_solution = solution
    return state

# Define the graph
graph = StateGraph(InferenceState)
graph.add_node("LoadKB", load_kb_node)
graph.add_node("GenerateLogic", logic_generate_node)
graph.add_node("CheckValidity", check_validity_node)
graph.add_node("SelfRefine", self_refine_node)
graph.add_node("Solve", solve_node)

# Define the flow
graph.set_entry_point("LoadKB")
graph.add_edge("LoadKB", "GenerateLogic")
graph.add_edge("GenerateLogic", "CheckValidity")
graph.add_conditional_edges(
    "CheckValidity",
    lambda state: state._next
)

graph.add_edge("SelfRefine", "CheckValidity")

compiled_graph = graph.compile()
