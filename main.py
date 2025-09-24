from inference_graph import compiled_graph, InferenceState

def main():
    # Define the input question (what we want the system to solve)
    question = "Define family relationships like uncle, aunt, cousin, grandparent."
    
    # Create an initial state object with the question
    state = InferenceState(question)
    
    # Run the state through the compiled graph (workflow pipeline)
    final_state = compiled_graph.invoke(state)

    print("\nFinal Derived Facts:")
    
    # Check if the graph produced a final solution
    if final_state.get("final_solution"):
        # Clean the facts: remove any that contain '?' (unknown placeholders)
        clean_facts = [
            (pred, args) for pred, args in final_state["final_solution"]
            if '?' not in args
        ]
        # Deduplicate results and sort them alphabetically
        clean_facts = sorted(set(clean_facts))

        # Print each fact in standard Prolog style
        for pred, args in clean_facts:
            print(f"{pred}({', '.join(args)})")
    else:
        # If no solution, print fallback message
        print("No solution was found.")


if __name__ == "__main__":
    main()

