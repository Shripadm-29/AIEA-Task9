from inference_graph import compiled_graph, InferenceState

def main():
    question = "Define family relationships like uncle, aunt, cousin, grandparent."
    state = InferenceState(question)
    final_state = compiled_graph.invoke(state)

    print("\nFinal Derived Facts:")
    if final_state.get("final_solution"):
        # Filter out facts with '?' in args
        clean_facts = [
            (pred, args) for pred, args in final_state["final_solution"]
            if '?' not in args
        ]
        # Deduplicate and sort
        clean_facts = sorted(set(clean_facts))

        for pred, args in clean_facts:
            print(f"{pred}({', '.join(args)})")
    else:
        print("No solution was found.")

if __name__ == "__main__":
    main()
