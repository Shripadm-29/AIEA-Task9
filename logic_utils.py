#logic_utils.py

import re
from collections import defaultdict
from itertools import product

# Check if Prolog-like logic rules have basic syntax issues
def check_logic_validity(logic_text):
    errors = []
    for line in logic_text.strip().split('\n'):   # Process each line
        stripped = line.strip()
        # Every rule/fact must end with a period
        if not stripped.endswith('.'):
            errors.append("Missing period at end.")
        # Must have parentheses to look like a fact or rule
        if '(' not in stripped or ')' not in stripped:
            errors.append("Missing parentheses.")
    return errors


# Split a rule's body into separate predicates
# Example: "parent(X,Y), sibling(X,Z)" → ["parent(X,Y)", "sibling(X,Z)"]
# Regex makes sure commas inside (...) aren’t split
def split_predicates(body_text):
    return [x.strip() for x in re.split(r',\s*(?![^()]*\))', body_text)]


# Core logic solver
class LogicSolver:
    def __init__(self):
        self.facts = []  # List of parsed facts (predicate, args)
        self.rules = []  # List of parsed rules (raw strings)

    # Parse logic text into facts + rules
    def parse_logic(self, logic_text):
        # Reset stored facts and rules
        self.facts.clear()
        self.rules.clear()

        # Process each line of logic text
        lines = logic_text.strip().split('\n')
        for line in lines:
            line = line.strip().strip('.')  # Remove whitespace + trailing "."
            if not line:
                continue  # Skip empty lines
            if ':-' in line:
                # Rules go into self.rules
                self.rules.append(line)
            else:
                # Facts → parse predicate + args
                if '(' in line and ')' in line:
                    predicate, args = line.split('(')
                    args = args.strip(')').split(',')
                    self.facts.append((predicate.strip(), [arg.strip() for arg in args]))
                else:
                    # Skip garbage lines
                    print(f"Skipping invalid line: {line}")

        # Auto-add symmetric sibling facts
        # If sibling(A, B) exists, also add sibling(B, A)
        new_sibling_facts = []
        for pred, args in self.facts:
            if pred == "sibling" and len(args) == 2:
                reversed_fact = ("sibling", [args[1], args[0]])
                if reversed_fact not in self.facts:
                    new_sibling_facts.append(reversed_fact)
        self.facts.extend(new_sibling_facts)

    # Try to solve the logic program
    def solve_logic(self, logic_text):
        # Parse facts and rules
        self.parse_logic(logic_text)

        # Debug Print: Facts
        print("\nFacts:")
        for pred, args in self.facts:
            print(f"{pred}({', '.join(args)})")

        # Debug Print: Rules
        print("\nRules:")
        for rule in self.rules:
            print(rule)

        # Put facts into dictionary: predicate → list of arg-lists
        fact_dict = defaultdict(list)
        for predicate, args in self.facts:
            fact_dict[predicate].append(args)

        derived_facts = set()

        # Try to apply each rule
        for rule in self.rules:
            # Split into head and body
            head, body = rule.split(':-')
            head_predicate, head_args = head.strip().split('(')
            head_args = [h.strip() for h in head_args.strip(')').split(',')]

            # Clean body (remove wrapping parentheses if any)
            body = body.strip()
            if body.startswith('(') and body.endswith(')'):
                body = body[1:-1]

            # Split body into predicates
            body_predicates = split_predicates(body)
            body_preds = []
            for b in body_predicates:
                pred_name, pred_args = b.split('(')
                pred_args = [arg.strip() for arg in pred_args.strip(')').split(',')]
                body_preds.append((pred_name.strip(), pred_args))

            # For each predicate in body, get all matching facts
            fact_sets = []
            for pred_name, pred_vars in body_preds:
                fact_sets.append(fact_dict.get(pred_name, []))

            # Try all combinations of facts from the body predicates
            for fact_combo in product(*fact_sets):
                var_bindings = {}  # Map variables → concrete values
                match = True

                # Check if variables can be consistently bound
                for (pred_vars, fact_args) in zip([bp[1] for bp in body_preds], fact_combo):
                    for var, val in zip(pred_vars, fact_args):
                        if var in var_bindings:
                            if var_bindings[var] != val:
                                match = False  # Conflict in variable assignment
                                break
                        else:
                            var_bindings[var] = val
                    if not match:
                        break

                if match:
                    # Build result for head using variable bindings
                    result = tuple(var_bindings.get(var.strip(), '?') for var in head_args)

                    # 🛠️ NEW: Skip nonsense results where X == Y
                    if result[0] != result[1]:
                        derived_facts.add((head_predicate, result))

        return derived_facts
