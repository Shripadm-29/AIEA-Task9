# utils/logic_utils.py

import re
from collections import defaultdict
from itertools import product

def check_logic_validity(logic_text):
    errors = []
    for line in logic_text.strip().split('\n'):
        stripped = line.strip()
        if not stripped.endswith('.'):
            errors.append("Missing period at end.")
        if '(' not in stripped or ')' not in stripped:
            errors.append("Missing parentheses.")
    return errors

def split_predicates(body_text):
    return [x.strip() for x in re.split(r',\s*(?![^()]*\))', body_text)]

class LogicSolver:
    def __init__(self):
        self.facts = []
        self.rules = []

    def parse_logic(self, logic_text):
        self.facts.clear()
        self.rules.clear()

        lines = logic_text.strip().split('\n')
        for line in lines:
            line = line.strip().strip('.')
            if not line:
                continue
            if ':-' in line:
                self.rules.append(line)
            else:
                if '(' in line and ')' in line:
                    predicate, args = line.split('(')
                    args = args.strip(')').split(',')
                    self.facts.append((predicate.strip(), [arg.strip() for arg in args]))
                else:
                    print(f"Skipping invalid line: {line}")

        # 🧠 Auto-add symmetric sibling facts
        new_sibling_facts = []
        for pred, args in self.facts:
            if pred == "sibling" and len(args) == 2:
                reversed_fact = ("sibling", [args[1], args[0]])
                if reversed_fact not in self.facts:
                    new_sibling_facts.append(reversed_fact)
        self.facts.extend(new_sibling_facts)

    def solve_logic(self, logic_text):
            self.parse_logic(logic_text)

            # Debug Print: Facts
            print("\nFacts:")
            for pred, args in self.facts:
                print(f"{pred}({', '.join(args)})")

            # Debug Print: Rules
            print("\nRules:")
            for rule in self.rules:
                print(rule)

            fact_dict = defaultdict(list)
            for predicate, args in self.facts:
                fact_dict[predicate].append(args)

            derived_facts = set()

            for rule in self.rules:
                head, body = rule.split(':-')
                head_predicate, head_args = head.strip().split('(')
                head_args = [h.strip() for h in head_args.strip(')').split(',')]

                body = body.strip()
                if body.startswith('(') and body.endswith(')'):
                    body = body[1:-1]

                body_predicates = split_predicates(body)
                body_preds = []
                for b in body_predicates:
                    pred_name, pred_args = b.split('(')
                    pred_args = [arg.strip() for arg in pred_args.strip(')').split(',')]
                    body_preds.append((pred_name.strip(), pred_args))

                fact_sets = []
                for pred_name, pred_vars in body_preds:
                    fact_sets.append(fact_dict.get(pred_name, []))

                for fact_combo in product(*fact_sets):
                    var_bindings = {}
                    match = True
                    for (pred_vars, fact_args) in zip([bp[1] for bp in body_preds], fact_combo):
                        for var, val in zip(pred_vars, fact_args):
                            if var in var_bindings:
                                if var_bindings[var] != val:
                                    match = False
                                    break
                            else:
                                var_bindings[var] = val
                        if not match:
                            break
                    if match:
                        result = tuple(var_bindings.get(var.strip(), '?') for var in head_args)

                        # 🛠️ NEW: Skip results where X == Y
                        if result[0] != result[1]:
                            derived_facts.add((head_predicate, result))

            return derived_facts
