"""Turning visit counts into a policy.   Run:  python user/policy_demo.py

Pure dict mechanics, no MCTS. Goal: go from {move: count} to {move: probability}
where the probabilities sum to 1. That's all `run` has to do at the end.
"""

# ----- Warm-up: when the counts are plain numbers -----
visits = {"e4": 60, "d4": 30, "Nf3": 10}
print("raw visit counts:", visits)

# Step 1 — total up all the counts.
total = 0
for count in visits.values():          # .values() -> 60, 30, 10
    total += count
print("total visits:", total)          # 100
# compact form:  total = sum(visits.values())

# Step 2 — each count divided by the total = its share = a probability.
policy = {}
for move, count in visits.items():     # .items() -> ('e4', 60), ('d4', 30), ...
    policy[move] = count / total
print("policy:", policy)               # {'e4': 0.6, 'd4': 0.3, 'Nf3': 0.1}
print("sums to:", sum(policy.values()))   # 1.0
# compact form:  policy = {move: count / total for move, count in visits.items()}


# ----- The REAL shape: the dict value is a NODE, and the count is node.visit_count -----
class FakeNode:
    def __init__(self, visit_count):
        self.visit_count = visit_count

children = {"e4": FakeNode(60), "d4": FakeNode(30), "Nf3": FakeNode(10)}

# Only difference from above: read .visit_count off each child instead of using it directly.
total = sum(child.visit_count for child in children.values())
policy = {move: child.visit_count / total for move, child in children.items()}
print()
print("real-shape policy:", policy)    # same numbers, but pulled from Node objects
print("sums to:", sum(policy.values()))
