"""What does max(..., key=...) mean?   Run:  python user/max_key_demo.py

Pure Python — no chess, no MCTS. Just demystifying the one line you need.
"""

# 1) Plain max compares the items directly.
print("1) max of numbers:", max([3, 1, 4, 1, 5]))            # -> 5

# 2) max with a KEY: it applies a function to each item, compares THOSE values,
#    and returns the original ITEM that scored highest.
words = ["cat", "elephant", "dog"]
print("2) longest word:", max(words, key=len))               # -> 'elephant'
print("   len('cat')=3, len('elephant')=8, len('dog')=3")
print("   -> it returns the WORD 'elephant', NOT the number 8")

# 3) A lambda is just a tiny inline function with no name.
#       key=lambda w: len(w)
#    is exactly the same as defining:
#       def f(w): return len(w)
#    and passing f.
print("3) same answer via lambda:", max(words, key=lambda w: len(w)))

# 4) A dict's .items() gives you (key, value) PAIRS.
scores = {"a": 10, "b": 30, "c": 20}
print("4) items():", list(scores.items()))                   # [('a',10),('b',30),('c',20)]
print("   for a pair like ('b', 30): pair[0] is 'b', pair[1] is 30")

# 5) Now: pick the PAIR whose number (pair[1]) is largest.
best = max(scores.items(), key=lambda pair: pair[1])
print("5) best pair:", best)                                 # -> ('b', 30)
letter, number = best                                        # unpack the winning pair
print("   unpacked -> letter:", letter, "number:", number)
