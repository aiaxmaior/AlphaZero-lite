#mcts_user.py
# Rebuilt MCTS script

import random
import math
import chess
# import configs, not yet defined
# import encoding, not yet defined
# Classes to define (and why)
# Class Node
# Class MCTS
# Class BatchedMCTS

"""
This is a user generated version of the MCTS script. This is intended to implement learned knowledge into the MCTS / AlphaZero algorithm.
It is crucial that the user understand and implement this translation of theory to code properly.
chess.Board
chess.Move
Node
MCTS
evaluate(board)
Network, encoding
numpy / torch arrays
"""


class Node:
    def __init__(self, prior, to_play):
        self.prior = prior       # P(s,a): prior of the move that led here
        self.to_play = to_play   # whose turn AT this node - drives the sign-flip
        self.visit_count = 0     # N
        self.value_sum = 0.0     # W
        self.children = {}       # move -> Node (empty dictionary == unexpanded leaf)
    
    def value(self) -> float:
        # Q = W / N if N > 0 else 0
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def is_expanded(self) -> bool:
        return bool(self.children)

class MCTS:
    def __init__(self, evaluate, c_puct = 1.5, simulations = 400):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.simulations = simulations
    
    def run(self,board):
        root = Node(prior=1.0, to_play=board.turn)
        self._expand(root,board)
        for _ in range(self.simulations):
            self._simulate(root, board)

        # Harvest the answer: the visit-count distribution over the root's children.
        total = sum(child.visit_count for child in root.children.values())
        return {move: child.visit_count / total for move, child in root.children.items()}
    
        
        

    def _simulate(self, root, board):
        node, scratch, path = root, board.copy(), [root]
        # TO DO (you) 1. SELECT: while node.is_expanded(): pick child by _ucb, push, descend

        while node.is_expanded():
            action, child = max(node.children.items(), key=lambda pair: self._ucb(node,pair[1]))
            scratch.push(action)            
            node = child
            path.append(node)

        # 2. Expand _ evaluate
        if scratch.is_game_over(claim_draw=True):
            value= -1.0 if scratch.is_checkmate() else 0.00 # accounting for draw
        else:
            value = self._expand(node,scratch)        
        # 3. 
        self._backup(path,value)
        # 4.  Finish

    def _ucb(self, parent, child):
        # UCB1 (k,p) = E[win |k,p] + c*sqrt(2ln(n_parentk)/n_k).... E[win | k,p] ~= average reward
        # UCT: a* = argmax_a [  Q(s,a) + c_puct * sqrt(ln N(s) / N(s,a)) ]
        u = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        q = -child.value() if child.visit_count else 0.0

        return q + u # explore + exploit

    def _expand(self, node, board):
        priors, value = self.evaluate(board)
        for move, p in priors.items():
            node.children[move] = Node(prior=p, to_play=not board.turn)
        return value
    
    def _backup(self,path,value):
        # Flipping sign for each "node" depth?
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
