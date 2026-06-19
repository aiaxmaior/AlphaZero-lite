"""Hands-on tour of python-chess.  Run me:  python user/chess_intro.py

This is NOT part of the engine — it's a sandbox so the `board` object stops
being mysterious. Everything MCTS will ever ask of the chess library is here.
"""
import chess

# ---------------------------------------------------------------------------
print("=" * 60)
print("1) A board is just an object you create:")
board = chess.Board()                 # standard starting position
print(board)                          # python-chess prints an ASCII board
print()

# ---------------------------------------------------------------------------
print("=" * 60)
print("2) Whose turn is it?  ->  board.turn")
print("   board.turn      =", board.turn)        # True
print("   chess.WHITE     =", chess.WHITE)        # True
print("   chess.BLACK     =", chess.BLACK)        # False
print("   so True == White to move, False == Black to move")
print()

# ---------------------------------------------------------------------------
print("=" * 60)
print("3) What moves are legal?  ->  board.legal_moves")
print("   how many:", board.legal_moves.count())
print("   first 5 :", [m.uci() for m in list(board.legal_moves)[:5]])
print("   (each one is a chess.Move object; .uci() is its text form)")
print()

# ---------------------------------------------------------------------------
print("=" * 60)
print("4) Making a move  ->  board.push(move)   (this MUTATES the board)")
e4 = chess.Move.from_uci("e2e4")
print("   is e2e4 legal? ", e4 in board.legal_moves)
print("   e2e4 in chess notation:", board.san(e4))
board.push(e4)
print("   after push, board.turn =", board.turn, "(now Black's move)")
print()

# ---------------------------------------------------------------------------
print("=" * 60)
print("5) Copying  ->  board.copy()   (MCTS descends on a COPY, not the real game)")
scratch = board.copy()
scratch.push(chess.Move.from_uci("e7e5"))
print("   advanced the scratch copy, but the real board is untouched:")
print("   real board still Black-to-move?", board.turn == chess.BLACK)
print()

# ---------------------------------------------------------------------------
print("=" * 60)
print("6) Is the game over?  ->  is_checkmate(), is_game_over(), result()")
mate = chess.Board()
for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:   # fool's mate
    mate.push(chess.Move.from_uci(uci))
print(mate)
print("   is_checkmate():", mate.is_checkmate())
print("   is_game_over():", mate.is_game_over())
print("   result()      :", mate.result(), " (0-1 == Black won)")
print("   board.turn at the mate:", mate.turn, "-> White is to move AND is mated,")
print("   which is exactly why a checkmate node's value is -1 for the side to move.")
print("=" * 60)
