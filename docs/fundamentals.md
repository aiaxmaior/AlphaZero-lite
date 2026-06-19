# AlphaZero from the fundamentals

A from-the-theory walkthrough for someone who already knows the RL objective and
the policy-gradient theorem. Build order: **concept → pseudocode → study the code
→ rebuild from blank files.**

---

## 1. The reframe: AlphaZero is policy *iteration*, not policy *gradient*

GAE (Not related to GPI):
$$
\hat{A}^{\text{GAE}}t = \sum{l\ge 0}(\gamma\lambda)^l \delta_{t+l},
\qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$


You know the policy-gradient theorem:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a\mid s)\,\Psi(s,a)\right],
\qquad \Psi \in \{\,G_t,\; A^\pi,\; Q^\pi,\;\dots\}
$$

**AlphaZero does not use this.** There is no $\nabla\log\pi \cdot \text{return}$
anywhere in it. If you go looking for REINFORCE / advantage in the code you will
be confused.

Instead it is **Generalized Policy Iteration (GPI)**, where the *improvement*
step is a tree search rather than a gradient-ascent step. The cleanest mental
model is **Expert Iteration** (Anthony et al., 2017):

- a slow **expert** $=$ MCTS (searches ahead, strong but expensive)
- a fast **apprentice** $=$ the network (one forward pass, cheap)
- the apprentice is trained to **imitate** the expert; a better apprentice makes
  the next expert stronger; repeat.

The loop, in GPI terms:

| GPI stage              | Classical RL                     | AlphaZero                                          |
|------------------------|----------------------------------|----------------------------------------------------|
| Policy **evaluation**  | $V^\pi$ via Bellman / TD backups | value head $v_\theta$ regressed onto outcomes $z$  |
| Policy **improvement** | act greedily w.r.t. $Q$          | **MCTS**: run sims with $(p_\theta, v_\theta)\to\boldsymbol\pi$ |
| Policy **representation** | table / params                | network $p_\theta$                                 |
| **Closing the loop**   | (implicit)                       | **distill** $\boldsymbol\pi$ back into $p_\theta$ (supervised fit) |

### Why search counts as "improvement"

The claim that makes this valid GPI: running search yields a policy
$\boldsymbol\pi_{\text{MCTS}}$ that is **at least as good as the bare prior**
$p_\theta$.

- Asymptotically, PUCT/UCT concentrates visits on the minimax-optimal move.
- Even at finite simulation counts, the visit distribution sharpens toward
  high-value moves relative to the prior.

So $\boldsymbol\pi_{\text{MCTS}} \succeq p_\theta$ (in policy-value order), and
training $p_\theta \to \boldsymbol\pi_{\text{MCTS}}$ is a **projection** of the
improved policy back onto your function class. Iterate, and you get the
monotone-ish improvement of GPI.

---

## 2. The objective and its gradients (this part *is* a clean gradient theorem)

Once search has produced targets, training is **supervised**. For each visited
position you store a triple $(s,\ \boldsymbol\pi,\ z)$:

- $s$ — board state
- $\boldsymbol\pi$ — MCTS visit-count distribution at $s$ (the improved policy)
- $z \in \{-1, 0, +1\}$ — final game outcome, **from the perspective of the
  player to move at $s$**

Minimize:

$$
\mathcal{L}(\theta) \;=\;
\underbrace{\big(z - v_\theta(s)\big)^2}_{\text{value, MSE}}
\;-\;
\underbrace{\boldsymbol\pi^\top \log p_\theta(s)}_{\text{policy, cross-entropy}}
\;+\;
\underbrace{c\,\lVert\theta\rVert^2}_{\text{L2 reg}}
$$

Both gradients are textbook. Derive them once by hand so they're yours.

### Policy gradient (softmax + cross-entropy)

With logits $\ell$, predicted $p=\mathrm{softmax}(\ell)$, target
$\boldsymbol\pi$, and $\mathcal{L}_p = -\sum_a \pi_a \log p_a$:

$$
\boxed{\;\frac{\partial \mathcal{L}_p}{\partial \ell_a} \;=\; p_a - \pi_a\;}
\qquad\text{(``predicted minus target'')}
$$

That's the whole thing. Compare with REINFORCE, where the update is
$(\text{return})\cdot\nabla\log\pi$ for a *single sampled* action and is
high-variance. Here you regress against the **entire** search distribution. The
credit assignment that policy gradient gets from noisy Monte-Carlo returns,
AlphaZero gets from lookahead instead. That is the variance-reduction trade.

### Value gradient (tanh + MSE)

With $v=\tanh(u)$ (so $u$ is the pre-activation) and $\mathcal{L}_v=(v-z)^2$:

$$
\frac{\partial \mathcal{L}_v}{\partial u} \;=\; 2\,(v - z)\,(1 - v^2)
$$

This is why the loss in code is literally
$\text{cross\_entropy}(\text{soft target}) + \text{mse}$: no advantages, no
importance weights, no $\log\pi \cdot \text{return}$.

---

## 3. Pseudocode — the four pieces, in build order

### (1) Encoding

```
encode(s):                       # s = board, ALWAYS from side-to-move's view
    planes = zeros(C, 8, 8)
    for piece on board:
        planes[plane(piece), mirror_if_black(square)] = 1
    return planes

# fixed action space (AlphaZero 8x8x73 = 4672):
#   index(move) = from_square * 73 + move_type_plane
#   73 planes = 56 queen-moves (8 dirs x 7 dist) + 8 knight + 9 underpromotions
```

### (2) Network

$$
f_\theta(s) \;\to\; (p,\ v),\qquad
p \in \Delta^{4671}\ \text{(masked legal)},\quad v \in [-1, 1]
$$

where $p$ is a distribution over the fixed action space and $v$ is the expected
outcome for the side to move.

### (3) MCTS — the improvement operator

The selection rule is **PUCT**: at a node, pick

$$
a^\star \;=\; \arg\max_a \left[\, Q(s,a) \;+\; c_{\text{puct}}\,P(s,a)\,
\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \,\right]
$$

where $Q(s,a)$ is the mean backed-up value through $a$ (parent-mover's
perspective), $P(s,a)$ is the prior from $p_\theta$, and $N$ is the visit count.

```
search(s0, n_sims):
    root = expand(s0)                         # priors from p_theta, legal-masked
    add_dirichlet_noise(root)                 # exploration at the root only

    repeat n_sims times:
        node, path, s = root, [root], copy(s0)

        # --- SELECT: descend by PUCT until a leaf ---
        while node.expanded:
            a = argmax over actions of PUCT(node, a)     # formula above
            s.push(a)
            node = node.child[a]
            path.append(node)

        # --- EXPAND + EVALUATE at the leaf ---
        if terminal(s):
            v = -1 if checkmate else 0        # MOVER'S perspective (just got mated)
        else:
            p, v = f_theta(s)                 # value head replaces random rollout
            create children of node with priors p

        # --- BACKUP: walk up, flip sign each ply (zero-sum) ---
        for node in reverse(path):
            node.N += 1
            node.W += v
            v = -v

    return visit_counts(root) / sum(visit_counts(root))    # this is pi
```

The returned policy is the normalized visit count,
$\pi(a) = N(s_0, a) / \sum_b N(s_0, b)$.

### (4) Self-play + train — the GPI loop closing on itself

```
selfplay_game():
    examples = []
    while not game_over:
        pi = search(s, n_sims)
        examples.append( (encode(s), pi, side_to_move) )

        # explore early, exploit late
        a = sample(pi, temperature=1)  if ply < N  else argmax(pi)
        s.push(a)

    z = outcome(s)                            # +1 white win, -1 black win, 0 draw
    for ex in examples:                       # convert to each mover's view
        ex.target_z = +z if ex.mover_is_white else -z
    return examples

train_loop():
    repeat:
        buffer += [ selfplay_game() for _ in range(games_per_iter) ]
        for batch in buffer:
            logits, v = f_theta(batch.states)
            loss =   mse(v, batch.z)
                   - (batch.pi * log_softmax(logits)).sum(axis=1).mean()
            theta -= lr * grad(loss)
        checkpoint()
        evaluate_vs_stockfish()               # is it actually getting stronger?
```

### The two sign conventions people get wrong

1. **Backup sign flip.** A value good for the mover is bad for the opponent one
   ply up. Forget the flip and $Q$ is meaningless.
2. **Terminal frame.** Terminal values are in the *mover-at-leaf* frame
   ($\text{checkmate} = -1$, because the side to move just got mated) — the
   **same** frame the value head outputs. Backup is only consistent if both heads
   and the terminal case agree on the frame. Mixing an absolute (White-positive)
   terminal value with a side-to-move value head is a real, subtle bug.

---

## 4. Study the reference code in this order

1. `src/encoding.py` — `encode_board`, `move_to_index` / `index_to_move`.
   Run `python -m src.encoding --selftest`. Convince yourself the 4672-action
   space is a bijection on legal moves.
2. `src/network.py` — `ChessNet.forward` (two heads) and `policy_value_loss`
   (matches section 2 exactly).
3. `src/mcts.py` — `_ucb_score` (PUCT), `run` (select/expand/backup),
   `_evaluate_terminal` (the sign convention). This is the conceptual core.
4. `src/selfplay.py` `play_game` → `src/train.py` `train_on_buffer` — the GPI
   loop closing on itself.

---

## 5. Rebuild from blank files — with a falsifiable check after each step

Same order, but each step has a test that must pass before you move on, so you're
never debugging blind:

1. **Encoding** — your own round-trip test passes on thousands of random
   positions. (Reference checks 59,264 legal moves.)
2. **Network** — shape test: $(B,17,8,8) \to (B,4672),\ (B,)$. Plus hand-verify
   $\partial\mathcal{L}_p/\partial\ell = p - \pi$ on a tiny example.
3. **MCTS** — on a forced mate-in-1, a *random* network with enough simulations
   must still find the mate. This single test catches almost every sign / PUCT
   bug.
4. **Loop** — `--vs random` climbs to ~100%, then beat Stockfish level 1.

---

## References

- Silver et al., *Mastering Chess and Shogi by Self-Play* (AlphaZero), 2017.
- Silver et al., *Mastering the game of Go without human knowledge*
  (AlphaGo Zero), 2017 — the clearest statement of MCTS-as-policy-improvement.
- Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree
  Search* (Expert Iteration), 2017 — the expert/apprentice framing.
