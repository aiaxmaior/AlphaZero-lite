# UCB1 → UCT → PUCT: the selection rule, derived

The whole point of these formulas is **one decision**: standing at a node, which
child move do I explore next? Everything below is about balancing "the move that
looks best so far" against "the move I haven't tried enough to trust."

---

## 1. The setting: a multi-armed bandit

Forget trees for a second. You have $K$ slot machines (arms). Arm $a$ pays a
random reward in $[0,1]$ with unknown mean $\mu_a$. You pull arms one at a time;
after pulling arm $a$ a total of $n_a$ times you have an empirical mean
$\bar{x}_a$. You want to maximize total reward, i.e. minimize **regret** vs.
always pulling the best arm.

The tension:

- **Exploit:** pull the arm with the highest $\bar{x}_a$ so far.
- **Explore:** pull an arm you've sampled too few times — your $\bar{x}_a$ might
  be wrong because $n_a$ is tiny.

You need a principled way to trade these off. That principle is **optimism in
the face of uncertainty**: act as if each arm is as good as its statistics
*plausibly* allow, then pull the best of those optimistic estimates.

---

## 2. UCB1, and where $\sqrt{\ln t / n_a}$ comes from

How wrong can $\bar{x}_a$ be? **Hoeffding's inequality** bounds it: for an arm
pulled $n_a$ times,

$$
\Pr\!\big(\mu_a > \bar{x}_a + \varepsilon\big) \;\le\; e^{-2 n_a \varepsilon^2}.
$$

So with high probability the *true* mean is no more than $\bar{x}_a + \varepsilon$.
That $\varepsilon$ is the **width of a confidence interval** on the mean. Solve
for the $\varepsilon$ that keeps the failure probability small as the game goes
on: set $e^{-2 n_a \varepsilon^2} = t^{-4}$ (where $t$ = total pulls so far across
all arms), which gives

$$
\varepsilon_a \;=\; \sqrt{\frac{2\ln t}{n_a}}.
$$

UCB1 then always pulls the arm with the largest **upper confidence bound**:

$$
\boxed{\;a^\star = \arg\max_a \left[\, \underbrace{\bar{x}_a}_{\text{exploit}} \;+\; c\,\underbrace{\sqrt{\frac{\ln t}{n_a}}}_{\text{explore (CI width)}} \,\right]\;}
$$

(the constant $c$ absorbs the $2$ and the reward scale). Auer et al. (2002)
proved this gives **logarithmic regret** $O(\ln T)$ — the best possible.

### Reading the exploration term piece by piece

This is the part that trips people up. Take it slowly:

$$
\sqrt{\dfrac{\ln t}{n_a}}
$$

- **$n_a$ in the denominator** (= $N(s,a)$ in a tree): the number of times you've
  pulled *this* arm. The more you've tried arm $a$, the more you trust
  $\bar{x}_a$, so the bonus **shrinks**. An arm pulled once is suspicious; an arm
  pulled 1000 times is well-known. The bonus $\propto 1/\sqrt{n_a}$ — exactly how
  fast a sample mean's uncertainty shrinks (standard error $\sim 1/\sqrt{n}$).
- **$t$ in the numerator** (= $N(s)$, the parent's total pulls): the total number
  of decisions made *at this node*. As you make more decisions overall, you want
  the confidence bounds to be more reliable (a union bound over more rounds), so
  the interval **widens slightly** to avoid permanently starving an arm that had
  an early run of bad luck.
- **Why $\ln t$ and not $t$?** If the numerator were $t$, exploration would never
  die down and you'd waste pulls forever (linear regret). $\ln$ grows *just*
  fast enough to keep re-checking neglected arms, but slowly enough that total
  exploration stays $O(\ln T)$. It's the sweet spot the regret proof needs.
- **Why the square root?** Because Hoeffding's bound puts $\varepsilon^2$ in the
  exponent — solving for $\varepsilon$ takes a square root. It's literally the
  width of a confidence interval, not an arbitrary choice.

So the whole term reads: *"the radius of the plausible-upside region for arm $a$,
which is wide when I've barely tried it and slowly grows as the node gets busier."*

### A concrete numeric feel

Parent visited $N(s) = 100$ times, so $\ln N(s) \approx 4.6$:

| arm tried $n_a$ times | exploration bonus $\sqrt{\ln 100 / n_a}$ |
|---|---|
| $n_a = 1$   | $\sqrt{4.6/1} \approx 2.15$ |
| $n_a = 5$   | $\sqrt{4.6/5} \approx 0.96$ |
| $n_a = 50$  | $\sqrt{4.6/50} \approx 0.30$ |

The barely-tried arm gets a **7×** larger exploration bonus than the well-tried
one. That gap is what forces the search to revisit neglected moves.

---

## 3. UCT = UCB1 applied at every tree node

Kocsis & Szepesvári (2006): treat **each node of the search tree as its own
bandit**, where the "arms" are the legal moves and the "reward" of a move is the
value backed up through it. Selection descends the tree by applying UCB1 at each
node:

$$
a^\star = \arg\max_a \left[\, Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \,\right]
$$

The only renaming from UCB1:

| bandit symbol | UCT symbol | meaning |
|---|---|---|
| $\bar{x}_a$ | $Q(s,a)$ | mean value backed up through move $a$ |
| $t$ (total pulls) | $N(s)$ | visits to the parent node $s$ |
| $n_a$ | $N(s,a)$ | visits to child move $a$ |

They proved UCT converges to the **minimax** value as $N(s)\to\infty$. So UCT is
just "UCB1, recursively, all the way down the tree."

---

## 4. PUCT: what AlphaZero uses instead, and why

AlphaZero replaces the UCT bonus with the **PUCT** rule (Rosin 2011, "Predictor"
UCB):

$$
\boxed{\;a^\star = \arg\max_a \left[\, Q(s,a) + c_{\text{puct}}\,P(s,a)\,
\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \,\right]\;}
$$

Compare the two exploration terms directly:

$$
\text{UCT:}\;\; \sqrt{\frac{\ln N(s)}{N(s,a)}}
\qquad\qquad
\text{PUCT:}\;\; P(s,a)\,\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}
$$

Three deliberate changes, each with a reason:

1. **Multiply by the prior $P(s,a)$.** Chess has ~35 legal moves per position;
   UCT's uniform exploration would waste the budget trying obvious blunders. The
   policy head's prior $P(s,a)$ focuses exploration on a-priori reasonable moves.
   This is the entire reason a learned network helps the search. (Note
   $\sum_b N(s,b) = N(s)$, the parent visits — same quantity as UCT's $t$.)

2. **$1 + N(s,a)$ in the denominator instead of $N(s,a)$.** The $+1$ means an
   **unvisited** child ($N(s,a)=0$) gets a finite, prior-weighted bonus
   $c_{\text{puct}}\,P(s,a)\sqrt{N(s)}$ instead of dividing by zero. UCT needs a
   special "try every child once first" rule; PUCT doesn't.

3. **Drop the $\ln$ and the inner square root.** PUCT uses $\sqrt{N(s)}$ in the
   numerator and $N(s,a)$ (linear, not $\sqrt{\cdot}$) in the denominator, so the
   bonus decays like $1/N(s,a)$ — **faster** than UCT's $1/\sqrt{N(s,a)}$. The
   search stops exploring a move and starts trusting its $Q$ sooner. That's
   appropriate here because the value head gives a usable estimate immediately,
   so you don't need UCT's slower, more cautious exploration schedule. PUCT is a
   well-tested **heuristic**, not a tight regret bound like UCB1.

### PUCT numeric feel

Parent visits $\sum_b N(s,b) = 100$, $c_{\text{puct}} = 1.5$, prior $P = 0.2$:

| child visits $N(s,a)$ | exploration bonus $1.5 \cdot 0.2 \cdot \sqrt{100}/(1+N(s,a))$ |
|---|---|
| $N(s,a) = 0$  | $1.5 \cdot 0.2 \cdot 10 / 1 = 3.00$ |
| $N(s,a) = 1$  | $\;\;= 1.50$ |
| $N(s,a) = 50$ | $\;\;\approx 0.059$ |

Early ($N$ small) the bonus dwarfs $Q$, so the search follows the prior; once a
move is well-visited the bonus collapses and $Q$ decides. $c_{\text{puct}}$ sets
where that crossover happens.

---

## 5. Side-by-side summary

| | UCB1 (bandit) | UCT (tree) | PUCT (AlphaZero) |
|---|---|---|---|
| value term | $\bar{x}_a$ | $Q(s,a)$ | $Q(s,a)$ |
| exploration | $\sqrt{\ln t / n_a}$ | $\sqrt{\ln N(s)/N(s,a)}$ | $P(s,a)\,\sqrt{\sum_b N(s,b)}/(1{+}N(s,a))$ |
| uses a prior? | no | no | **yes** ($P$ from policy head) |
| unvisited arm | pull once first | pull once first | finite bonus via $+1$ |
| bonus decay in $n_a$ | $1/\sqrt{n_a}$ | $1/\sqrt{N(s,a)}$ | $1/N(s,a)$ (faster) |
| guarantee | $O(\ln T)$ regret | converges to minimax | heuristic, empirically strong |

**In the code:** the PUCT term is `_ucb_score` in `src/mcts.py`; `child.prior`
is $P(s,a)$, `parent.visit_count` is $\sum_b N(s,b)$, `child.visit_count` is
$N(s,a)$, and `child.value` is $Q$ (negated for the parent's perspective).

---

## 6. Where the exact bonus comes from: variance → Hoeffding → $\ln t$

The UCB1 bonus $\sqrt{\ln t / n_a}$ is not guessed — every piece is forced by a
derivation. Three steps.

### Step A — why a sample mean's error shrinks like $1/\sqrt{n}$

Let the rewards from arm $a$ be i.i.d. with mean $\mu$ and variance $\sigma^2$.
After $n$ pulls the estimate is the sample mean $\bar{x}_n = \frac1n\sum_{i=1}^n x_i$.
Variances of independent variables add, and constants pull out squared:

$$
\operatorname{Var}(\bar{x}_n) = \frac{1}{n^2}\sum_{i=1}^n \operatorname{Var}(x_i)
= \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
\quad\Longrightarrow\quad
\operatorname{std}(\bar{x}_n) = \frac{\sigma}{\sqrt{n}}.
$$

That is the $1/\sqrt{n}$. The intuition: averaging $n$ independent noisy
measurements lets errors partially cancel, but because **variances** add (not
standard deviations), the cancellation rate is $\sqrt{n}$, not $n$ — the same
random-walk scaling. Doubling your samples does *not* halve your error; it
divides it by $\sqrt{2}$.

### Step B — from "typical error" to a high-probability upper bound (Hoeffding)

Variance gives the *typical* spread, but UCB needs **optimism**: an upper bound
on $\mu$ that holds with high probability, so we never under-explore a secretly
good arm. For rewards bounded in $[0,1]$, **Hoeffding's inequality** gives exactly
that — a tail bound with no distributional assumptions:

$$
\Pr\!\big(\mu \ge \bar{x}_n + \varepsilon\big) \le e^{-2 n \varepsilon^2}.
$$

Call the failure probability we'll tolerate $\delta$. Set $e^{-2n\varepsilon^2}=\delta$
and solve for the interval half-width:

$$
\varepsilon = \sqrt{\frac{\ln(1/\delta)}{2n}}
\quad\Longrightarrow\quad
\mu \le \bar{x}_n + \sqrt{\frac{\ln(1/\delta)}{2n}}\ \text{ w.p. } \ge 1-\delta.
$$

Notice the shape is already the UCB1 bonus: still $1/\sqrt{n}$ in the denominator
(Step A's variance scaling survives), with a $\sqrt{\ln(1/\delta)}$ on top that
came purely from inverting the exponential tail. The $\ln$ is the **price of
demanding high confidence** — it grows only logarithmically as you ask for a
tighter guarantee.

### Step C — why specifically $\ln t$ (a union bound over all rounds)

We don't make this decision once; we make it every timestep, for every arm. We
want the bound to hold for *all* of them *simultaneously*, forever. So we spend a
shrinking failure budget per round — choose $\delta_t = t^{-4}$:

$$
\sum_{t\ge 1}\delta_t = \sum_{t\ge 1} t^{-4} < \infty,
$$

so the probability of the bound *ever* failing is finite and small (union bound).
Substituting $\delta = t^{-4}$:

$$
\varepsilon = \sqrt{\frac{\ln(t^{4})}{2n}} = \sqrt{\frac{4\ln t}{2n}}
= \sqrt{\frac{2\ln t}{n}}.
$$

That is UCB1 exactly: $\;\bar{x}_a + \sqrt{2}\,\sqrt{\ln t / n_a}$, i.e. the
constant $c=\sqrt 2$ and the $\ln t$ in the numerator. **So $\ln t$ is the cost of
the union bound over time, and $1/\sqrt{n_a}$ is the concentration of the mean.**
Two different jobs, two different parts of the formula.

### Step D — the payoff: logarithmic regret

This exact balance is what makes total regret $O(\ln T)$. Sketch: a suboptimal
arm with value gap $\Delta = \mu^\star - \mu_a$ keeps getting pulled only while its
upper bound can still exceed the best arm's, i.e. while the bonus
$\sqrt{2\ln t/n_a} \gtrsim \Delta$. That stops once

$$
n_a \gtrsim \frac{2\ln t}{\Delta^2},
$$

so each bad arm is sampled only $\mathcal{O}(\ln T / \Delta^2)$ times. Summing the
$\Delta\cdot n_a$ regret over arms gives $\mathcal{O}\!\big(\sum_a \ln T / \Delta_a\big)$
— matching the Lai–Robbins lower bound up to constants. UCB1 is, in this sense,
optimal: the $\ln t$ is not just sufficient for exploration, it is the least you
can get away with.

---

## References

- Auer, Cesa-Bianchi, Fischer, *Finite-time Analysis of the Multiarmed Bandit
  Problem* (UCB1), 2002.
- Kocsis & Szepesvári, *Bandit based Monte-Carlo Planning* (UCT), 2006.
- Rosin, *Multi-armed bandits with episode context* (PUCT), 2011.
- Silver et al., AlphaGo Zero / AlphaZero, 2017 (the exact PUCT form used here).
