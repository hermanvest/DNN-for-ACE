# Formal proofs in Lean

SDDP's guarantees rest on convexity: each stage problem must be a convex program, and the value
functions must be concave in the state, so that every cut computed from a value and a
supergradient is a valid upper bound. This directory proves these properties for ACE, and the
closed-form facts the paper uses, in [Lean 4](https://lean-lang.org) with
[Mathlib](https://github.com/leanprover-community/mathlib4).

| File | What it proves | Paper |
|---|---|---|
| [`AceProofs/Concavity.lean`](AceProofs/Concavity.lean) | `log(1 − θ₁ μ^θ₂)`, `log x` and `log(1 − x)` are concave; the reward and next capital are concave in the decisions; with cuts that value capital positively, the feasible set of the stage problem is convex and its objective concave | Proposition 1 |
| [`AceProofs/Bellman.lean`](AceProofs/Bellman.lean) | A stage's objective is jointly concave in the state and the decisions if the next value function is concave and nondecreasing in capital; maximising over a convex set of decisions preserves both properties; hence, by backward induction from `V_T = 0`, **every value function is concave and nondecreasing in capital** | Validity of the cuts |
| [`AceProofs/ClosedForm.lean`](AceProofs/ClosedForm.lean) | The capital slope `κ/(1 − βκ)` is a fixed point of the cut recursion and gives the consumption rate `1 − βκ`; the truncated capital slope is `(1 − (βκ)^n)` times the closed form; the carbon matrix conserves carbon, so `Φᵀ 1 = 1` and the discounted carbon recursion shrinks that direction exactly like `β^n` | Proposition 3, Remark 1 |

## How the model is represented

- **Decisions:** `(x, μ) ∈ [x_lo, x_hi] × [0, 1]` with `0 < x_lo ≤ x_hi < 1`. In the
  code, `[0.2, 0.98]`.
- **Abatement cost:** `0 ≤ θ₁ < 1 ≤ θ₂`. In DICE, `θ₂ = 2.6` and `θ₁` is well below one.
- **State:** `(k, m)`, with log capital `k` and the other states in an arbitrary real vector
  space `E` (in the code, the three carbon stocks and two transformed temperatures).
- **A stage:** given as a `Stage` structure.
  - Reward `R(a) + κ k + ℓ(m) + c`, and next capital `K(a) + κ k + ℓ(m) + c'`.
  - Next `m` is `P m + μ q + q₀`, with `ℓ` and `P` linear. This holds when business-as-usual
    emissions follow an exogenous path, ACE's assumption.
  - `R` and `K` are any functions that are concave on the decisions; `Concavity.lean` shows that
    ACE's are.
- **Maximisation:** the value function `V t` is the maximum of the stage objective. The maximum
  is taken as attained (`IsGreatest`). In ACE this holds because the decision set is compact and
  the objective continuous on it; that step is assumed, not formalised.

## What is not formalised

- **Cuts from duals.** The cut gradient comes from the solver's multipliers (Proposition 2) by
  Lagrangian duality and the envelope theorem.
- **Corollary 1**, that SDDP converges in two iterations.
- **The numerical results.** These remain computational.

## Building

The proofs need Lean 4.35.0 and Mathlib `v4.35.0`, pinned in `lean-toolchain` and
`lake-manifest.json`:

```bash
cd 6_neural_sddp/lean
lake exe cache get   # downloads prebuilt Mathlib (skip if unavailable: lake build compiles it)
lake build
```

A successful `lake build` means that every theorem is proved: the files contain no `sorry`.
