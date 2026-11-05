/-
# The closed-form cuts of ACE and the horizon truncation

* The slope of the value function on capital, `b_k = κ / (1 - β κ)`, is the fixed point of the
  capital part of the cut recursion `b ↦ κ (1 + β b)` (paper, Proposition 3), and the optimal
  consumption rate `1 / (1 + β b_k)` equals `1 - β κ`.
* With a zero value after the last stage, the capital slope `n` periods before the end is
  `(1 - (β κ)^n) b_k`: the truncation error shrinks like `(β κ)^n`.
* The carbon matrix of DICE (with the atmosphere-ocean exchange scaled by `ω`) conserves carbon:
  its columns sum to one, so its transpose maps the vector of ones to itself, and the discounted
  carbon recursion `v ↦ β Φᵀ v` shrinks that direction exactly like `β^n`, no faster. This is why
  carbon's shadow price converges only at the rate of discounting (paper, Remark 1).
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Fin.VecNotation
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring

open Matrix

namespace Ace

/-! ### Capital -/

/-- The closed-form capital slope `κ / (1 - β κ)` is a fixed point of `b ↦ κ (1 + β b)`. -/
theorem capital_slope_fixed (β κ : ℝ) (h : β * κ ≠ 1) :
    κ * (1 + β * (κ / (1 - β * κ))) = κ / (1 - β * κ) := by
  have h' : 1 - β * κ ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  field_simp
  ring

/-- The optimal consumption rate `1 / (1 + β b_k)` is `1 - β κ`. -/
theorem consumption_rate (β κ : ℝ) (h : β * κ ≠ 1) :
    1 / (1 + β * (κ / (1 - β * κ))) = 1 - β * κ := by
  have h' : 1 - β * κ ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  field_simp
  ring

/-- With a zero value after the last stage, the capital slope `n` periods before the end is
`(1 - (β κ)^n)` times its closed-form value. -/
theorem capital_slope_truncated (β κ : ℝ) (h : β * κ ≠ 1) (b : ℕ → ℝ) (h₀ : b 0 = 0)
    (hrec : ∀ n, b (n + 1) = κ * (1 + β * b n)) :
    ∀ n, b n = (1 - (β * κ) ^ n) * (κ / (1 - β * κ)) := by
  have h' : 1 - β * κ ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  intro n
  induction n with
  | zero => simp [h₀]
  | succ n ih =>
    rw [hrec, ih, pow_succ]
    field_simp
    ring

/-! ### Carbon -/

/-- The carbon transition matrix of DICE between the atmosphere, the upper ocean and the deep
ocean, with transfer rates `ξ₀ … ξ₃` and the atmosphere-ocean exchange scaled by `ω`. -/
def carbonMatrix (ξ₀ ξ₁ ξ₂ ξ₃ ω : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![1 - ω * ξ₀, ω * ξ₂, 0;
     ω * ξ₀, 1 - ξ₁ - ω * ξ₂, ξ₃;
     0, ξ₁, 1 - ξ₃]

/-- **The carbon cycle conserves carbon:** every column of the transition matrix sums to one. -/
theorem carbon_columns_sum (ξ₀ ξ₁ ξ₂ ξ₃ ω : ℝ) (j : Fin 3) :
    ∑ i, carbonMatrix ξ₀ ξ₁ ξ₂ ξ₃ ω i j = 1 := by
  fin_cases j <;> simp [carbonMatrix, Fin.sum_univ_three] <;> ring

/-- Hence the transpose maps the vector of ones to itself: one is an eigenvalue of `Φᵀ`. -/
theorem carbon_transpose_ones (ξ₀ ξ₁ ξ₂ ξ₃ ω : ℝ) :
    (carbonMatrix ξ₀ ξ₁ ξ₂ ξ₃ ω)ᵀ *ᵥ (fun _ => (1 : ℝ)) = fun _ => 1 := by
  funext j
  simp only [mulVec, dotProduct, transpose_apply, mul_one]
  exact carbon_columns_sum ξ₀ ξ₁ ξ₂ ξ₃ ω j

/-- The discounted carbon recursion `v ↦ β Φᵀ v` shrinks the direction of ones exactly like
`β^n`: the truncation error of the carbon slopes cannot decay faster than the discount factor. -/
theorem carbon_recursion_ones (ξ₀ ξ₁ ξ₂ ξ₃ ω β : ℝ) (v : ℕ → Fin 3 → ℝ)
    (h₀ : v 0 = fun _ => 1) (hrec : ∀ n, v (n + 1) = β • ((carbonMatrix ξ₀ ξ₁ ξ₂ ξ₃ ω)ᵀ *ᵥ v n)) :
    ∀ n, v n = fun _ => β ^ n := by
  intro n
  induction n with
  | zero => simp [h₀]
  | succ n ih =>
    rw [hrec, ih]
    have : (fun _ : Fin 3 => β ^ n) = β ^ n • (fun _ : Fin 3 => (1 : ℝ)) := by
      funext; simp
    rw [this, mulVec_smul, carbon_transpose_ones]
    funext
    simp [pow_succ]
    ring

end Ace
