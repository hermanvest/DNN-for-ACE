/-
# Concavity of ACE's value functions

SDDP approximates each value function from above by cuts, `V_t(s) ≤ α + b · s`. A cut built from
the value and a supergradient at one state is a valid upper bound everywhere only if `V_t` is
concave. This file proves, by backward induction from `V_T = 0`, that every value function of the
stage problems of ACE is concave in the state and nondecreasing in capital.

The state is `s = (k, m)`: log capital `k` and the remaining states `m ∈ E` (carbon stocks and
transformed temperatures; any real vector space). The decisions `a = (x, μ)` lie in a convex set
`D`. With business-as-usual emissions on an exogenous path, a stage of ACE has the form

  reward        R(a) + κ k + ℓ(m) + c
  next capital  K(a) + κ k + ℓ(m) + c'
  next m        P m + μ • q + q₀

with `R` and `K` concave on `D` (see `Concavity.lean`), `κ ≥ 0`, `ℓ` and `P` linear: the damages
`-ξ₀ τ₁` enter both the reward and next capital through `ℓ`, and carbon and temperatures follow
linear laws of motion in which abatement enters linearly.
-/
import Mathlib.Analysis.Convex.Function
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Module

open Set

namespace Ace

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- One stage of ACE, with the parts of the reward and of next capital that depend on the
decisions (`R`, `K`) kept abstract. -/
structure Stage (E : Type*) [AddCommGroup E] [Module ℝ E] where
  D : Set (ℝ × ℝ)
  R : ℝ × ℝ → ℝ
  K : ℝ × ℝ → ℝ
  κ : ℝ
  ℓ : E →ₗ[ℝ] ℝ
  c : ℝ
  c' : ℝ
  P : E →ₗ[ℝ] E
  q : E
  q₀ : E
  β : ℝ

/-- The conditions under which a stage is well behaved: convex decisions, concave reward and
next capital in the decisions, and nonnegative capital share and discount factor. -/
structure Stage.Regular (S : Stage E) : Prop where
  convex_D : Convex ℝ S.D
  concave_R : ConcaveOn ℝ S.D S.R
  concave_K : ConcaveOn ℝ S.D S.K
  κ_nonneg : 0 ≤ S.κ
  β_nonneg : 0 ≤ S.β

namespace Stage

variable (S : Stage E)

/-- The reward, as a function of the state and the decisions `z = ((k, m), (x, μ))`. -/
def reward (z : (ℝ × E) × (ℝ × ℝ)) : ℝ := S.R z.2 + S.κ * z.1.1 + S.ℓ z.1.2 + S.c

/-- The next state. -/
def next (z : (ℝ × E) × (ℝ × ℝ)) : ℝ × E :=
  (S.K z.2 + S.κ * z.1.1 + S.ℓ z.1.2 + S.c', S.P z.1.2 + z.2.2 • S.q + S.q₀)

/-- The objective of the stage problem: the reward plus the discounted value of the next state. -/
def objective (V : ℝ × E → ℝ) (z : (ℝ × E) × (ℝ × ℝ)) : ℝ := S.reward z + S.β * V (S.next z)

end Stage

/-- A value function that is nondecreasing in capital. -/
def CapitalMonotone (V : ℝ × E → ℝ) : Prop := ∀ (k₁ k₂ : ℝ) (m : E), k₁ ≤ k₂ → V (k₁, m) ≤ V (k₂, m)

namespace Stage

variable {S : Stage E}

theorem next_snd_mix (z₁ z₂ : (ℝ × E) × (ℝ × ℝ)) {a b : ℝ} (hab : a + b = 1) :
    (S.next (a • z₁ + b • z₂)).2 = a • (S.next z₁).2 + b • (S.next z₂).2 := by
  obtain rfl : b = 1 - a := by linarith
  simp only [next, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd, smul_eq_mul,
    map_add, map_smul]
  module

theorem reward_mix (hS : S.Regular) {z₁ z₂ : (ℝ × E) × (ℝ × ℝ)} (h₁ : z₁.2 ∈ S.D) (h₂ : z₂.2 ∈ S.D)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a * S.reward z₁ + b * S.reward z₂ ≤ S.reward (a • z₁ + b • z₂) := by
  have hR := hS.concave_R.2 h₁ h₂ ha hb hab
  simp only [smul_eq_mul] at hR
  simp only [reward, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd, smul_eq_mul,
    map_add, map_smul]
  have hc : S.c = a * S.c + b * S.c := by rw [← add_mul, hab, one_mul]
  nlinarith [hR, hc]

theorem next_fst_mix (hS : S.Regular) {z₁ z₂ : (ℝ × E) × (ℝ × ℝ)} (h₁ : z₁.2 ∈ S.D) (h₂ : z₂.2 ∈ S.D)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a * (S.next z₁).1 + b * (S.next z₂).1 ≤ (S.next (a • z₁ + b • z₂)).1 := by
  have hK := hS.concave_K.2 h₁ h₂ ha hb hab
  simp only [smul_eq_mul] at hK
  simp only [next, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd, smul_eq_mul,
    map_add, map_smul]
  have hc : S.c' = a * S.c' + b * S.c' := by rw [← add_mul, hab, one_mul]
  nlinarith [hK, hc]

/-- **The objective of a stage is jointly concave in the state and the decisions**, if the next
value function is concave and nondecreasing in capital. -/
theorem objective_concave (hS : S.Regular) {V : ℝ × E → ℝ} (hV : ConcaveOn ℝ univ V)
    (hmono : CapitalMonotone V) : ConcaveOn ℝ (univ ×ˢ S.D) (S.objective V) := by
  refine ⟨convex_univ.prod hS.convex_D, fun z₁ hz₁ z₂ hz₂ a b ha hb hab => ?_⟩
  have h₁ : z₁.2 ∈ S.D := hz₁.2
  have h₂ : z₂.2 ∈ S.D := hz₂.2
  -- The value of the mixed next state is at least the mix of the values
  have hcap := next_fst_mix hS h₁ h₂ ha hb hab
  have hm := next_snd_mix (S := S) z₁ z₂ hab
  have hmix : a • S.next z₁ + b • S.next z₂ =
      (a * (S.next z₁).1 + b * (S.next z₂).1, a • (S.next z₁).2 + b • (S.next z₂).2) := by
    ext <;> simp
  have hVmix : a * V (S.next z₁) + b * V (S.next z₂) ≤ V (S.next (a • z₁ + b • z₂)) := by
    have hconc := hV.2 (mem_univ (S.next z₁)) (mem_univ (S.next z₂)) ha hb hab
    simp only [smul_eq_mul] at hconc
    calc a * V (S.next z₁) + b * V (S.next z₂)
        ≤ V (a • S.next z₁ + b • S.next z₂) := hconc
      _ = V (a * (S.next z₁).1 + b * (S.next z₂).1, (S.next (a • z₁ + b • z₂)).2) := by rw [hmix, hm]
      _ ≤ V ((S.next (a • z₁ + b • z₂)).1, (S.next (a • z₁ + b • z₂)).2) := hmono _ _ _ hcap
      _ = V (S.next (a • z₁ + b • z₂)) := rfl
  have hr := reward_mix hS h₁ h₂ ha hb hab
  simp only [objective, smul_eq_mul]
  have hβ := hS.β_nonneg
  nlinarith [mul_le_mul_of_nonneg_left hVmix hβ]

/-- The objective is nondecreasing in capital, for given decisions and other states. -/
theorem objective_monotone (hS : S.Regular) {V : ℝ × E → ℝ} (hmono : CapitalMonotone V)
    (m : E) (x : ℝ × ℝ) {k₁ k₂ : ℝ} (hk : k₁ ≤ k₂) :
    S.objective V ((k₁, m), x) ≤ S.objective V ((k₂, m), x) := by
  have hκ := hS.κ_nonneg
  have hβ := hS.β_nonneg
  have hcap : S.K x + S.κ * k₁ + S.ℓ m + S.c' ≤ S.K x + S.κ * k₂ + S.ℓ m + S.c' := by
    nlinarith [mul_le_mul_of_nonneg_left hk hκ]
  have hV := hmono _ _ (S.P m + x.2 • S.q + S.q₀) hcap
  simp only [objective, reward, next]
  nlinarith [mul_le_mul_of_nonneg_left hk hκ, mul_le_mul_of_nonneg_left hV hβ]

end Stage

/-- **Maximising a jointly concave function over a convex set of decisions gives a concave
function of the state.** `W s` is the maximum (attained) of `h (s, ·)` over `D`. -/
theorem bellman_concave {D : Set (ℝ × ℝ)} (hD : Convex ℝ D) {h : (ℝ × E) × (ℝ × ℝ) → ℝ}
    (hh : ConcaveOn ℝ (univ ×ˢ D) h) {W : ℝ × E → ℝ}
    (hW : ∀ s, IsGreatest ((fun a => h (s, a)) '' D) (W s)) : ConcaveOn ℝ univ W := by
  refine ⟨convex_univ, fun s₁ _ s₂ _ a b ha hb hab => ?_⟩
  obtain ⟨a₁, ha₁, hw₁⟩ := (hW s₁).1
  obtain ⟨a₂, ha₂, hw₂⟩ := (hW s₂).1
  have hmem : a • a₁ + b • a₂ ∈ D := hD ha₁ ha₂ ha hb hab
  have hconc := hh.2 (mk_mem_prod (mem_univ s₁) ha₁) (mk_mem_prod (mem_univ s₂) ha₂) ha hb hab
  have hpair : a • (s₁, a₁) + b • (s₂, a₂) = (a • s₁ + b • s₂, a • a₁ + b • a₂) := by
    ext <;> simp
  rw [hpair] at hconc
  simp only at hw₁ hw₂
  rw [← hw₁, ← hw₂]
  exact hconc.trans ((hW _).2 ⟨_, hmem, rfl⟩)

omit [AddCommGroup E] [Module ℝ E] in
/-- Maximising a function that is nondecreasing in capital gives a function nondecreasing in
capital. -/
theorem bellman_monotone {D : Set (ℝ × ℝ)} {h : (ℝ × E) × (ℝ × ℝ) → ℝ}
    (hh : ∀ (m : E), ∀ x ∈ D, ∀ k₁ k₂ : ℝ, k₁ ≤ k₂ → h ((k₁, m), x) ≤ h ((k₂, m), x))
    {W : ℝ × E → ℝ} (hW : ∀ s, IsGreatest ((fun a => h (s, a)) '' D) (W s)) :
    CapitalMonotone W := by
  intro k₁ k₂ m hk
  obtain ⟨x, hx, hw⟩ := (hW (k₁, m)).1
  simp only at hw
  rw [← hw]
  exact (hh m x hx k₁ k₂ hk).trans ((hW (k₂, m)).2 ⟨x, hx, rfl⟩)

/-- **Every value function of ACE is concave and nondecreasing in capital.** The stages `S t`
are regular, the value after the last stage is zero, and `V t` is the maximum of the objective
of stage `t` given `V (t + 1)`. Then every `V t`, `t ≤ T`, is concave in the state and
nondecreasing in capital, so every cut computed from a supergradient is a valid upper bound. -/
theorem value_concave (S : ℕ → Stage E) (hS : ∀ t, (S t).Regular) (T : ℕ) (V : ℕ → ℝ × E → ℝ)
    (hT : V T = fun _ => 0)
    (hbell : ∀ t < T, ∀ s, IsGreatest ((fun a => (S t).objective (V (t + 1)) (s, a)) '' (S t).D) (V t s)) :
    ∀ t ≤ T, ConcaveOn ℝ univ (V t) ∧ CapitalMonotone (V t) := by
  have key : ∀ n ≤ T, ConcaveOn ℝ univ (V (T - n)) ∧ CapitalMonotone (V (T - n)) := by
    intro n
    induction n with
    | zero =>
      intro _
      simp only [Nat.sub_zero, hT]
      exact ⟨concaveOn_const 0 convex_univ, fun _ _ _ _ => le_refl 0⟩
    | succ n ih =>
      intro hn
      have hprev := ih (by omega)
      have hidx : T - (n + 1) + 1 = T - n := by omega
      have hlt : T - (n + 1) < T := by omega
      have hb := hbell (T - (n + 1)) hlt
      rw [hidx] at hb
      exact ⟨bellman_concave (hS _).convex_D ((S _).objective_concave (hS _) hprev.1 hprev.2) hb,
        bellman_monotone (fun m x _ k₁ k₂ hk => (S _).objective_monotone (hS _) hprev.2 m x hk) hb⟩
  intro t ht
  have := key (T - t) (by omega)
  rwa [show T - (T - t) = t by omega] at this

end Ace
