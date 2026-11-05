/-
# Concavity of the stage problem of ACE

The stage problem of SDDP for ACE (paper, Proposition 1):

  maximise   log x + log(1 - θ₁ μ^θ₂) + (terms affine in the state) + β θ
  subject to θ ≤ α_j + b_{j,k} k'(x, μ) + b_{j,μ} μ   for every cut j,

with next capital k'(x, μ) = log(1 - θ₁ μ^θ₂) + log(1 - x) + (terms affine in the state).
This file proves that the reward and next capital are concave in the decisions (x, μ), and
that the feasible set is convex when every cut values capital positively (b_{j,k} ≥ 0):
the stage problem is a convex program.

Decisions live in `D = [x_lo, x_hi] × [0, 1]` with `0 < x_lo ≤ x_hi < 1`, and the abatement
cost parameters satisfy `0 ≤ θ₁ < 1 ≤ θ₂` (DICE: θ₂ = 2.6 and θ₁ well below one).
-/
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

open Set Real

namespace Ace

/-! ### General lemmas -/

/-- The logarithm of a positive concave function is concave. -/
theorem concaveOn_log_comp {E : Type*} [AddCommGroup E] [Module ℝ E] {s : Set E} {f : E → ℝ}
    (hf : ConcaveOn ℝ s f) (hpos : ∀ x ∈ s, 0 < f x) :
    ConcaveOn ℝ s (fun x => log (f x)) := by
  refine ⟨hf.1, fun x hx y hy a b ha hb hab => ?_⟩
  have hmix : a • f x + b • f y ∈ Ioi (0 : ℝ) :=
    convex_Ioi (0 : ℝ) (mem_Ioi.mpr (hpos x hx)) (mem_Ioi.mpr (hpos y hy)) ha hb hab
  calc a • log (f x) + b • log (f y)
      ≤ log (a • f x + b • f y) :=
        strictConcaveOn_log_Ioi.concaveOn.2 (mem_Ioi.mpr (hpos x hx)) (mem_Ioi.mpr (hpos y hy)) ha hb hab
    _ ≤ log (f (a • x + b • y)) := log_le_log hmix (hf.2 hx hy ha hb hab)

/-- A concave function of the first coordinate is concave on a product. -/
theorem concaveOn_comp_fst {s t : Set ℝ} {g : ℝ → ℝ} (hg : ConcaveOn ℝ s g) (ht : Convex ℝ t) :
    ConcaveOn ℝ (s ×ˢ t) (fun p => g p.1) :=
  ⟨hg.1.prod ht, fun p hp q hq a b ha hb hab => by simpa using hg.2 hp.1 hq.1 ha hb hab⟩

/-- A concave function of the second coordinate is concave on a product. -/
theorem concaveOn_comp_snd {s t : Set ℝ} {g : ℝ → ℝ} (hs : Convex ℝ s) (hg : ConcaveOn ℝ t g) :
    ConcaveOn ℝ (s ×ˢ t) (fun p => g p.2) :=
  ⟨hs.prod hg.1, fun p hp q hq a b ha hb hab => by simpa using hg.2 hp.2 hq.2 ha hb hab⟩

/-! ### The building blocks of ACE -/

/-- The share of output left after abatement costs, `1 - θ₁ μ^θ₂`, is concave in μ. -/
theorem abatement_share_concave {θ₁ θ₂ : ℝ} (hθ₁ : 0 ≤ θ₁) (hθ₂ : 1 ≤ θ₂) :
    ConcaveOn ℝ (Icc 0 1) (fun μ : ℝ => 1 - θ₁ * μ ^ θ₂) := by
  have hconv : ConvexOn ℝ (Icc 0 1) (fun μ : ℝ => θ₁ * μ ^ θ₂) :=
    ((convexOn_rpow hθ₂).subset Icc_subset_Ici_self (convex_Icc 0 1)).smul hθ₁
  refine ⟨convex_Icc 0 1, fun x hx y hy a b ha hb hab => ?_⟩
  have h := hconv.2 hx hy ha hb hab
  simp only [smul_eq_mul] at h ⊢
  nlinarith [h]

/-- The share left after abatement is positive when `θ₁ < 1`. -/
theorem abatement_share_pos {θ₁ θ₂ : ℝ} (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) :
    ∀ μ ∈ Icc (0 : ℝ) 1, 0 < 1 - θ₁ * μ ^ θ₂ := by
  intro μ hμ
  have hpow : μ ^ θ₂ ≤ 1 := rpow_le_one hμ.1 hμ.2 (by linarith)
  have : θ₁ * μ ^ θ₂ ≤ θ₁ := by
    calc θ₁ * μ ^ θ₂ ≤ θ₁ * 1 := mul_le_mul_of_nonneg_left hpow hθ₁
      _ = θ₁ := mul_one θ₁
  linarith

/-- Log output after abatement costs, `log(1 - θ₁ μ^θ₂)`, is concave in μ. -/
theorem log_abatement_concave {θ₁ θ₂ : ℝ} (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) :
    ConcaveOn ℝ (Icc 0 1) (fun μ : ℝ => log (1 - θ₁ * μ ^ θ₂)) :=
  concaveOn_log_comp (abatement_share_concave hθ₁ hθ₂) (abatement_share_pos hθ₁ hθ₁' hθ₂)

/-- Log consumption share, `log x`, is concave on `[x_lo, x_hi]` with `0 < x_lo`. -/
theorem log_consumption_concave {x_lo x_hi : ℝ} (hlo : 0 < x_lo) :
    ConcaveOn ℝ (Icc x_lo x_hi) log :=
  strictConcaveOn_log_Ioi.concaveOn.subset (fun x hx => mem_Ioi.mpr (lt_of_lt_of_le hlo hx.1))
    (convex_Icc x_lo x_hi)

/-- Log investment share, `log(1 - x)`, is concave on `[x_lo, x_hi]` with `x_hi < 1`. -/
theorem log_investment_concave {x_lo x_hi : ℝ} (hhi : x_hi < 1) :
    ConcaveOn ℝ (Icc x_lo x_hi) (fun x : ℝ => log (1 - x)) := by
  have haff : ConcaveOn ℝ (Icc x_lo x_hi) (fun x : ℝ => 1 - x) := by
    refine ⟨convex_Icc x_lo x_hi, fun x _ y _ a b _ _ hab => ?_⟩
    simp only [smul_eq_mul]
    nlinarith [hab]
  exact concaveOn_log_comp haff (fun x hx => by linarith [hx.2])

/-! ### The stage problem -/

/-- The decisions: consumption rate `x ∈ [x_lo, x_hi]` and abatement rate `μ ∈ [0, 1]`. -/
def decisions (x_lo x_hi : ℝ) : Set (ℝ × ℝ) := Icc x_lo x_hi ×ˢ Icc 0 1

theorem decisions_convex (x_lo x_hi : ℝ) : Convex ℝ (decisions x_lo x_hi) :=
  (convex_Icc x_lo x_hi).prod (convex_Icc 0 1)

/-- The reward, log consumption: `log x + log(1 - θ₁ μ^θ₂) + c`, where `c` collects the terms
that depend only on the state (productivity, labour, κ k and the damages ξ₀(1 - τ₁)). -/
noncomputable def reward (θ₁ θ₂ c : ℝ) (a : ℝ × ℝ) : ℝ :=
  log a.1 + log (1 - θ₁ * a.2 ^ θ₂) + c

/-- Next log capital: `log(1 - θ₁ μ^θ₂) + log(1 - x) + c`, with `c` again collecting the
state-dependent terms and the depreciation constant. -/
noncomputable def nextCapital (θ₁ θ₂ c : ℝ) (a : ℝ × ℝ) : ℝ :=
  log (1 - θ₁ * a.2 ^ θ₂) + log (1 - a.1) + c

variable {θ₁ θ₂ x_lo x_hi : ℝ}

/-- **The reward is concave in the decisions.** -/
theorem reward_concave (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) (hlo : 0 < x_lo) (c : ℝ) :
    ConcaveOn ℝ (decisions x_lo x_hi) (reward θ₁ θ₂ c) := by
  have h1 := concaveOn_comp_fst (log_consumption_concave (x_hi := x_hi) hlo) (convex_Icc (0 : ℝ) 1)
  have h2 := concaveOn_comp_snd (convex_Icc x_lo x_hi) (log_abatement_concave hθ₁ hθ₁' hθ₂)
  exact (h1.add h2).add_const c

/-- **Next capital is concave in the decisions.** -/
theorem nextCapital_concave (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) (hhi : x_hi < 1) (c : ℝ) :
    ConcaveOn ℝ (decisions x_lo x_hi) (nextCapital θ₁ θ₂ c) := by
  have h1 := concaveOn_comp_snd (convex_Icc x_lo x_hi) (log_abatement_concave hθ₁ hθ₁' hθ₂)
  have h2 := concaveOn_comp_fst (log_investment_concave (x_lo := x_lo) hhi) (convex_Icc (0 : ℝ) 1)
  exact (h1.add h2).add_const c

/-- One cut, written in the decisions: `θ ≤ α + b_k k'(x, μ) + b_μ μ`, where `b_μ` collects the
effect of abatement on next period's carbon (affine in μ). -/
noncomputable def cutBound (θ₁ θ₂ c α b_k b_μ : ℝ) (a : ℝ × ℝ) : ℝ :=
  α + b_k * nextCapital θ₁ θ₂ c a + b_μ * a.2

/-- A cut is concave in the decisions when it values capital positively. -/
theorem cutBound_concave (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) (hhi : x_hi < 1)
    (c α b_μ : ℝ) {b_k : ℝ} (hb : 0 ≤ b_k) :
    ConcaveOn ℝ (decisions x_lo x_hi) (cutBound θ₁ θ₂ c α b_k b_μ) := by
  have hk := (nextCapital_concave (x_lo := x_lo) hθ₁ hθ₁' hθ₂ hhi c).smul hb
  have hlin : ConcaveOn ℝ (decisions x_lo x_hi) (fun a : ℝ × ℝ => α + b_μ * a.2) := by
    refine ⟨decisions_convex x_lo x_hi, fun p _ q _ a b _ _ hab => ?_⟩
    simp only [smul_eq_mul, Prod.fst_add, Prod.snd_add, Prod.smul_snd]
    have : α = a * α + b * α := by rw [← add_mul, hab, one_mul]
    nlinarith [this]
  have := hk.add hlin
  refine ⟨this.1, fun p hp q hq a b ha hb' hab => ?_⟩
  have h := this.2 hp hq ha hb' hab
  simp only [cutBound, Pi.add_apply, smul_eq_mul] at h ⊢
  linarith

/-- **Proposition 1: the stage problem is a convex program.** With cuts `j ∈ J` that value
capital positively, the set of decisions and epigraph variables `((x, μ), θ)` that satisfy every
cut is convex; the objective `reward + β θ` is concave. -/
theorem stage_feasible_convex (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) (hhi : x_hi < 1)
    (c : ℝ) {J : Type*} (α b_k b_μ : J → ℝ) (hb : ∀ j, 0 ≤ b_k j) :
    Convex ℝ (⋂ j, { p : (ℝ × ℝ) × ℝ | p.1 ∈ decisions x_lo x_hi ∧
      p.2 ≤ cutBound θ₁ θ₂ c (α j) (b_k j) (b_μ j) p.1 }) :=
  convex_iInter fun j => (cutBound_concave hθ₁ hθ₁' hθ₂ hhi c (α j) (b_μ j) (hb j)).convex_hypograph

theorem stage_objective_concave (hθ₁ : 0 ≤ θ₁) (hθ₁' : θ₁ < 1) (hθ₂ : 1 ≤ θ₂) (hlo : 0 < x_lo)
    (c : ℝ) {β : ℝ} (hβ : 0 ≤ β) :
    ConcaveOn ℝ (decisions x_lo x_hi ×ˢ univ) (fun p : (ℝ × ℝ) × ℝ => reward θ₁ θ₂ c p.1 + β * p.2) := by
  have hr := reward_concave (x_hi := x_hi) hθ₁ hθ₁' hθ₂ hlo c
  refine ⟨(decisions_convex x_lo x_hi).prod convex_univ, fun p hp q hq a b ha hb hab => ?_⟩
  have h := hr.2 hp.1 hq.1 ha hb hab
  simp only [smul_eq_mul, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd] at h ⊢
  nlinarith [h]

end Ace
