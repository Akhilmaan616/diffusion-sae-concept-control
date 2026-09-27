# Why this repo defaults to TopK

Measured, not asserted. Run: `python tests/test_core.py` plus the sweep in
`scripts/sweep_l1.py`.

## Setup

Synthetic ground-truth sparse dictionary, so "did it find the real features" is
actually checkable:

- `x = D z + 0.01 eps`, `D` has 64 unit-norm columns in R^32
- `z` is exactly **k = 3** sparse, non-negative
- 20,000 tokens, 1,500 steps, Adam 1e-3, batch 1024, CPU

## Result

| mode | setting | explained variance | L0 |
|---|---|---|---|
| **TopK** | k = 4 | **0.963** | **4.0** |
| L1 | λ = 0.002 | 0.968 | 53.4 |
| L1 | λ = 0.01 | 0.941 | 40.2 |
| L1 | λ = 0.03 | 0.854 | 23.9 |
| L1 | λ = 0.1 | 0.533 | 8.2 |
| L1 | λ = 0.3 | 0.003 | 0.5 |

TopK also recovered **100% of the true dictionary columns** at cosine > 0.9
(median best-match cosine 1.000).

## Reading it

The true answer is L0 = 3 at high explained variance. TopK hits it directly,
because sparsity is a constraint rather than a penalty — you set `k` and the
optimiser spends its capacity on reconstruction alone.

L1 never gets there. Every λ that produces real sparsity has already destroyed
reconstruction (λ = 0.1 → L0 8.2 but EV 0.533), and every λ with good
reconstruction is barely sparse (λ = 0.002 → EV 0.968 but L0 53, i.e. **83% of
the dictionary active on every token**, which is not a sparse code at all).

Two separate causes, worth distinguishing:

1. **Shrinkage.** L1 penalises the magnitude of active codes, so the optimum
   under-estimates every feature it does detect. Reconstruction degrades for a
   reason unrelated to which features were selected.
2. **No direct control of L0.** λ → L0 is an indirect, data-dependent mapping
   that drifts during training, so hitting a target sparsity means a sweep, and
   the sweep has to be redone for every site and timestep bucket. With four
   capture sites × three buckets that is twelve sweeps instead of twelve numbers
   you simply set.

The second point is the practical one for this project: TopK makes the twelve
SAEs comparable to each other, because they all have identical L0 by
construction. Comparing L1 SAEs across sites means comparing models at different
points on different trade-off curves.

## The honest caveat

This is a synthetic problem where the generative process is exactly a sparse
non-negative dictionary — i.e. exactly the SAE's inductive bias. Real UNet
activations are not guaranteed to be. What this experiment establishes is that
the *optimiser and implementation* recover a sparse dictionary when one exists;
it does not establish that one exists in Stable Diffusion. That question is
settled by the substitution and causal-intervention evidence, not by this table.
