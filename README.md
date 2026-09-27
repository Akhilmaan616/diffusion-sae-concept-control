# Sparse Autoencoders for Concept Control in Diffusion Models

Find interpretable features inside Stable Diffusion's UNet, then use them to add
or erase concepts at inference time — no fine-tuning of the base model.

## Result summary

| | |
|---|---|
| Dictionary recovery (synthetic ground truth) | **100%** of true features at cos > 0.9, EV 0.963 |
| TopK vs L1 at matched sparsity | see [`outputs/l1_vs_topk.md`](outputs/l1_vs_topk.md) |
| SAE on SD1.5 cross-attention | _pending GPU run_ |
| Causal steering (concept gain vs collateral) | _pending GPU run_ |
| Erasure vs ESD/UCE | _day 2_ |

## The claim, and the evidence required for it

The claim is "these are features the model uses". Three levels of evidence, and
only the third supports that claim:

1. **Reconstruction** — necessary, not sufficient. PCA gets high explained
   variance and PCA components are not features.
2. **Substitution** — splice the SAE reconstruction back into the UNet and
   measure the model's *own* denoising loss. Reported as *loss recovered*
   relative to ablating the layer, because substituted loss alone tells you
   nothing about how much that layer mattered.
3. **Causal intervention** — amplify the feature and check the concept appears;
   erase it and check the concept goes *and* that near-neighbour prompts
   survive. A feature that fires on dogs but does nothing when amplified is a
   correlate, not a cause.

Stage 3 of the pipeline outputs **candidate** features. They are not called
concept features until stage 4 passes.

## Two design decisions worth defending

**TopK over L1.** Sparsity as a constraint rather than a penalty: no shrinkage,
and L0 is set rather than tuned. The measured justification is in
`outputs/l1_vs_topk.md` — on a synthetic problem with true k=3, TopK reaches
EV 0.963 at L0 4, while no L1 λ achieves both (λ=0.002 → EV 0.968 but L0 53;
λ=0.1 → L0 8.2 but EV 0.533).

**One SAE per timestep bucket.** A diffusion UNet evaluates the same weights at
~50 noise levels doing different jobs — composition at high t, texture at low t.
Fitting one SAE to that mixture yields features that average incompatible
regimes. Buckets keep the regimes separate and make the difference between them
directly observable by comparing atlases. See the header of `sae/hooks.py`.

## Two implementation details that silently break things

**Classifier-free guidance doubles the batch.** Half of every captured tensor is
the unconditional pass, which contains no prompt information. `cond_only=True`
keeps the conditional half. Without it, roughly half the training data teaches
the SAE the unconditional mean.

**Reconstruction error must be preserved during steering.** Replacing an
activation with `decode(edit(encode(x)))` discards the SAE's reconstruction
error on every token, edited or not. The fix is `decode(edit(encode(x))) + (x -
decode(encode(x)))`, which makes an empty edit exactly the identity. The unit
test `test_error_preservation_matters` shows a no-op edit perturbing the
activation by 59% of its norm without this.

## Pipeline

```bash
python scripts/01_capture.py --out data/acts --n-images 400 --steps 25 --all-sites
python scripts/02_train.py   --acts data/acts --out runs --steps 20000
python scripts/03_locate.py  --sae runs/<shard>/sae.pt --site <site> --bucket <b> \
                             --act-scale <from runs/<shard>/summary.json>
python scripts/04_steer.py   --sae runs/<shard>/sae.pt --site <site> \
                             --act-scale <scale> --mode amplify
python scripts/04_steer.py   ... --mode erase
```

`act_scale` matters: training normalises activations so E‖x‖ = √d, and the
steering hook sees raw activations, so it has to undo that. It is written to
`runs/<shard>/summary.json`.

## Layout

```
sae/model.py     TopK and L1 SAEs, dead-latent aux loss, decoder-norm handling
sae/hooks.py     UNet capture + steering hooks; the timestep and CFG discussion
sae/data.py      shard writer, memmap store, shuffling buffer, normalisation
sae/train.py     training loop and metrics
sae/metrics.py   the three evidence levels
sae/prompts.py   discovery / probe / holdout corpora, kept separate on purpose
tests/           12 tests, all on synthetic data — no GPU or weights needed
```

## Tests

```bash
python tests/test_core.py     # 12/12, CPU, ~2 min
```

The one that matters is `test_sae_recovers_known_dictionary`: data is generated
from a known sparse dictionary and the learned decoder columns are matched
against the true ones. If that fails nothing downstream is trustworthy.

## Notes

`runwayml/stable-diffusion-v1-5` was removed from the HuggingFace Hub; the live
mirror is `stable-diffusion-v1-5/stable-diffusion-v1-5`.

CLIP scoring deliberately uses ViT-B/32, not SD1.5's own ViT-L/14 text encoder.
Scoring with the model's own conditioning encoder measures how well you
satisfied its training objective, which is circular.

## Local checkpoint (partial HuggingFace cache)

`sae/loader.py` handles a cache that has real weights but no pipeline manifest:

```bash
python -m sae.loader "C:/Users/akhil/ife_thesis/models/models--runwayml--stable-diffusion-v1-5"
python -m sae.loader "<same path>" --repair     # writes model_index.json + scheduler/
```

Without `--repair` the loader still works — it assembles the pipeline from
components and reconstructs a DDIM scheduler from the published SD1.5 config
(`scaled_linear` betas 0.00085→0.012, `steps_offset=1`,
`set_alpha_to_one=False`). Those three values are not the diffusers defaults and
getting them wrong produces subtly wrong samples rather than an error.

Pass the cache path as `--model` to any pipeline script to run fully offline.
