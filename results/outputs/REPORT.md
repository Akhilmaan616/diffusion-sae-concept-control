# Project B — results

Generated 2026-10-08 15:35:27 · stages run this session: erase_multi

Model `stable-diffusion-v1-5/stable-diffusion-v1-5` · site `up_blocks.1.attentions.1.transformer_blocks.0` · expansion 8 · k 32 · 1000 capture images · analysis bucket 1

## SAE training

| shard | bucket | tokens | tokens/feature | EV | L0 | dead | never fired |
|---|---|---|---|---|---|---|---|
| up_blocks_1_attentions_1_transformer_blocks_0__b0 | 0 | 1,161,216 | 113 | 0.712 | 32.0 | 0.000 | 0 |
| up_blocks_1_attentions_1_transformer_blocks_0__b1 | 1 | 1,032,192 | 101 | 0.752 | 32.0 | 0.000 | 0 |
| up_blocks_1_attentions_1_transformer_blocks_0__b2 | 2 | 1,032,192 | 101 | 0.816 | 32.0 | 0.000 | 0 |
| up_blocks_1_attentions_1_transformer_blocks_0__bpooled | pooled | 3,096,576 | 302 | 0.719 | 32.0 | 0.000 | 0 |
| up_blocks_1_attentions_1_transformer_blocks_0__bpooled_matched | pooled_matched | 1,032,192 | 101 | 0.778 | 32.0 | 0.000 | 0 |

## Faithfulness: loss recovered

Each cell substitutes the SAE reconstruction into the UNet at that bucket's timesteps only. 1.0 = no loss; 0 = no better than ablating the layer. Mean ± std over noise draws. `pooled` saw 3x the data of a bucketed SAE; `pooled_matched` saw the same amount -- compare bucketed with pooled_matched for the fair test.

| | bucket 0 (texture) | bucket 1 (structure) | bucket 2 (composition) |
|---|---|---|---|
| bucketed SAE | 0.633 ± 0.013 | 0.651 ± 0.008 | 0.721 ± 0.023 |
| pooled SAE | 0.636 ± 0.010 | 0.678 ± 0.006 | 0.808 ± 0.029 |
| pooled_matched SAE | 0.550 ± 0.005 | 0.586 ± 0.011 | 0.676 ± 0.026 |

Raw denoising losses (clean / substituted / ablated) and how much ablating the layer costs. A small ablation cost makes the ratio noisy.

| cell | clean | substituted | ablated | ablation cost |
|---|---|---|---|---|
| bucketed_b0 | 0.34132 | 0.34239 | 0.34423 | 0.9% |
| bucketed_b1 | 0.09583 | 0.09633 | 0.09728 | 1.5% |
| bucketed_b2 | 0.01514 | 0.01520 | 0.01538 | 1.6% |
| pooled_b0 | 0.34132 | 0.34238 | 0.34423 | 0.9% |
| pooled_b1 | 0.09583 | 0.09629 | 0.09728 | 1.5% |
| pooled_b2 | 0.01514 | 0.01518 | 0.01538 | 1.6% |
| pooled_matched_b0 | 0.34132 | 0.34263 | 0.34423 | 0.9% |
| pooled_matched_b1 | 0.09583 | 0.09643 | 0.09728 | 1.5% |
| pooled_matched_b2 | 0.01514 | 0.01522 | 0.01538 | 1.6% |

## Candidate concept features (correlational)

| concept | feature | AUC | fires on concept | fires on neutral |
|---|---|---|---|---|
| van_gogh | 4804 | 0.774 | 0.250 | 0.002 |
| dog | 5935 | 0.734 | 0.166 | 0.014 |
| fire | 7518 | 0.746 | 0.193 | 0.009 |
| watercolour | 5891 | 0.883 | 0.558 | 0.016 |

## Causal test: amplify

ADD the feature direction on neutral prompts; strength = multiple of the feature's typical active code. Paired CLIP deltas (edited − unedited, same seed), mean ± std over 3 seeds.

| concept | feature | best strength | concept CLIP Δ | prompt CLIP Δ | holdout prompt Δ |
|---|---|---|---|---|---|
| van_gogh | 4804 | 8 | +0.0639 ± 0.0087 | -0.0021 ± 0.0084 | — |
| dog | 5935 | 8 | +0.0171 ± 0.0016 | -0.0510 ± 0.0047 | — |
| fire | 7518 | 8 | +0.0246 ± 0.0035 | -0.0482 ± 0.0105 | — |
| watercolour | 5891 | 8 | +0.0234 ± 0.0048 | -0.0376 ± 0.0044 | — |

## Causal test: erase

scale the feature's code by gamma on held-out concept prompts (EVAL); holdout = near-neighbour prompts, each against its own unedited baseline. Paired CLIP deltas (edited − unedited, same seed), mean ± std over 3 seeds.

| concept | feature | best strength | concept CLIP Δ | prompt CLIP Δ | holdout prompt Δ |
|---|---|---|---|---|---|
| van_gogh | 4804 | 0 | -0.0008 ± 0.0012 | -0.0015 ± 0.0015 | -0.0014 ± 0.0017 |
| dog | 5935 | -1 | -0.0011 ± 0.0003 | +0.0004 ± 0.0024 | +0.0011 ± 0.0008 |
| fire | 7518 | 0 | +0.0000 ± 0.0002 | +0.0002 ± 0.0003 | -0.0020 ± 0.0033 |
| watercolour | 5891 | -3 | -0.0003 ± 0.0049 | +0.0004 ± 0.0043 | -0.0014 ± 0.0006 |

## Multi-feature erasure (dose-response)

Scale the top-N concept features (by AUC) to gamma at every step, on EVAL prompts. `random20` = 20 features active on the concept prompts but not in its top 20 -- the control. `edit` = relative change to the activations. Paired CLIP deltas, mean ± std over seeds. Success = concept Δ clearly negative, below random20, with holdout Δ near zero.

### SAE: bucketed_b1

| concept | setting | concept CLIP Δ | prompt CLIP Δ | holdout prompt Δ | edit |
|---|---|---|---|---|---|
| van_gogh | top5 | -0.0014 ± 0.0033 | -0.0027 ± 0.0051 | -0.0018 ± 0.0014 | 16.2% |
| van_gogh | top10 | -0.0012 ± 0.0039 | -0.0039 ± 0.0043 | -0.0010 ± 0.0013 | 17.4% |
| van_gogh | top20 | -0.0001 ± 0.0033 | -0.0019 ± 0.0049 | -0.0020 ± 0.0004 | 18.5% |
| van_gogh | top20_neg | +0.0012 ± 0.0014 | -0.0004 ± 0.0032 | -0.0008 ± 0.0009 | 36.7% |
| van_gogh | random20 | -0.0007 ± 0.0011 | +0.0002 ± 0.0021 | +0.0014 ± 0.0009 | 6.7% |
| dog | top5 | -0.0012 ± 0.0003 | +0.0008 ± 0.0020 | -0.0010 ± 0.0028 | 5.1% |
| dog | top10 | -0.0009 ± 0.0011 | +0.0006 ± 0.0031 | -0.0009 ± 0.0028 | 8.8% |
| dog | top20 | -0.0019 ± 0.0009 | -0.0008 ± 0.0031 | +0.0009 ± 0.0016 | 12.8% |
| dog | top20_neg | -0.0029 ± 0.0010 | -0.0037 ± 0.0009 | -0.0001 ± 0.0014 | 24.4% |
| dog | random20 | +0.0003 ± 0.0012 | -0.0008 ± 0.0032 | -0.0005 ± 0.0009 | 9.3% |
| fire | top5 | +0.0006 ± 0.0017 | +0.0062 ± 0.0026 | -0.0017 ± 0.0023 | 14.0% |
| fire | top10 | +0.0006 ± 0.0019 | +0.0063 ± 0.0015 | -0.0018 ± 0.0024 | 14.5% |
| fire | top20 | +0.0006 ± 0.0017 | +0.0077 ± 0.0018 | +0.0004 ± 0.0056 | 16.7% |
| fire | top20_neg | -0.0088 ± 0.0041 | +0.0085 ± 0.0019 | -0.0026 ± 0.0065 | 30.4% |
| fire | random20 | -0.0000 ± 0.0002 | +0.0013 ± 0.0010 | -0.0002 ± 0.0111 | 27.2% |
| watercolour | top5 | +0.0006 ± 0.0038 | +0.0067 ± 0.0011 | -0.0045 ± 0.0022 | 22.5% |
| watercolour | top10 | -0.0018 ± 0.0051 | +0.0038 ± 0.0029 | -0.0034 ± 0.0028 | 23.3% |
| watercolour | top20 | -0.0012 ± 0.0047 | +0.0033 ± 0.0027 | -0.0026 ± 0.0021 | 24.9% |
| watercolour | top20_neg | -0.0032 ± 0.0063 | -0.0020 ± 0.0021 | -0.0081 ± 0.0035 | 46.4% |
| watercolour | random20 | -0.0000 ± 0.0014 | +0.0005 ± 0.0017 | -0.0011 ± 0.0019 | 5.9% |

### SAE: pooled

| concept | setting | concept CLIP Δ | prompt CLIP Δ | holdout prompt Δ | edit |
|---|---|---|---|---|---|
| van_gogh | top5 | -0.0014 ± 0.0031 | -0.0032 ± 0.0056 | -0.0022 ± 0.0006 | 16.9% |
| van_gogh | top10 | -0.0006 ± 0.0037 | -0.0027 ± 0.0051 | -0.0030 ± 0.0006 | 17.7% |
| van_gogh | top20 | +0.0005 ± 0.0038 | -0.0015 ± 0.0066 | -0.0013 ± 0.0011 | 19.1% |
| van_gogh | top20_neg | +0.0011 ± 0.0020 | -0.0039 ± 0.0038 | -0.0023 ± 0.0010 | 37.6% |
| van_gogh | random20 | +0.0011 ± 0.0006 | +0.0008 ± 0.0005 | +0.0011 ± 0.0007 | 5.5% |
| dog | top5 | +0.0000 ± 0.0016 | -0.0007 ± 0.0006 | +0.0003 ± 0.0019 | 5.6% |
| dog | top10 | -0.0003 ± 0.0012 | +0.0002 ± 0.0013 | +0.0005 ± 0.0018 | 8.3% |
| dog | top20 | +0.0014 ± 0.0002 | -0.0010 ± 0.0013 | +0.0007 ± 0.0012 | 12.1% |
| dog | top20_neg | -0.0001 ± 0.0020 | -0.0036 ± 0.0044 | -0.0049 ± 0.0049 | 22.4% |
| dog | random20 | +0.0003 ± 0.0009 | +0.0004 ± 0.0012 | +0.0002 ± 0.0010 | 8.8% |
| fire | top5 | +0.0003 ± 0.0002 | -0.0000 ± 0.0006 | -0.0033 ± 0.0023 | 7.6% |
| fire | top10 | +0.0006 ± 0.0003 | +0.0010 ± 0.0012 | -0.0030 ± 0.0024 | 11.0% |
| fire | top20 | -0.0002 ± 0.0006 | +0.0024 ± 0.0017 | +0.0030 ± 0.0106 | 13.6% |
| fire | top20_neg | -0.0013 ± 0.0031 | +0.0067 ± 0.0012 | -0.0025 ± 0.0085 | 23.1% |
| fire | random20 | -0.0014 ± 0.0012 | +0.0011 ± 0.0012 | +0.0005 ± 0.0032 | 9.3% |
| watercolour | top5 | -0.0007 ± 0.0029 | +0.0046 ± 0.0013 | -0.0044 ± 0.0012 | 24.4% |
| watercolour | top10 | -0.0009 ± 0.0038 | +0.0015 ± 0.0011 | -0.0046 ± 0.0033 | 25.1% |
| watercolour | top20 | +0.0006 ± 0.0025 | +0.0027 ± 0.0011 | -0.0037 ± 0.0019 | 26.5% |
| watercolour | top20_neg | +0.0000 ± 0.0026 | -0.0016 ± 0.0033 | -0.0044 ± 0.0033 | 49.9% |
| watercolour | random20 | -0.0002 ± 0.0007 | +0.0018 ± 0.0003 | -0.0010 ± 0.0027 | 7.2% |

Image grids: `/kaggle/working/project_b/outputs/erase_multi` (rows: base, then each setting; columns: first 4 prompts).

## Where things are

- SAEs: `/kaggle/working/project_b/runs`
- atlas sheets: `/kaggle/working/project_b/outputs/atlas`
- steering images: `/kaggle/working/project_b/outputs/grids`
- logs: `/kaggle/working/project_b/logs`
