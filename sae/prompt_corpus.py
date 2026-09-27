"""A larger, combinatorial discovery corpus.

WHY THE HAND-WRITTEN LIST IS NOT ENOUGH

``prompts.DISCOVERY`` has 32 entries. Capturing 400 images from it means each
prompt is rendered ~12 times with different noise. That gives you more *tokens*
but not more *concepts*: the semantic support of the dataset is still 32 scenes.
A 10,000-feature dictionary fit on 32 scenes cannot learn 10,000 distinct
concepts, because there are not that many distinct things in the data. The
dictionary fills up with variations on the same few scenes and the atlas looks
repetitive.

Token count and semantic diversity are separate axes and they fail differently:

    too few tokens      -> dead latents, unstable training
    too few concepts    -> live latents that all describe the same handful of
                           scenes; looks fine in the metrics, useless in the atlas

So this module builds a corpus by crossing subjects, settings, lighting and
media. Combinatorial rather than hand-written because 400 hand-written prompts
would be dominated by whatever the author happened to think of, and the crossing
guarantees that each axis varies independently of the others -- which is exactly
the condition under which a feature for "night-time" can be separated from a
feature for "street", instead of the two always co-occurring.

Deterministic given a seed, so the corpus is reproducible.
"""

from __future__ import annotations

import random

SUBJECTS = [
    "an old fisherman", "a young woman in a linen dress", "a border collie",
    "a tabby cat", "a red bicycle", "a wooden sailboat", "a grand piano",
    "a bowl of lemons", "a stack of hardback books", "a brass telescope",
    "a vintage typewriter", "a crowd of commuters", "a solitary lighthouse",
    "a glass of red wine", "a violin and bow", "a rusted tractor",
    "a marble bust", "a flock of starlings", "a steaming bowl of ramen",
    "a child on a swing", "a mountain goat", "a fountain pen on paper",
    "an abandoned petrol station", "a chess board mid-game",
    "a pair of leather boots", "a hot air balloon", "a street food cart",
    "a sleeping fox", "a cracked ceramic vase", "a suspension bridge",
    "a greenhouse full of ferns", "a saxophone player",
]

SETTINGS = [
    "in a narrow cobbled alley", "on a windswept cliff", "inside a quiet library",
    "in a crowded night market", "on a frozen lake", "in a sunlit kitchen",
    "beside a rain-slicked road", "in a dense pine forest", "on a rooftop terrace",
    "in an empty warehouse", "at the edge of a desert", "in a tiled subway station",
    "on a wooden pier", "in a field of tall grass", "inside a cathedral",
    "in a cluttered workshop",
]

LIGHT = [
    "at golden hour", "under harsh midday sun", "lit by a single candle",
    "in dense fog", "under neon signs", "at blue hour", "backlit by a window",
    "under overcast grey sky", "lit by firelight", "in moonlight",
]

MEDIUM = [
    "photograph", "oil painting", "watercolour", "charcoal sketch",
    "35mm film still", "pencil study", "gouache illustration",
    "black and white photograph", "pastel drawing", "linocut print",
]


def build_discovery(n: int = 256, seed: int = 0, with_medium: float = 0.45) -> list[str]:
    """Return ``n`` distinct prompts crossing the four axes.

    ``with_medium`` is the fraction that carry an explicit medium. Not all of
    them, because if every prompt names a medium the model never sees the
    default photographic mode and medium features become entangled with
    everything else.
    """
    rng = random.Random(seed)
    seen, out = set(), []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        s, st, li = rng.choice(SUBJECTS), rng.choice(SETTINGS), rng.choice(LIGHT)
        if rng.random() < with_medium:
            p = f"{rng.choice(MEDIUM)} of {s} {st}, {li}"
        else:
            p = f"{s} {st}, {li}"
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def coverage(prompts: list[str]) -> dict:
    """How well a corpus covers each axis. Run this before trusting a corpus."""
    def hits(axis):
        return sum(any(v.lower() in p.lower() for p in prompts) for v in axis)
    return {
        "n_prompts": len(prompts),
        "n_unique": len(set(prompts)),
        "subjects_covered": f"{hits(SUBJECTS)}/{len(SUBJECTS)}",
        "settings_covered": f"{hits(SETTINGS)}/{len(SETTINGS)}",
        "lighting_covered": f"{hits(LIGHT)}/{len(LIGHT)}",
        "media_covered": f"{hits(MEDIUM)}/{len(MEDIUM)}",
        "max_combinations": len(SUBJECTS) * len(SETTINGS) * len(LIGHT) * (len(MEDIUM) + 1),
    }


if __name__ == "__main__":
    import json
    ps = build_discovery(256)
    print(json.dumps(coverage(ps), indent=2))
    for p in ps[:8]:
        print(" ", p)
