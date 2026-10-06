"""Prompt corpora.

Three separate sets, and keeping them separate is the point:

  DISCOVERY  -- broad, concept-agnostic. Used to *train* the SAE. If you train
                on dog prompts you will find dog features and learn nothing
                about whether the method works; the dictionary must be fit on a
                distribution that was not chosen to contain the answer.

  PROBE      -- concept-labelled. Used to *locate* which learned feature
                corresponds to a concept, by comparing mean activation on
                concept prompts against DISCOVERY. Never used for training.

  HOLDOUT    -- prompts that must survive an erasure. This is the collateral-
                damage set, and choosing it well is most of the experimental
                design: it has to contain things that are *near* the erased
                concept, not random unrelated prompts. Erasing "Van Gogh" and
                then checking that "a photo of a cat" still works proves
                nothing. Checking that "a swirling starry night sky" and "an
                impressionist landscape" still work is the real test.
"""

DISCOVERY = [
    "a photograph of a city street at night",
    "a bowl of fruit on a wooden table",
    "an old man reading a newspaper",
    "a mountain range at sunrise",
    "a red sports car parked outside a cafe",
    "two children playing in a park",
    "a bustling market with colourful stalls",
    "a lighthouse on a rocky coast",
    "a plate of pasta with basil",
    "an empty classroom with wooden desks",
    "a golden retriever running on grass",
    "a glass skyscraper reflecting clouds",
    "a watercolour painting of a harbour",
    "a close-up of a mechanical watch",
    "a field of sunflowers under blue sky",
    "a snowy forest path in winter",
    "a woman in a yellow raincoat",
    "an astronaut floating above earth",
    "a vintage bicycle against a brick wall",
    "a steaming cup of coffee on a windowsill",
    "a fishing boat in a calm bay",
    "a crowded subway platform",
    "a stack of old books and a candle",
    "a desert canyon at midday",
    "a neon sign above a diner",
    "a cat sleeping on a sofa",
    "a chef plating food in a kitchen",
    "a waterfall in a tropical jungle",
    "a violin resting on sheet music",
    "a hot air balloon over farmland",
    "a rainy window with city lights behind",
    "a marble statue in a museum hall",
]

# Each probe concept: prompts that contain it, for feature localisation.
PROBE = {
    "van_gogh": [
        "a wheat field in the style of Van Gogh",
        "a Van Gogh painting of a village at night",
        "a self portrait in Van Gogh's brushwork",
        "swirling impasto brushstrokes, Van Gogh style",
    ],
    "dog": [
        "a dog sitting on a porch",
        "a puppy chasing a ball",
        "a border collie in a field",
        "a dog wearing a scarf",
    ],
    "fire": [
        "a campfire at night",
        "a burning building with thick smoke",
        "a candle flame in the dark",
        "a forest fire on a hillside",
    ],
    "watercolour": [
        "a watercolour painting of a garden",
        "soft watercolour washes of a coastline",
        "a loose watercolour sketch of a street",
        "watercolour on textured paper, a still life",
    ],
}

# Near-neighbour prompts that must NOT break when the matching concept is erased.
HOLDOUT = {
    "van_gogh": [
        "a swirling starry night sky over a village",
        "an impressionist landscape with thick brushstrokes",
        "a wheat field under a windy sky",
        "an oil painting of sunflowers in a vase",
        "a post-impressionist portrait of a man",
    ],
    "dog": [
        "a wolf standing in snow",
        "a fox in a meadow",
        "a cat sitting on a porch",
        "a horse running on grass",
        "a person walking on a leafy path",
    ],
    "fire": [
        "a warm orange sunset over hills",
        "a glowing lantern in a tent",
        "steam rising from a hot spring",
        "an autumn forest in red and orange",
        "a blacksmith at an anvil",
    ],
    "watercolour": [
        "a pastel drawing of a garden",
        "an ink sketch of a coastline",
        "a soft pencil study of a street",
        "an acrylic still life on canvas",
        "a faded vintage photograph of flowers",
    ],
}


def all_training_prompts(repeat: int = 1) -> list[str]:
    return DISCOVERY * repeat


# Concept prompts used ONLY to evaluate erasure -- never to locate features.
# Run 1 evaluated on PROBE, the same prompts the feature was selected on, which
# biases the result toward the feature looking effective.
EVAL = {
    "van_gogh": [
        "a cafe terrace at night in the style of Van Gogh",
        "a portrait of a farmer painted like Van Gogh",
        "cypress trees under a swirling sky, Van Gogh style",
        "a bedroom interior in Van Gogh's style",
        "irises in a garden, a Van Gogh painting",
        "a harbour with boats in the style of Van Gogh",
    ],
    "dog": [
        "a dog lying by a fireplace",
        "a dalmatian on a beach",
        "a dog catching a frisbee",
        "a sleepy beagle on a couch",
        "a dog in a snowy park",
        "a terrier looking out of a car window",
    ],
    "fire": [
        "a bonfire on a beach at night",
        "flames rising from a fireplace",
        "a torch burning in a dark cave",
        "a wildfire seen from a distance",
        "a matchstick bursting into flame",
        "fire in a metal barrel in an alley",
    ],
    "watercolour": [
        "a watercolour of a mountain lake",
        "a watercolour painting of a cat",
        "a watercolour of a city skyline at dusk",
        "a watercolour still life of pears",
        "a watercolour of a sailboat",
        "a watercolour of autumn trees",
    ],
}
