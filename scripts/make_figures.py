#!/usr/bin/env python
"""Build the README figures from the saved results (no GPU needed).

    python scripts/make_figures.py --results results --out assets
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated categorical order
CONCEPTS = ["van_gogh", "dog", "fire", "watercolour"]
NAMES = {"van_gogh": "Van Gogh", "dog": "dog", "fire": "fire", "watercolour": "watercolour"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.grid.axis": "y", "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def load(p):
    return json.loads(Path(p).read_text())


def bars(ax, groups, series, values, errors, labels_for=()):
    n = len(series)
    w = 0.8 / n
    for i, (name, color) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, values[i], w, yerr=errors[i], color=color, edgecolor=SURFACE,
               linewidth=2, label=name, error_kw={"ecolor": INK2, "elinewidth": 1,
                                                  "capsize": 3})
        if name in labels_for:
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for x, v, e in zip(xs, values[i], errors[i]):
                ax.text(x, v + e + 0.01 * span, f"{v:.2f}" if abs(v) > 0.1
                        else f"{v:+.3f}", ha="center", va="bottom",
                        fontsize=9, color=INK2)
    ax.set_xticks(range(len(groups)), groups)


def fig_loss_recovered(R, out):
    tab = load(R / "outputs" / "substitution_table.json")
    kinds = [("per-timestep SAE (1M tokens each)", "bucketed"),
             ("pooled SAE, matched data (1M)", "pooled_matched"),
             ("pooled SAE, all data (3M)", "pooled")]
    vals = [[tab[f"{k}_b{b}"]["loss_recovered"] for b in range(3)] for _, k in kinds]
    errs = [[tab[f"{k}_b{b}"]["loss_recovered_std"] for b in range(3)] for _, k in kinds]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_ylim(0, 1)
    bars(ax, ["texture\n(t < 334)", "structure\n(334 – 667)", "composition\n(t ≥ 667)"],
         list(zip([k[0] for k in kinds], SERIES)), vals, errs,
         labels_for=[k[0] for k in kinds])
    ax.set_ylim(0, 1)
    ax.set_ylabel("loss recovered (1 = lossless)")
    ax.set_title("Loss recovered by SAE type and timestep regime (mean ± std, 4 draws)",
                 loc="left")
    ax.legend(loc="upper left", ncol=1, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "fig_loss_recovered.png", dpi=160)
    plt.close(fig)


def fig_amplify_vs_erase(R, out):
    amp = load(R / "outputs" / "steer_amplify.json")["report"]
    er = load(R / "outputs" / "erase_multi_bucketed_b1.json")["report"]
    series = [("amplify: add 1 feature", lambda c: (amp[c]["concept_delta"],
                                                     amp[c]["concept_delta_std"])),
              ("erase: top-20 features → 0", lambda c: _e(er, c, "top20")),
              ("erase: top-20 features → −1", lambda c: _e(er, c, "top20_neg")),
              ("control: 20 other features → 0", lambda c: _e(er, c, "random20"))]
    vals = [[f(c)[0] for c in CONCEPTS] for _, f in series]
    errs = [[f(c)[1] for c in CONCEPTS] for _, f in series]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.set_ylim(-0.016, 0.085)
    bars(ax, [NAMES[c] for c in CONCEPTS], list(zip([s[0] for s in series], SERIES)),
         vals, errs, labels_for=[series[0][0]])
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("Δ CLIP similarity to the concept")
    ax.set_title("Add one feature vs remove twenty (bucket-1 SAE, 3 seeds)", loc="left")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "fig_amplify_vs_erase.png", dpi=160)
    plt.close(fig)


def _e(er, c, k):
    s = er[c]["settings"][k]
    return s["concept_delta"], s["concept_delta_std"]


def fig_amplify_dose(R, out):
    rows = load(R / "outputs" / "steer_amplify.json")["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, c in zip(axes, CONCEPTS):
        rs = sorted([r for r in rows if r["concept"] == c], key=lambda r: r["strength"])
        xs = [0] + [r["strength"] for r in rs]
        for key, name, color in (("concept_delta", "toward the concept", SERIES[0]),
                                 ("prompt_delta", "match to the original prompt", SERIES[1])):
            ys = [0] + [r[key] for r in rs]
            es = [0] + [r[key + "_std"] for r in rs]
            ax.errorbar(xs, ys, yerr=es, color=color, linewidth=2, marker="o",
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        ecolor=color, elinewidth=1, capsize=2, label=name)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(NAMES[c], loc="left")
        ax.set_xticks([0, 1, 2, 4, 8])
        ax.set_xlabel("strength (× typical activation)")
    axes[0].set_ylabel("Δ CLIP similarity")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Amplification dose-response (3 seeds): only Van Gogh rises without "
                 "losing the prompt", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_amplify_dose.png", dpi=160)
    plt.close(fig)


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def labelled_grid(rows, row_labels, col_labels, cell=200, pad=4, left=150, top=34):
    w = left + len(col_labels) * (cell + pad)
    h = top + len(rows) * (cell + pad)
    sheet = Image.new("RGB", (w, h), SURFACE)
    d = ImageDraw.Draw(sheet)
    f = _font(15)
    for j, t in enumerate(col_labels):
        d.text((left + j * (cell + pad) + cell / 2, top / 2), t, fill=INK, font=f, anchor="mm")
    for i, (row, lab) in enumerate(zip(rows, row_labels)):
        y = top + i * (cell + pad)
        d.multiline_text((10, y + cell / 2), lab, fill=INK, font=f, anchor="lm")
        for j, im in enumerate(row):
            sheet.paste(im.convert("RGB").resize((cell, cell)), (left + j * (cell + pad), y))
    return sheet


def img_amplify(R, out):
    g = R / "outputs" / "grids"
    feats = {c: next(g.glob(f"{c}_f*_amplify_base.*")).name.split("_amplify")[0] for c in CONCEPTS}
    rows = [[Image.open(next(g.glob(f"{feats[c]}_amplify_{s}.*"))) for s in
             ("base", "1", "2", "4", "8")] for c in CONCEPTS]
    sheet = labelled_grid(rows, [f"{NAMES[c]}\nfeature\n{feats[c].split('_f')[-1]}"
                                 for c in CONCEPTS],
                          ["unedited", "×1", "×2", "×4", "×8"])
    sheet.save(out / "img_amplify.jpg", quality=88)


def img_erase(R, out, arm="pooled"):
    g = R / "outputs" / "erase_multi" / arm
    keep = {"base": 0, "top-20 → 0": 3, "top-20 → −1": 4, "control:\n20 others → 0": 5}
    rows, labels = [], []
    for c in ("dog", "van_gogh"):
        strip = Image.open(next(g.glob(f"{c}_eval.*")))
        size, pad = 192, 4
        for lab, r in keep.items():
            y = pad + r * (size + pad)
            rows.append([strip.crop((pad + j * (size + pad), y,
                                     pad + j * (size + pad) + size, y + size)) for j in range(4)])
            labels.append(f"{NAMES[c]}\n{lab}" if lab == "base" else lab)
    sheet = labelled_grid(rows, labels, ["prompt 1", "prompt 2", "prompt 3", "prompt 4"],
                          cell=170, left=170)
    sheet.save(out / "img_erase.jpg", quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_loss_recovered(R, out)
    fig_amplify_vs_erase(R, out)
    fig_amplify_dose(R, out)
    img_amplify(R, out)
    img_erase(R, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
