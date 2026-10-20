import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_path = Path("output/results.json")
with results_path.open(encoding="utf-8") as f:
    results = json.load(f)

base = results["runs"]["base"]
sft = results["runs"]["sft"]
dpo = sorted(results["runs"]["dpo"], key=lambda run: run["beta"])
betas = [run["beta"] for run in dpo]

fig, (behavior_ax, mmlu_ax) = plt.subplots(
    2, 1,
    figsize=(8, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

# DPO behavior rates, with Base and SFT reference lines.
behavior_ax.plot(
    betas, [run["ask_rate"] * 100 for run in dpo],
    marker="o", label="DPO ask-rate (ambiguous)",
)
behavior_ax.plot(
    betas, [run["answer_rate"] * 100 for run in dpo],
    marker="s", label="DPO answer-rate (clear)",
)

for model_name, metrics, style in (
    ("Base", base, "--"),
    ("SFT", sft, ":"),
):
    behavior_ax.axhline(
        metrics["ask_rate"] * 100,
        linestyle=style,
        alpha=0.8,
        label=f"{model_name} ask-rate ({metrics['ask_rate']:.0%})",
    )
    behavior_ax.axhline(
        metrics["answer_rate"] * 100,
        linestyle=style,
        alpha=0.8,
        label=f"{model_name} answer-rate ({metrics['answer_rate']:.0%})",
    )

behavior_ax.set_ylabel("Rate (%)")
behavior_ax.set_ylim(0, 105)
behavior_ax.set_title("DPO beta sweep on the development sets")
behavior_ax.grid(alpha=0.25)
behavior_ax.legend(fontsize=8, ncol=2)

# MMLU accuracy. Skip plotting if this evaluation has null MMLU values.
if base["mmlu_acc"] is not None and sft["mmlu_acc"] is not None and all(
    run["mmlu_acc"] is not None for run in dpo
):
    mmlu_ax.plot(
        betas, [run["mmlu_acc"] * 100 for run in dpo],
        marker="o", label="DPO MMLU accuracy",
    )
    mmlu_ax.axhline(
        base["mmlu_acc"] * 100, linestyle="--",
        label=f"Base ({base['mmlu_acc']:.1%})",
    )
    mmlu_ax.axhline(
        sft["mmlu_acc"] * 100, linestyle=":",
        label=f"SFT ({sft['mmlu_acc']:.1%})",
    )
    mmlu_ax.set_ylabel("MMLU accuracy (%)")
    mmlu_ax.legend(fontsize=8)
    mmlu_ax.grid(alpha=0.25)
else:
    mmlu_ax.text(
        0.5, 0.5, "MMLU not evaluated",
        ha="center", va="center",
        transform=mmlu_ax.transAxes,
    )
    mmlu_ax.set_yticks([])
    mmlu_ax.set_ylabel("MMLU")

mmlu_ax.set_xlabel("DPO beta")
mmlu_ax.set_xticks(betas)

sizes = results["eval_sets"]
fig.text(
    0.5, 0.01,
    f"Dev set sizes: ambiguous n={sizes['ambiguous']['n']}, "
    f"clear n={sizes['clear']['n']}, MMLU n={sizes['mmlu']['n']}",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.04, 1, 1))

output_path = Path("output/beta_sweep.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {output_path}")