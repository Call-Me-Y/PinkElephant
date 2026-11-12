"""Export standard research figures from the completed behavior summary."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(__file__).resolve().parents[1] / "results" / "interventions-batched-fp16-v2"
rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8-sig")))
labels = {
    "drop": "Drop / no feedback", "verbatim": "Verbatim", "abstract": "FTB abstract",
    "positive_history": "Positive history", "ftb_ban": "Verbatim + string ban",
    "abstract_ban": "Abstract + string ban", "local_constraints": "Abstract + local constraints",
    "contrast_025": "Copy contrast (lambda=0.25)", "contrast_05": "Copy contrast (lambda=0.5)",
    "contrast_1": "Copy contrast (lambda=1.0)", "sft_head": "SFT output-head adapter", "ul_head": "SFT + UL output-head adapter",
}
arms = list(labels)
fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)
for axis, metric, title in zip(axes, ("task_success", "first_action_valid"),
                                ("Task success within 3 steps (higher is better)", "First action accepted by tool (higher is better)")):
    points = [next(r for r in rows if r["history_depth"] == "all" and r["arm"] == arm and r["metric"] == metric) for arm in arms]
    mean = np.array([float(r["mean"]) for r in points]) * 100
    lo = np.array([float(r["ci_lo"]) for r in points]) * 100
    hi = np.array([float(r["ci_hi"]) for r in points]) * 100
    y = np.arange(len(arms))
    axis.errorbar(mean, y, xerr=np.vstack((mean - lo, hi - mean)), fmt="o", color="#245b86", capsize=3, linewidth=1.3)
    axis.set_title(title, fontsize=10)
    axis.set_xlim(-2, 102)
    axis.set_xlabel("Percent of trajectories")
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(np.arange(len(arms)), [labels[a] for a in arms])
axes[0].invert_yaxis()
fig.suptitle("Qwen2.5-0.5B: 48 failure checkpoints / 24 task clusters", fontsize=13)
fig.text(0.5, 0.015, "95% task-cluster bootstrap intervals (10,000 resamples). Synthetic failures; exploratory pilot.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(out / "comparison.png", dpi=180)
fig.savefig(out / "comparison.svg")
print(out / "comparison.png")
