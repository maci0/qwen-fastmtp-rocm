#!/usr/bin/env python3
"""Generate benchmark charts for qwen-fastmtp-rocm."""

import matplotlib.pyplot as plt
import numpy as np

# Styling configuration
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#30363d"
plt.rcParams["axes.linewidth"] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0d1117")

# Chart 1: Decode Generation Speed (tokens/s)
categories = [
    "Baseline\n(No MTP)",
    "FastMTP\n(Reasoning)",
    "FastMTP\n(Code / Doc)",
    "FastMTP Boost\n(Jaro Video)",
    "FastMTP Peak\n(Burst)",
]
speeds = [34.3, 48.0, 70.3, 110.0, 151.0]
colors = ["#484f58", "#58a6ff", "#3fb950", "#2ea043", "#bc8cff"]

ax1.set_facecolor("#161b22")
bars1 = ax1.bar(categories, speeds, color=colors, width=0.55, edgecolor="#30363d", zorder=3)
ax1.grid(axis="y", linestyle="--", alpha=0.25, color="#8b949e", zorder=0)
ax1.set_ylabel("Tokens / Second (Decode)", color="#c9d1d9", fontsize=11, labelpad=8)
ax1.set_title("Decode Throughput on AMD Radeon RX 7900 XTX (24 GB)", color="#f0f6fc", fontsize=12, pad=12, fontweight="bold")
ax1.tick_params(colors="#8b949e", labelsize=9.5)
ax1.set_ylim(0, 175)

for bar, speed in zip(bars1, speeds):
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 3.0,
        f"{speed:.1f} t/s",
        ha="center",
        va="bottom",
        color="#f0f6fc",
        fontsize=10,
        fontweight="bold",
    )

# Annotation for speedup
ax1.annotate(
    "3.2x Throughput",
    xy=(3, 110),
    xytext=(2.2, 130),
    arrowprops=dict(facecolor="#58a6ff", edgecolor="#58a6ff", arrowstyle="->", lw=1.5),
    color="#58a6ff",
    fontsize=10,
    fontweight="bold",
)

# Chart 2: Effective Memory Bandwidth Demand per Token
# At ~960 GB/s physical memory bandwidth:
# Baseline reads 17.9 GB per token (960 / 17.9 = ~53.6 t/s theoretical max)
# FastMTP depth 3 with ~89% acceptance reads ~4.86 GB per token (effective bandwidth ~2,200 GB/s)
bandwidth_cats = [
    "Target Model\n(Uncached Step)",
    "FastMTP Step\n(Depth 3 Verified)",
    "Theoretical\nPhysical Limit",
]
bandwidths = [583, 1260, 960]
b_colors = ["#484f58", "#3fb950", "#8b949e"]

ax2.set_facecolor("#161b22")
bars2 = ax2.bar(bandwidth_cats, bandwidths, color=b_colors, width=0.5, edgecolor="#30363d", zorder=3)
ax2.grid(axis="y", linestyle="--", alpha=0.25, color="#8b949e", zorder=0)
ax2.set_ylabel("Effective Bandwidth (GB/s)", color="#c9d1d9", fontsize=11, labelpad=8)
ax2.set_title("Effective Memory Bandwidth (960 GB/s Physical)", color="#f0f6fc", fontsize=12, pad=12, fontweight="bold")
ax2.tick_params(colors="#8b949e", labelsize=9.5)
ax2.set_ylim(0, 1500)

for bar, bw in zip(bars2, bandwidths):
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 25.0,
        f"{bw} GB/s",
        ha="center",
        va="bottom",
        color="#f0f6fc",
        fontsize=10,
        fontweight="bold",
    )

ax2.axhline(960, color="#f85149", linestyle=":", linewidth=1.2, zorder=2)
ax2.text(0.5, 980, "RX 7900 XTX 960 GB/s Ceiling", color="#f85149", fontsize=9)

plt.tight_layout()
plt.savefig("/home/maci/Desktop/Projects/qwen-fastmtp-rocm/assets/benchmark_comparison.png", dpi=200, facecolor=fig.get_facecolor())
print("Saved assets/benchmark_comparison.png")
