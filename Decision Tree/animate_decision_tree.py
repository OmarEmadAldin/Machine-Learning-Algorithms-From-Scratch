import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from collections import Counter

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Colour palette ───────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT1 = "#7c6aff"   # purple
ACCENT2 = "#00d4aa"   # teal
ACCENT3 = "#ff6b6b"   # coral
ACCENT4 = "#ffd166"   # yellow
WHITE   = "#e8eaf6"
GRAY    = "#4a4e69"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   GRAY,
    "text.color":       WHITE,
    "axes.labelcolor":  WHITE,
    "xtick.color":      WHITE,
    "ytick.color":      WHITE,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# Tiny synthetic dataset (2-D, 2-class)
# ─────────────────────────────────────────────────────────────────────────────
N = 60
X0 = np.random.randn(N // 2, 2) + np.array([-1.5,  1.0])
X1 = np.random.randn(N // 2, 2) + np.array([ 1.5, -1.0])
X  = np.vstack([X0, X1])
y  = np.array([0] * (N // 2) + [1] * (N // 2))

# ─────────────────────────────────────────────────────────────────────────────
# Helper maths
# ─────────────────────────────────────────────────────────────────────────────
def entropy(labels):
    if len(labels) == 0:
        return 0.0
    hist = np.bincount(labels, minlength=2)
    ps   = hist / len(labels)
    return float(-np.sum([p * np.log2(p) for p in ps if p > 0]))

def information_gain(y, col, thresh):
    left  = y[col <  thresh]
    right = y[col >= thresh]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    n = len(y)
    return entropy(y) - (len(left)/n * entropy(left) + len(right)/n * entropy(right))

# Best split on feature 0
col      = X[:, 0]
thresholds = np.linspace(col.min() + 0.1, col.max() - 0.1, 80)
gains    = [information_gain(y, col, t) for t in thresholds]
best_idx = int(np.argmax(gains))
best_t   = thresholds[best_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout  (5 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

# Grid:  left half = scatter (tall) | right half = 4 stacked panels
gs = fig.add_gridspec(4, 2, left=0.05, right=0.97,
                      top=0.90, bottom=0.06,
                      hspace=0.55, wspace=0.35)

ax_scatter = fig.add_subplot(gs[:, 0])      # full left column
ax_entropy = fig.add_subplot(gs[0, 1])      # top-right
ax_split   = fig.add_subplot(gs[1, 1])      # 2nd-right
ax_ig      = fig.add_subplot(gs[2, 1])      # 3rd-right
ax_tree    = fig.add_subplot(gs[3, 1])      # bottom-right

for ax in [ax_scatter, ax_entropy, ax_split, ax_ig, ax_tree]:
    ax.set_facecolor(PANEL)

# ── Title ─────────────────────────────────────────────────────────────────
fig.text(0.5, 0.96, "Decision Tree — Step-by-Step",
         ha="center", va="top", fontsize=18, color=WHITE,
         fontweight="bold", fontfamily="monospace")

# ─────────────────────────────────────────────────────────────────────────────
# Static scatter base
# ─────────────────────────────────────────────────────────────────────────────
colors_pt = [ACCENT1 if yi == 0 else ACCENT2 for yi in y]
ax_scatter.scatter(X[:, 0], X[:, 1], c=colors_pt, s=40, alpha=0.85, zorder=3)
ax_scatter.set_xlabel("Feature 0", fontsize=9)
ax_scatter.set_ylabel("Feature 1", fontsize=9)
ax_scatter.set_title("Dataset  (2 classes)", fontsize=10, color=ACCENT4, pad=6)
patch0 = mpatches.Patch(color=ACCENT1, label="Class 0")
patch1 = mpatches.Patch(color=ACCENT2, label="Class 1")
ax_scatter.legend(handles=[patch0, patch1], fontsize=8,
                  facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)

split_line = ax_scatter.axvline(x=best_t, color=ACCENT3, lw=2.5,
                                 linestyle="--", alpha=0, zorder=5)
split_txt  = ax_scatter.text(best_t + 0.05, X[:, 1].max() - 0.3,
                              f"x < {best_t:.2f}", color=ACCENT3,
                              fontsize=9, alpha=0, zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 – Entropy curve
# ─────────────────────────────────────────────────────────────────────────────
ps_range = np.linspace(0.001, 0.999, 200)
ent_vals  = [-p * np.log2(p) - (1-p) * np.log2(1-p) for p in ps_range]

ax_entropy.set_xlim(0, 1); ax_entropy.set_ylim(0, 1.15)
ax_entropy.set_xlabel("p₁  (fraction of class 1)", fontsize=8)
ax_entropy.set_ylabel("H  (bits)", fontsize=8)
ax_entropy.set_title("[1] Entropy:  H = −Σ pᵢ log₂(pᵢ)", fontsize=9,
                      color=ACCENT4, pad=5)
ent_line,  = ax_entropy.plot([], [], color=ACCENT1, lw=2)
ent_dot    = ax_entropy.scatter([], [], color=ACCENT3, s=60, zorder=5)
ent_label  = ax_entropy.text(0.5, 0.05, "", ha="center", fontsize=8, color=ACCENT3)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 – Split visualisation (bar chart of left / right counts)
# ─────────────────────────────────────────────────────────────────────────────
ax_split.set_title("[2] Split:  x < threshold → Left  |  ≥ → Right",
                    fontsize=9, color=ACCENT4, pad=5)
ax_split.set_xticks([])
bar_containers = []   # filled during animation

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 – Information-Gain curve
# ─────────────────────────────────────────────────────────────────────────────
ax_ig.set_xlim(thresholds[0], thresholds[-1])
ax_ig.set_ylim(-0.02, max(gains) * 1.25)
ax_ig.set_xlabel("Threshold  (feature 0)", fontsize=8)
ax_ig.set_ylabel("IG  (bits)", fontsize=8)
ax_ig.set_title("[3] Info Gain:  IG = H(parent) − Σ wₖ H(child_k)",
                 fontsize=9, color=ACCENT4, pad=5)
ig_line, = ax_ig.plot([], [], color=ACCENT2, lw=2)
ig_dot   = ax_ig.scatter([], [], color=ACCENT3, s=80, zorder=5)
ig_best  = ax_ig.axvline(x=best_t, color=ACCENT4, lw=1.5,
                          linestyle=":", alpha=0)
ig_label = ax_ig.text(best_t, max(gains) * 1.05, "", color=ACCENT4,
                       fontsize=8, ha="center")

# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 – Tree diagram (3-node sketch)
# ─────────────────────────────────────────────────────────────────────────────
ax_tree.set_xlim(0, 10); ax_tree.set_ylim(0, 4)
ax_tree.axis("off")
ax_tree.set_title("[4] Leaf prediction:  ŷ = argmax |{yᵢ = c}|",
                   fontsize=9, color=ACCENT4, pad=5)

node_patches = []
node_texts   = []
node_arrows  = []

def add_node(ax, cx, cy, w, h, txt, color, alpha=0):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=WHITE,
                          linewidth=1.2, alpha=alpha, zorder=4)
    ax.add_patch(box)
    t = ax.text(cx, cy, txt, ha="center", va="center",
                fontsize=7.5, color=WHITE, zorder=5, alpha=alpha,
                fontfamily="monospace")
    return box, t

root_box, root_txt = add_node(ax_tree, 5, 3.2, 3.2, 0.7,
                               f"x₀ < {best_t:.2f} ?", ACCENT1)
left_box, left_txt = add_node(ax_tree, 2.5, 1.4, 2.4, 0.7,
                               "Leaf → Class 0", ACCENT1)
rght_box, rght_txt = add_node(ax_tree, 7.5, 1.4, 2.4, 0.7,
                               "Leaf → Class 1", ACCENT2)

arr_left  = FancyArrowPatch((3.4, 2.85), (2.5, 1.75),
                             arrowstyle="-|>", color=WHITE,
                             mutation_scale=12, lw=1.2, alpha=0, zorder=3)
arr_right = FancyArrowPatch((6.6, 2.85), (7.5, 1.75),
                             arrowstyle="-|>", color=WHITE,
                             mutation_scale=12, lw=1.2, alpha=0, zorder=3)
ax_tree.add_patch(arr_left)
ax_tree.add_patch(arr_right)
ax_tree.text(3.5, 2.4, "True", fontsize=7, color=ACCENT2, alpha=0).set_visible(False)
lbl_true  = ax_tree.text(3.0, 2.28, "True",  fontsize=7.5, color=ACCENT2, alpha=0)
lbl_false = ax_tree.text(6.8, 2.28, "False", fontsize=7.5, color=ACCENT3, alpha=0)

all_nodes  = [root_box, root_txt, left_box, left_txt, rght_box, rght_txt]
all_arrows = [arr_left, arr_right, lbl_true, lbl_false]

# ─────────────────────────────────────────────────────────────────────────────
# Animation frames
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_FRAMES = 120

def update(frame):
    artists = []

    # ── Phase 1 (0-25): Draw entropy curve ───────────────────────────────
    if frame <= 25:
        n = max(1, int(len(ps_range) * frame / 25))
        ent_line.set_data(ps_range[:n], ent_vals[:n])
        # current class-1 fraction in full dataset
        p1 = y.mean()
        h_full = entropy(y)
        if frame == 25:
            ent_dot.set_offsets([[p1, h_full]])
            ent_label.set_text(f"Dataset: p₁={p1:.2f}, H={h_full:.3f} bits")
        artists += [ent_line, ent_dot, ent_label]

    # ── Phase 2 (26-50): Show split bar chart ────────────────────────────
    elif frame <= 50:
        if frame == 26:
            # clear old bars
            for c in bar_containers:
                c.remove()
            bar_containers.clear()
            ax_split.cla()
            ax_split.set_facecolor(PANEL)
            ax_split.set_title("[2] Split:  x < threshold → Left  |  ≥ → Right",
                                fontsize=9, color=ACCENT4, pad=5)
            ax_split.spines["top"].set_visible(False)
            ax_split.spines["right"].set_visible(False)
            ax_split.tick_params(colors=WHITE)
            ax_split.yaxis.label.set_color(WHITE)

            left_mask  = col < best_t
            right_mask = ~left_mask
            lc = np.bincount(y[left_mask],  minlength=2)
            rc = np.bincount(y[right_mask], minlength=2)

            x_pos  = np.arange(2)
            labels = ["Class 0", "Class 1"]
            w = 0.3
            b1 = ax_split.bar(x_pos - w/2, lc, w, label="Left  (x < t)",
                               color=ACCENT1, alpha=0.85, edgecolor=WHITE, lw=0.8)
            b2 = ax_split.bar(x_pos + w/2, rc, w, label="Right (x ≥ t)",
                               color=ACCENT2, alpha=0.85, edgecolor=WHITE, lw=0.8)
            ax_split.set_xticks(x_pos); ax_split.set_xticklabels(labels, fontsize=8)
            ax_split.set_ylabel("Count", fontsize=8)
            ax_split.legend(fontsize=7, facecolor=PANEL,
                             edgecolor=GRAY, labelcolor=WHITE)

            h_left  = entropy(y[left_mask])
            h_right = entropy(y[right_mask])
            ax_split.text(0.5, -0.22,
                           f"H(left)={h_left:.3f}  |  H(right)={h_right:.3f}",
                           transform=ax_split.transAxes,
                           ha="center", fontsize=8, color=ACCENT4)
            bar_containers.extend([b1, b2])

        # Reveal split line on scatter
        alpha_val = min(1.0, (frame - 26) / 12)
        split_line.set_alpha(alpha_val)
        split_txt.set_alpha(alpha_val)
        artists += [split_line, split_txt]

    # ── Phase 3 (51-80): Draw IG curve & mark best ───────────────────────
    elif frame <= 80:
        prog = (frame - 51) / 29
        n    = max(1, int(len(thresholds) * prog))
        ig_line.set_data(thresholds[:n], gains[:n])
        artists.append(ig_line)
        if frame >= 75:
            ig_best.set_alpha(min(1.0, (frame - 75) / 5))
            ig_dot.set_offsets([[best_t, max(gains)]])
            ig_label.set_text(f"Best t={best_t:.2f}\nIG={max(gains):.4f}")
            artists += [ig_best, ig_dot, ig_label]

    # ── Phase 4 (81-120): Reveal tree nodes ──────────────────────────────
    elif frame <= 120:
        sub = frame - 81
        # root appears 0-15
        a_root = min(1.0, sub / 15)
        root_box.set_alpha(a_root); root_txt.set_alpha(a_root)
        # arrows + leaves appear 16-35
        if sub >= 16:
            a_leaves = min(1.0, (sub - 16) / 20)
            for obj in all_arrows:
                obj.set_alpha(a_leaves)
            for obj in [left_box, left_txt, rght_box, rght_txt]:
                obj.set_alpha(a_leaves)
        artists += all_nodes + all_arrows

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                    interval=60, blit=False, repeat=False)

# ── Save GIF ─────────────────────────────────────────────────────────────────
writer = PillowWriter(fps=18)
gif_path = "/mnt/user-data/outputs/decision_tree_animation.gif"
ani.save(gif_path, writer=writer, dpi=100)
print(f"✅  GIF saved → {gif_path}")

plt.close(fig)
