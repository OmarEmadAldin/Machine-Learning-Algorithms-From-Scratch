import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
ACCENT5 = "#a8dadc"   # light blue
WHITE   = "#e8eaf6"
GRAY    = "#4a4e69"
TREE_COLORS = [ACCENT1, ACCENT2, ACCENT5, "#f4a261", "#e76f51"]

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
N   = 80
X0  = np.random.randn(N // 2, 2) + np.array([-1.5,  1.0])
X1  = np.random.randn(N // 2, 2) + np.array([ 1.5, -1.0])
X   = np.vstack([X0, X1])
y   = np.array([0] * (N // 2) + [1] * (N // 2))
n   = len(y)

# ── Bootstrap samples (5 trees) ──────────────────────────────────────────────
N_TREES = 5
boots = []
for _ in range(N_TREES):
    idx = np.random.choice(n, size=n, replace=True)
    boots.append(idx)

# ── Helpers ───────────────────────────────────────────────────────────────────
def entropy(labels):
    if len(labels) == 0: return 0.0
    hist = np.bincount(labels, minlength=2)
    ps   = hist / len(labels)
    return float(-np.sum([p * np.log(p) for p in ps if p > 0]))

def ig(y, col, thresh):
    l = y[col <  thresh]; r = y[col >= thresh]
    if not len(l) or not len(r): return 0.0
    return entropy(y) - (len(l)/len(y)*entropy(l) + len(r)/len(y)*entropy(r))

col        = X[:, 0]
thresholds = np.linspace(col.min()+0.1, col.max()-0.1, 80)
gains      = [ig(y, col, t) for t in thresholds]
best_t     = thresholds[int(np.argmax(gains))]

# ── Simulate simple per-tree predictions ─────────────────────────────────────
# Each tree uses its bootstrap sample's majority near a slightly varied threshold
tree_thresholds = [best_t + np.random.uniform(-0.3, 0.3) for _ in range(N_TREES)]
test_points     = np.array([[-2.5, 0.5], [0.0, 0.0], [2.5, -0.5]])
tree_votes      = []   # shape: (N_TREES, n_test)
for t_idx in range(N_TREES):
    preds = (test_points[:, 0] >= tree_thresholds[t_idx]).astype(int)
    tree_votes.append(preds)
tree_votes  = np.array(tree_votes)           # (5, 3)
final_preds = [Counter(tree_votes[:, i]).most_common(1)[0][0] for i in range(3)]

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout  (6 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(3, 3,
                      left=0.04, right=0.97,
                      top=0.90, bottom=0.06,
                      hspace=0.60, wspace=0.38)

ax_data    = fig.add_subplot(gs[:, 0])       # full left – original data
ax_boot    = fig.add_subplot(gs[0, 1])       # bootstrap panel
ax_ig      = fig.add_subplot(gs[1, 1])       # IG curve
ax_trees   = fig.add_subplot(gs[2, 1])       # tree silhouettes
ax_vote    = fig.add_subplot(gs[0, 2])       # voting matrix
ax_var     = fig.add_subplot(gs[1:, 2])      # variance reduction

for ax in [ax_data, ax_boot, ax_ig, ax_trees, ax_vote, ax_var]:
    ax.set_facecolor(PANEL)

fig.text(0.5, 0.96, "Random Forest — Step-by-Step",
         ha="center", va="top", fontsize=18,
         color=WHITE, fontweight="bold", fontfamily="monospace")

# ── Panel 0: static scatter ───────────────────────────────────────────────────
c_pts = [ACCENT1 if yi == 0 else ACCENT2 for yi in y]
ax_data.scatter(X[:, 0], X[:, 1], c=c_pts, s=35, alpha=0.80, zorder=3)
ax_data.set_xlabel("Feature 0", fontsize=9)
ax_data.set_ylabel("Feature 1", fontsize=9)
ax_data.set_title("Full Training Dataset", fontsize=10, color=ACCENT4, pad=6)
ax_data.legend(handles=[mpatches.Patch(color=ACCENT1, label="Class 0"),
                         mpatches.Patch(color=ACCENT2, label="Class 1")],
               fontsize=8, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)

boot_scatters = []   # will hold per-tree scatter artists on ax_data
split_lines   = []   # per-tree vertical lines

# ── Panel 1: bootstrap title & dynamic content ────────────────────────────────
ax_boot.set_title("[1] Bootstrap Sampling  (Bagging)", fontsize=9, color=ACCENT4, pad=5)
ax_boot.axis("off")
boot_text = ax_boot.text(0.5, 0.85, "", transform=ax_boot.transAxes,
                          ha="center", va="top", fontsize=9,
                          color=WHITE, fontfamily="monospace")
boot_eq   = ax_boot.text(0.5, 0.55,
                          "D_b ~ Uniform({1..n})\n    with replacement",
                          transform=ax_boot.transAxes,
                          ha="center", va="top", fontsize=9,
                          color=ACCENT4, fontfamily="monospace", alpha=0)
boot_bars = []

# ── Panel 2: IG curve ─────────────────────────────────────────────────────────
ax_ig.set_xlim(thresholds[0], thresholds[-1])
ax_ig.set_ylim(-0.02, max(gains) * 1.3)
ax_ig.set_xlabel("Threshold (feature 0)", fontsize=8)
ax_ig.set_ylabel("IG (nats)", fontsize=8)
ax_ig.set_title("[2] Best Split: IG = H(parent) - sum(w_k H(child_k))",
                fontsize=8, color=ACCENT4, pad=5)
ig_line, = ax_ig.plot([], [], color=ACCENT2, lw=2)
ig_dot   = ax_ig.scatter([], [], color=ACCENT3, s=80, zorder=5)
ig_vline = ax_ig.axvline(x=best_t, color=ACCENT4, lw=1.5, ls=":", alpha=0)
ig_lbl   = ax_ig.text(best_t, max(gains) * 1.1, "", color=ACCENT4,
                       fontsize=8, ha="center")

# ── Panel 3: tree silhouettes ─────────────────────────────────────────────────
ax_trees.set_xlim(0, 10); ax_trees.set_ylim(0, 4); ax_trees.axis("off")
ax_trees.set_title("[3] T Independent Trees (random feature subsets)",
                    fontsize=8, color=ACCENT4, pad=5)
tree_patches = []
tree_labels  = []
for i in range(N_TREES):
    cx = 1 + i * 2.0
    box = FancyBboxPatch((cx - 0.7, 0.5), 1.4, 2.8,
                          boxstyle="round,pad=0.1",
                          facecolor=TREE_COLORS[i], edgecolor=WHITE,
                          linewidth=1.2, alpha=0, zorder=3)
    ax_trees.add_patch(box)
    txt = ax_trees.text(cx, 1.9, f"Tree\n{i+1}", ha="center", va="center",
                         fontsize=8, color=WHITE, alpha=0, zorder=4,
                         fontfamily="monospace")
    tree_patches.append(box)
    tree_labels.append(txt)

# ── Panel 4: voting matrix ────────────────────────────────────────────────────
ax_vote.set_title("[4] Majority Vote:  y = argmax sum(1[y_t = c])",
                   fontsize=8, color=ACCENT4, pad=5)
ax_vote.set_xlim(-0.5, N_TREES - 0.5)
ax_vote.set_ylim(-0.5, 2.5)
ax_vote.set_xticks(range(N_TREES))
ax_vote.set_xticklabels([f"T{i+1}" for i in range(N_TREES)], fontsize=7)
ax_vote.set_yticks(range(3))
ax_vote.set_yticklabels([f"pt {i+1}" for i in range(3)], fontsize=7)
ax_vote.set_xlabel("Tree", fontsize=8)
ax_vote.set_ylabel("Test Point", fontsize=8)
vote_cells = {}
for ti in range(N_TREES):
    for pi in range(3):
        cell = ax_vote.text(ti, pi, "", ha="center", va="center",
                             fontsize=9, color=WHITE, alpha=0,
                             fontfamily="monospace")
        vote_cells[(ti, pi)] = cell
final_vote_texts = []
for pi in range(3):
    t = ax_vote.text(N_TREES - 0.1, pi, "", ha="left", va="center",
                      fontsize=8, color=ACCENT4, alpha=0)
    final_vote_texts.append(t)

# ── Panel 5: variance reduction curve ────────────────────────────────────────
T_range  = np.arange(1, 51)
rho, sig2 = 0.3, 1.0
var_rf   = rho * sig2 + (1 - rho) / T_range * sig2
var_single = np.ones_like(T_range) * sig2

ax_var.set_xlim(1, 50); ax_var.set_ylim(0, 1.1)
ax_var.set_xlabel("Number of Trees (T)", fontsize=8)
ax_var.set_ylabel("Variance", fontsize=8)
ax_var.set_title("[5] Variance: rho*s2 + (1-rho)/T * s2",
                  fontsize=8, color=ACCENT4, pad=5)
var_single_line = ax_var.axhline(y=1.0, color=ACCENT3, lw=1.5,
                                  ls="--", label="Single Tree")
var_rf_line, = ax_var.plot([], [], color=ACCENT2, lw=2.5, label="Random Forest")
ax_var.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
var_dot = ax_var.scatter([], [], color=ACCENT4, s=60, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────
TOTAL = 150

def update(frame):
    artists = []

    # ── Phase 1 (0-30): Bootstrap sampling ───────────────────────────────
    if frame <= 30:
        tree_i = min(int(frame / 6), N_TREES - 1)
        boot_text.set_text(f"Sampling bootstrap set {tree_i+1} / {N_TREES}\n"
                           f"n={n} rows, with replacement (~63% unique)")
        boot_eq.set_alpha(min(1.0, frame / 10))

        # Show highlighted scatter for current bootstrap on ax_data
        for sc in boot_scatters:
            sc.remove()
        boot_scatters.clear()
        idxs = boots[tree_i]
        sc = ax_data.scatter(X[idxs, 0], X[idxs, 1],
                              c=TREE_COLORS[tree_i],
                              s=18, alpha=0.5, zorder=2, marker="o",
                              linewidths=0)
        boot_scatters.append(sc)
        artists += [boot_text, boot_eq] + boot_scatters

    # ── Phase 2 (31-60): IG sweep ─────────────────────────────────────────
    elif frame <= 60:
        prog = (frame - 31) / 29
        n_pts = max(1, int(len(thresholds) * prog))
        ig_line.set_data(thresholds[:n_pts], gains[:n_pts])
        artists.append(ig_line)
        if frame >= 56:
            a = min(1.0, (frame - 56) / 4)
            ig_vline.set_alpha(a)
            ig_dot.set_offsets([[best_t, max(gains)]])
            ig_lbl.set_text(f"best t={best_t:.2f}")
            artists += [ig_vline, ig_dot, ig_lbl]

    # ── Phase 3 (61-90): Grow tree silhouettes ────────────────────────────
    elif frame <= 90:
        trees_done = int((frame - 61) / 6)
        for i in range(min(trees_done + 1, N_TREES)):
            a = min(1.0, (frame - 61 - i * 6) / 4)
            tree_patches[i].set_alpha(a)
            tree_labels[i].set_alpha(a)
            # add split line on scatter
            if a >= 0.5 and len(split_lines) <= i:
                vl = ax_data.axvline(x=tree_thresholds[i],
                                      color=TREE_COLORS[i],
                                      lw=1.5, ls="--", alpha=0.65, zorder=4)
                split_lines.append(vl)
        artists += tree_patches + tree_labels + split_lines

    # ── Phase 4 (91-120): Voting matrix ───────────────────────────────────
    elif frame <= 120:
        sub = frame - 91
        # reveal one tree column at a time
        trees_shown = min(int(sub / 6) + 1, N_TREES)
        for ti in range(trees_shown):
            a = min(1.0, (sub - ti * 6) / 4)
            for pi in range(3):
                pred = tree_votes[ti, pi]
                cell = vote_cells[(ti, pi)]
                cell.set_text(str(pred))
                cell.set_color(ACCENT2 if pred == 1 else ACCENT1)
                cell.set_alpha(a)
        # Show final vote when all trees revealed
        if trees_shown == N_TREES:
            a_final = max(0.0, min(1.0, (sub - N_TREES * 6) / 5))
            for pi, t in enumerate(final_vote_texts):
                t.set_text(f"-> {final_preds[pi]}")
                t.set_alpha(a_final)
        artists += list(vote_cells.values()) + final_vote_texts

    # ── Phase 5 (121-150): Variance reduction ────────────────────────────
    elif frame <= 150:
        prog  = (frame - 121) / 29
        n_pts = max(1, int(len(T_range) * prog))
        var_rf_line.set_data(T_range[:n_pts], var_rf[:n_pts])
        var_dot.set_offsets([[T_range[n_pts-1], var_rf[n_pts-1]]])
        artists += [var_rf_line, var_single_line, var_dot]

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL,
                    interval=65, blit=False, repeat=False)

writer  = PillowWriter(fps=18)
out_path = "/mnt/user-data/outputs/random_forest_animation.gif"
ani.save(out_path, writer=writer, dpi=100)
print(f"GIF saved -> {out_path}")
plt.close(fig)
