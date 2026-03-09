import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

np.random.seed(42)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT1 = "#7c6aff"
ACCENT2 = "#00d4aa"
ACCENT3 = "#ff6b6b"
ACCENT4 = "#ffd166"
ACCENT5 = "#a8dadc"
WHITE   = "#e8eaf6"
GRAY    = "#4a4e69"
CLASS_COLORS = [ACCENT1, ACCENT2, ACCENT3]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRAY, "text.color": WHITE,
    "axes.labelcolor": WHITE, "xtick.color": WHITE,
    "ytick.color": WHITE, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "monospace",
})

# ── Load Iris (4D → 2D) ───────────────────────────────────────────────────────
from sklearn import datasets
iris   = datasets.load_iris()
X_raw  = iris.data          # (150, 4)
y      = iris.target

# Manual PCA
mean_   = X_raw.mean(axis=0)
X_c     = X_raw - mean_
cov_    = np.cov(X_c.T)
eig_vals_all, eig_vecs_all = np.linalg.eig(cov_)
eig_vecs_all = eig_vecs_all.T
order   = np.argsort(eig_vals_all)[::-1]
eig_vals_all  = eig_vals_all[order]
eig_vecs_all  = eig_vecs_all[order]
evr     = eig_vals_all / eig_vals_all.sum()
cumevr  = np.cumsum(evr)
W       = eig_vecs_all[:2]           # (2, 4)
Z       = X_c @ W.T                  # (150, 2) — final projection

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.05, right=0.97,
                         top=0.90, bottom=0.07,
                         hspace=0.55, wspace=0.40)

ax_raw   = fig.add_subplot(gs[0, 0])   # raw 2-feature scatter (F1 vs F2)
ax_cov   = fig.add_subplot(gs[1, 0])   # covariance heatmap
ax_eig   = fig.add_subplot(gs[0, 1])   # eigenvalue bar chart
ax_evr   = fig.add_subplot(gs[1, 1])   # cumulative EVR
ax_arrow = fig.add_subplot(gs[0, 2])   # principal component arrows on data
ax_proj  = fig.add_subplot(gs[1, 2])   # final 2-D projection

for ax in [ax_raw, ax_cov, ax_eig, ax_evr, ax_arrow, ax_proj]:
    ax.set_facecolor(PANEL)

fig.text(0.5, 0.95, "PCA — Step-by-Step",
         ha="center", fontsize=18, color=WHITE,
         fontweight="bold", fontfamily="monospace")

# ── [1] Raw scatter (feature 0 vs 1) ─────────────────────────────────────────
ax_raw.set_title("[1] Raw Data  (Feature 0 vs Feature 1)", fontsize=9, color=ACCENT4, pad=5)
ax_raw.set_xlabel("Sepal Length", fontsize=8)
ax_raw.set_ylabel("Sepal Width",  fontsize=8)
raw_sc = [ax_raw.scatter([], [], c=CLASS_COLORS[c], s=22, alpha=0.75, label=f"Class {c}")
          for c in range(3)]
ax_raw.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
mean_dot = ax_raw.scatter([], [], color=ACCENT4, marker="+", s=200, zorder=6, linewidths=2)
mean_txt = ax_raw.text(0, 0, "", color=ACCENT4, fontsize=7, ha="left")

# ── [2] Covariance heatmap ────────────────────────────────────────────────────
ax_cov.set_title("[2] Covariance Matrix  cov = X_c.T @ X_c / (n-1)",
                  fontsize=8, color=ACCENT4, pad=5)
cov_img = ax_cov.imshow(np.zeros_like(cov_), cmap="RdYlGn",
                          vmin=cov_.min(), vmax=cov_.max(), alpha=0)
plt.colorbar(cov_img, ax=ax_cov, fraction=0.046, pad=0.04)
ax_cov.set_xticks(range(4)); ax_cov.set_yticks(range(4))
ax_cov.set_xticklabels(["F0","F1","F2","F3"], fontsize=7)
ax_cov.set_yticklabels(["F0","F1","F2","F3"], fontsize=7)
cov_cell_txts = []
for i in range(4):
    row = []
    for j in range(4):
        t = ax_cov.text(j, i, "", ha="center", va="center",
                         fontsize=6.5, color=WHITE, alpha=0)
        row.append(t)
    cov_cell_txts.append(row)

# ── [3] Eigenvalue bar chart ──────────────────────────────────────────────────
ax_eig.set_title("[3] Eigenvalues  (variance explained per PC)",
                  fontsize=8, color=ACCENT4, pad=5)
ax_eig.set_xlabel("Principal Component", fontsize=8)
ax_eig.set_ylabel("Eigenvalue", fontsize=8)
ax_eig.set_xticks(range(4))
ax_eig.set_xticklabels([f"PC{i+1}" for i in range(4)], fontsize=8)
eig_bars = ax_eig.bar(range(4), [0]*4,
                       color=[ACCENT1, ACCENT2, ACCENT5, ACCENT3],
                       edgecolor=WHITE, lw=0.8)
eig_txts = [ax_eig.text(i, 0.05, "", ha="center", fontsize=7, color=WHITE)
             for i in range(4)]

# ── [4] Cumulative EVR ────────────────────────────────────────────────────────
ax_evr.set_title("[4] Cumulative Explained Variance  EVR_i = lam_i / sum(lam)",
                  fontsize=8, color=ACCENT4, pad=5)
ax_evr.set_xlabel("Number of Components", fontsize=8)
ax_evr.set_ylabel("Cumulative EVR", fontsize=8)
ax_evr.set_xlim(0.5, 4.5); ax_evr.set_ylim(0, 1.1)
ax_evr.axhline(y=0.95, color=GRAY, lw=1, ls=":")
ax_evr.text(4.4, 0.96, "95%", fontsize=7, color=GRAY, ha="right")
evr_line, = ax_evr.plot([], [], color=ACCENT2, lw=2.5, marker="o", markersize=6)
evr_dots  = ax_evr.scatter([], [], color=ACCENT4, s=60, zorder=5)

# ── [5] PC arrows on centered data ───────────────────────────────────────────
ax_arrow.set_title("[5] Principal Components on Centered Data",
                    fontsize=8, color=ACCENT4, pad=5)
ax_arrow.set_xlabel("Feature 0 (centered)", fontsize=8)
ax_arrow.set_ylabel("Feature 1 (centered)", fontsize=8)
arrow_sc = [ax_arrow.scatter([], [], c=CLASS_COLORS[c], s=18, alpha=0.55)
             for c in range(3)]
pc_arrows = []   # added dynamically
pc_labels = []

# ── [6] Final 2-D projection ──────────────────────────────────────────────────
ax_proj.set_title("[6] Projection onto 2 PCs  Z = X_c @ W.T",
                   fontsize=9, color=ACCENT4, pad=5)
ax_proj.set_xlabel("PC 1", fontsize=8)
ax_proj.set_ylabel("PC 2", fontsize=8)
proj_sc = [ax_proj.scatter([], [], c=CLASS_COLORS[c], s=22, alpha=0.80,
                             label=f"Class {c}") for c in range(3)]
ax_proj.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)

# ─────────────────────────────────────────────────────────────────────────────
# Animation helpers
# ─────────────────────────────────────────────────────────────────────────────
def ease(v): return max(0.0, min(1.0, float(v)))

TOTAL = 150

def update(frame):
    artists = []

    # Phase 1 (0-20): Raw scatter fades in + mean marker
    if frame <= 20:
        a = ease(frame / 20)
        for c in range(3):
            mask = y == c
            raw_sc[c].set_offsets(X_raw[mask, :2])
            raw_sc[c].set_alpha(a * 0.75)
        if frame >= 12:
            a2 = ease((frame-12)/8)
            mean_dot.set_offsets([[mean_[0], mean_[1]]])
            mean_dot.set_alpha(a2)
            mean_txt.set_position((mean_[0]+0.05, mean_[1]+0.05))
            mean_txt.set_text(f"mean=({mean_[0]:.2f}, {mean_[1]:.2f})")
            mean_txt.set_alpha(a2)
        artists += raw_sc + [mean_dot, mean_txt]

    # Phase 2 (21-45): Covariance heatmap appears
    elif frame <= 45:
        a = ease((frame-21)/24)
        cov_img.set_alpha(a)
        cov_img.set_data(cov_ * a)
        for i in range(4):
            for j in range(4):
                t = cov_cell_txts[i][j]
                t.set_text(f"{cov_[i,j]:.2f}")
                t.set_alpha(a)
        flat_cells = [cov_cell_txts[i][j] for i in range(4) for j in range(4)]
        artists += [cov_img] + flat_cells

    # Phase 3 (46-70): Eigenvalue bars grow
    elif frame <= 70:
        prog = ease((frame-46)/24)
        for i, bar in enumerate(eig_bars):
            bar.set_height(eig_vals_all[i] * prog)
            eig_txts[i].set_text(f"{eig_vals_all[i]*prog:.2f}")
            eig_txts[i].set_position((i, eig_vals_all[i]*prog + 0.05))
        artists += list(eig_bars) + eig_txts

    # Phase 4 (71-90): Cumulative EVR line draws
    elif frame <= 90:
        prog = ease((frame-71)/19)
        n = max(1, int(4 * prog))
        evr_line.set_data(range(1, n+1), cumevr[:n])
        evr_dots.set_offsets([[n, cumevr[n-1]]])
        artists += [evr_line, evr_dots]

    # Phase 5 (91-115): Centered scatter + PC arrows
    elif frame <= 115:
        a = ease((frame-91)/15)
        for c in range(3):
            mask = y == c
            arrow_sc[c].set_offsets(X_c[mask, :2])
            arrow_sc[c].set_alpha(a * 0.55)

        # Draw PC arrows once at frame 106
        if frame >= 106:
            if len(pc_arrows) == 0:
                scale = 3.0
                for i, (vec, col) in enumerate(zip(eig_vecs_all[:2], [ACCENT1, ACCENT2])):
                    arr = FancyArrowPatch(
                        (0, 0),
                        (vec[0]*scale, vec[1]*scale),
                        arrowstyle="-|>",
                        color=col, mutation_scale=14,
                        lw=2.5, zorder=6
                    )
                    ax_arrow.add_patch(arr)
                    pc_arrows.append(arr)
                    lbl = ax_arrow.text(
                        vec[0]*scale*1.12, vec[1]*scale*1.12,
                        f"PC{i+1}", color=col, fontsize=9,
                        ha="center", fontweight="bold"
                    )
                    pc_labels.append(lbl)
            a2 = ease((frame-106)/9)
            for arr in pc_arrows:
                arr.set_alpha(a2)
            for lbl in pc_labels:
                lbl.set_alpha(a2)
        artists += arrow_sc + pc_arrows + pc_labels

    # Phase 6 (116-150): Final projection scatter
    elif frame <= 150:
        prog = ease((frame-116)/34)
        n = max(1, int(len(Z) * prog))
        for c in range(3):
            mask = (y == c) & (np.arange(len(y)) < n)
            if mask.sum() > 0:
                proj_sc[c].set_offsets(Z[mask])
                proj_sc[c].set_alpha(0.85)
        artists += proj_sc

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL,
                    interval=65, blit=False, repeat=False)

writer = PillowWriter(fps=18)
out = "/mnt/user-data/outputs/pca_animation.gif"
ani.save(out, writer=writer, dpi=100)
print(f"GIF saved -> {out}")
plt.close(fig)
