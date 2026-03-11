import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn import datasets
from sklearn.model_selection import train_test_split

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

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRAY, "text.color": WHITE,
    "axes.labelcolor": WHITE, "xtick.color": WHITE,
    "ytick.color": WHITE, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "monospace",
})

# ── Dataset ───────────────────────────────────────────────────────────────────
X, y = datasets.make_blobs(n_samples=50, n_features=2,
                             centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# ── Train SVM, snapshot every N steps ─────────────────────────────────────────
lr       = 0.001
lam      = 0.01
n_iters  = 60          # epochs shown in animation
y_       = np.where(y_train <= 0, -1, 1)

w = np.zeros(2)
b = 0.0
snapshots  = []
hinge_log  = []
margin_log = []

for epoch in range(n_iters):
    for idx, x_i in enumerate(X_train):
        cond = y_[idx] * (np.dot(x_i, w) - b) >= 1
        if cond:
            w -= lr * (2 * lam * w)
        else:
            w -= lr * (2 * lam * w - np.dot(x_i, y_[idx]))
            b -= lr * y_[idx]
    # hinge loss this epoch
    scores = y_ * (X_train @ w - b)
    hl = np.mean(np.maximum(0, 1 - scores))
    margin = 2 / (np.linalg.norm(w) + 1e-9)
    snapshots.append((w.copy(), float(b)))
    hinge_log.append(hl)
    margin_log.append(margin)

final_w, final_b = snapshots[-1]

# ── Helpers ───────────────────────────────────────────────────────────────────
x_lo = X[:, 0].min() - 1
x_hi = X[:, 0].max() + 1
xr   = np.linspace(x_lo, x_hi, 200)

def boundary(w, b, offset=0):
    if abs(w[1]) < 1e-9: return None, None
    return xr, (-w[0] * xr + b + offset) / w[1]

y_lo = X[:, 1].min() - 3
y_hi = X[:, 1].max() + 3

c_pts = [ACCENT1 if yi == -1 else ACCENT2 for yi in y_train]

# hinge curve over z = y*f(x)
z_vals  = np.linspace(-1, 3, 300)
hinge_v = np.maximum(0, 1 - z_vals)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.05, right=0.97,
                         top=0.90, bottom=0.07,
                         hspace=0.55, wspace=0.42)

ax_main  = fig.add_subplot(gs[:, 0])   # main scatter + live margins
ax_hinge = fig.add_subplot(gs[0, 1])   # hinge loss curve
ax_eq    = fig.add_subplot(gs[1, 1])   # equations
ax_loss  = fig.add_subplot(gs[0, 2])   # training loss
ax_margin= fig.add_subplot(gs[1, 2])   # margin width

for ax in [ax_main, ax_hinge, ax_eq, ax_loss, ax_margin]:
    ax.set_facecolor(PANEL)

fig.text(0.5, 0.95, "SVM (Hinge Loss + Gradient Descent) — Step-by-Step",
         ha="center", fontsize=17, color=WHITE,
         fontweight="bold", fontfamily="monospace")

# ── [1] Main: scatter + 3 live hyperplanes ────────────────────────────────────
ax_main.set_title("[1] Max-Margin Decision Boundary (live)", fontsize=9, color=ACCENT4, pad=5)
ax_main.set_xlabel("Feature 0", fontsize=8)
ax_main.set_ylabel("Feature 1", fontsize=8)
ax_main.set_ylim(y_lo, y_hi)
ax_main.scatter(X_train[:, 0], X_train[:, 1], c=c_pts, s=35, alpha=0.85, zorder=3)

db_line,  = ax_main.plot([], [], color=ACCENT4, lw=2.5, ls="--", zorder=5, label="Decision boundary")
pos_line, = ax_main.plot([], [], color=ACCENT2, lw=1.5, zorder=4, label="+1 margin")
neg_line, = ax_main.plot([], [], color=ACCENT1, lw=1.5, zorder=4, label="−1 margin")
margin_fill = [None]
ax_main.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE, loc="upper left")
epoch_lbl = ax_main.text(0.04, 0.96, "", transform=ax_main.transAxes,
                          fontsize=9, color=ACCENT4, va="top")
sv_scatter = ax_main.scatter([], [], s=120, facecolors="none",
                               edgecolors=ACCENT3, linewidths=2, zorder=6, label="Support vectors")

# ── [2] Hinge loss shape ──────────────────────────────────────────────────────
ax_hinge.set_title("[2] Hinge Loss:  L = max(0, 1 - y·f(x))",
                    fontsize=9, color=ACCENT4, pad=5)
ax_hinge.set_xlabel("y · f(x)  (margin score)", fontsize=8)
ax_hinge.set_ylabel("Loss", fontsize=8)
ax_hinge.set_xlim(-1, 3); ax_hinge.set_ylim(-0.1, 2.2)
ax_hinge.axvline(x=1, color=GRAY, lw=1, ls=":")
ax_hinge.text(1.05, 2.0, "margin = 1", fontsize=7, color=GRAY)
hinge_line, = ax_hinge.plot([], [], color=ACCENT3, lw=3)
live_hinge_dot  = ax_hinge.scatter([], [], color=ACCENT4, s=80, zorder=6)
live_hinge_lbl  = ax_hinge.text(0, 0, "", fontsize=7.5, color=ACCENT4)

# ── [3] Equations ─────────────────────────────────────────────────────────────
ax_eq.axis("off")
ax_eq.set_title("[3] Gradient Update Rules", fontsize=9, color=ACCENT4, pad=5)
eq_lines = [
    ("Condition:",       "y*(w.T*x - b) >= 1  ->  correct",   ACCENT2),
    ("  -> w update:",   "w  <-  w - lr * 2*lambda*w",         ACCENT5),
    ("Condition:",       "y*(w.T*x - b) < 1   ->  violation",  ACCENT3),
    ("  -> w update:",   "w  <-  w - lr*(2*lambda*w - y*x)",   ACCENT5),
    ("  -> b update:",   "b  <-  b - lr*(-y)",                 ACCENT5),
    ("Margin width:",    "margin = 2 / ||w||",                  ACCENT4),
]
eq_objs = []
for i, (lbl, eq, col) in enumerate(eq_lines):
    yp = 0.93 - i * 0.145
    t1 = ax_eq.text(0.02, yp, lbl, transform=ax_eq.transAxes,
                     fontsize=7.5, color=GRAY, va="top", alpha=0)
    t2 = ax_eq.text(0.02, yp - 0.05, eq, transform=ax_eq.transAxes,
                     fontsize=8.5, color=col, va="top", alpha=0,
                     fontfamily="monospace")
    eq_objs.append((t1, t2))
live_wb = ax_eq.text(0.02, 0.04, "", transform=ax_eq.transAxes,
                      fontsize=8, color=ACCENT3, va="bottom", alpha=0)

# ── [4] Training hinge loss ────────────────────────────────────────────────────
ax_loss.set_title("[4] Mean Hinge Loss per Epoch", fontsize=9, color=ACCENT4, pad=5)
ax_loss.set_xlabel("Epoch", fontsize=8)
ax_loss.set_ylabel("Hinge Loss", fontsize=8)
ax_loss.set_xlim(0, n_iters)
ax_loss.set_ylim(-0.02, max(hinge_log) * 1.15)
loss_line, = ax_loss.plot([], [], color=ACCENT3, lw=2.5)
loss_dot   = ax_loss.scatter([], [], color=ACCENT4, s=60, zorder=5)

# ── [5] Margin width ──────────────────────────────────────────────────────────
ax_margin.set_title("[5] Margin Width  =  2 / ||w||  (should grow)",
                     fontsize=9, color=ACCENT4, pad=5)
ax_margin.set_xlabel("Epoch", fontsize=8)
ax_margin.set_ylabel("Margin", fontsize=8)
ax_margin.set_xlim(0, n_iters)
ax_margin.set_ylim(0, max(margin_log) * 1.2)
margin_line, = ax_margin.plot([], [], color=ACCENT2, lw=2.5)
margin_dot   = ax_margin.scatter([], [], color=ACCENT4, s=60, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────
TOTAL = 160

def ease(v): return max(0.0, min(1.0, float(v)))

def update(frame):
    artists = []

    # Phase 1 (0-20): Hinge loss curve draws
    if frame <= 20:
        n = max(1, int(len(z_vals) * ease(frame / 20)))
        hinge_line.set_data(z_vals[:n], hinge_v[:n])
        artists.append(hinge_line)

    # Phase 2 (21-42): Equation panel reveals
    elif frame <= 42:
        sub = frame - 21
        for i, (t1, t2) in enumerate(eq_objs):
            a = ease((sub - i * 3.2) / 3)
            t1.set_alpha(a); t2.set_alpha(a)
            artists += [t1, t2]

    # Phase 3 (43-140): Live training
    elif frame <= 140:
        sub   = frame - 43
        epoch = min(int(sub / 1.6), n_iters - 1)
        w_e, b_e = snapshots[epoch]

        # Decision boundary + margins
        _, yr_db  = boundary(w_e, b_e,  0)
        _, yr_pos = boundary(w_e, b_e,  1)
        _, yr_neg = boundary(w_e, b_e, -1)
        if yr_db is not None:
            db_line.set_data(xr, yr_db)
            pos_line.set_data(xr, yr_pos)
            neg_line.set_data(xr, yr_neg)

        # Highlight support vectors (points inside/on margin)
        scores = y_ * (X_train @ w_e - b_e)
        sv_mask = scores <= 1.05
        if sv_mask.sum() > 0:
            sv_scatter.set_offsets(X_train[sv_mask])
        else:
            sv_scatter.set_offsets(np.empty((0, 2)))

        epoch_lbl.set_text(f"Epoch {epoch+1:02d}/{n_iters}  "
                            f"| loss={hinge_log[epoch]:.3f}  "
                            f"| margin={margin_log[epoch]:.3f}")

        # Live hinge dot — use current mean score
        mean_score = float(np.mean(scores))
        lv = max(0, 1 - mean_score)
        live_hinge_dot.set_offsets([[np.clip(mean_score, -0.9, 2.9), lv]])
        live_hinge_lbl.set_text(f"mean={mean_score:.2f}")
        live_hinge_lbl.set_position((np.clip(mean_score + 0.05, -0.8, 2.4), lv + 0.08))

        # Live w/b
        live_wb.set_text(f"w=[{w_e[0]:.3f}, {w_e[1]:.3f}]  b={b_e:.3f}")
        live_wb.set_alpha(1)

        # Loss + margin curves
        loss_line.set_data(range(1, epoch + 2), hinge_log[:epoch + 1])
        loss_dot.set_offsets([[epoch + 1, hinge_log[epoch]]])
        margin_line.set_data(range(1, epoch + 2), margin_log[:epoch + 1])
        margin_dot.set_offsets([[epoch + 1, margin_log[epoch]]])

        artists += [db_line, pos_line, neg_line, sv_scatter,
                    epoch_lbl, live_hinge_dot, live_hinge_lbl,
                    live_wb, loss_line, loss_dot, margin_line, margin_dot]

    # Phase 4 (141-160): Final — solidify & show accuracy
    elif frame <= 160:
        _, yr_db  = boundary(final_w, final_b,  0)
        _, yr_pos = boundary(final_w, final_b,  1)
        _, yr_neg = boundary(final_w, final_b, -1)
        if yr_db is not None:
            db_line.set_data(xr, yr_db); db_line.set_color(ACCENT4); db_line.set_lw(3)
            pos_line.set_data(xr, yr_pos)
            neg_line.set_data(xr, yr_neg)

        preds = np.sign(X_test @ final_w - final_b)
        acc   = np.sum(preds == np.where(y_test <= 0, -1, 1)) / len(y_test)
        epoch_lbl.set_text(f"FINAL  |  Test Acc: {acc*100:.1f}%  "
                            f"|  Margin: {margin_log[-1]:.3f}")

        artists += [db_line, pos_line, neg_line, epoch_lbl,
                    loss_line, margin_line, loss_dot, margin_dot]

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL,
                    interval=65, blit=False, repeat=False)

writer = PillowWriter(fps=18)
out = "/mnt/user-data/outputs/svm_animation.gif"
ani.save(out, writer=writer, dpi=100)
print(f"GIF saved -> {out}")
plt.close(fig)
