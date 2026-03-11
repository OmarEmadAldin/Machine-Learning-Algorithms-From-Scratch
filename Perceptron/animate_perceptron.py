import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

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
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, Y = datasets.make_blobs(n_samples=150, n_features=2,
                             centers=2, cluster_std=1.05, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      train_size=0.8,
                                                      random_state=47)

# ── Train perceptron, record snapshots every epoch ────────────────────────────
def unit_step(x):
    return np.where(x > 0, 1, 0)

def decision_line(weights, bias, x_range):
    if abs(weights[1]) < 1e-9:
        return None, None
    x1 = (-weights[0] * x_range - bias) / weights[1]
    return x_range, x1

lr       = 0.1
n_iters  = 40   # keep animation manageable
w        = np.zeros(2)
b        = 0.0
y_       = np.where(y_train > 0, 1, 0)

snapshots   = []   # (w_copy, b, n_mistakes)
mistake_log = []

for epoch in range(n_iters):
    mistakes = 0
    for idx, x_i in enumerate(X_train):
        z    = np.dot(x_i, w) + b
        yhat = unit_step(z)
        upd  = lr * (y_[idx] - yhat)
        w   += upd * x_i
        b   += upd
        if upd != 0:
            mistakes += 1
    snapshots.append((w.copy(), float(b), mistakes))
    mistake_log.append(mistakes)

# final weights
final_w, final_b = snapshots[-1][0], snapshots[-1][1]

# x range for boundary lines
x_lo = X_train[:, 0].min() - 1
x_hi = X_train[:, 0].max() + 1
x_range = np.linspace(x_lo, x_hi, 200)

# ── Activation curve ─────────────────────────────────────────────────────────
z_vals   = np.linspace(-6, 6, 300)
step_out = np.where(z_vals > 0, 1, 0)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.05, right=0.97,
                         top=0.90, bottom=0.07,
                         hspace=0.55, wspace=0.40)

ax_data   = fig.add_subplot(gs[:, 0])   # decision boundary (full height)
ax_neuron = fig.add_subplot(gs[0, 1])   # neuron diagram / activation
ax_update = fig.add_subplot(gs[1, 1])   # weight update equation panel
ax_bound  = fig.add_subplot(gs[0, 2])   # boundary evolution over epochs
ax_err    = fig.add_subplot(gs[1, 2])   # mistake count per epoch

for ax in [ax_data, ax_neuron, ax_update, ax_bound, ax_err]:
    ax.set_facecolor(PANEL)

fig.text(0.5, 0.95, "Perceptron — Step-by-Step",
         ha="center", fontsize=18, color=WHITE,
         fontweight="bold", fontfamily="monospace")

# ── [1] Main scatter + live decision boundary ─────────────────────────────────
ax_data.set_title("[1] Data & Decision Boundary (live)", fontsize=9, color=ACCENT4, pad=5)
ax_data.set_xlabel("Feature 0", fontsize=8)
ax_data.set_ylabel("Feature 1", fontsize=8)

c_pts = [ACCENT1 if yi == 0 else ACCENT2 for yi in y_train]
ax_data.scatter(X_train[:, 0], X_train[:, 1], c=c_pts, s=30, alpha=0.85, zorder=3)

ymin_d = X_train[:, 1].min() - 3
ymax_d = X_train[:, 1].max() + 3
ax_data.set_ylim(ymin_d, ymax_d)

db_line, = ax_data.plot([], [], color=ACCENT3, lw=2.5, zorder=5)
epoch_txt = ax_data.text(0.04, 0.96, "", transform=ax_data.transAxes,
                          fontsize=9, color=ACCENT4, va="top")
highlight_dot = ax_data.scatter([], [], color=ACCENT4, s=120, zorder=6,
                                 edgecolors=WHITE, linewidths=1.5)

# ── [2] Activation function ────────────────────────────────────────────────────
ax_neuron.set_title("[2] Activation: g(z) = 1 if z > 0 else 0",
                     fontsize=9, color=ACCENT4, pad=5)
ax_neuron.set_xlabel("z  (linear output)", fontsize=8)
ax_neuron.set_ylabel("g(z)", fontsize=8)
ax_neuron.set_xlim(-6, 6); ax_neuron.set_ylim(-0.1, 1.3)
act_line, = ax_neuron.plot([], [], color=ACCENT2, lw=3)
ax_neuron.axvline(x=0, color=GRAY, lw=1, ls=":")
ax_neuron.text(0.3, 1.12, "threshold", fontsize=7, color=GRAY)
live_z_dot   = ax_neuron.scatter([], [], color=ACCENT3, s=80, zorder=6)
live_z_label = ax_neuron.text(0, 0, "", fontsize=7.5, color=ACCENT3)

# ── [3] Update rule equations ─────────────────────────────────────────────────
ax_update.axis("off")
ax_update.set_title("[3] Perceptron Update Rule", fontsize=9, color=ACCENT4, pad=5)
update_lines = [
    ("Linear output:",    "z = w.T * x + b",           ACCENT5),
    ("Prediction:",       "y_hat = g(z) = step(z)",    ACCENT2),
    ("Error:",            "delta = lr * (y - y_hat)",  ACCENT4),
    ("Weight update:",    "w  <-  w + delta * x",      ACCENT1),
    ("Bias update:",      "b  <-  b + delta",          ACCENT1),
    ("No mistake:",       "delta = 0  ->  no change",  GRAY),
]
upd_objs = []
for i, (lbl, eq, col) in enumerate(update_lines):
    y_pos = 0.90 - i * 0.145
    t1 = ax_update.text(0.03, y_pos, lbl, transform=ax_update.transAxes,
                         fontsize=7.5, color=GRAY, va="top", alpha=0)
    t2 = ax_update.text(0.03, y_pos - 0.05, eq, transform=ax_update.transAxes,
                         fontsize=9, color=col, va="top", alpha=0,
                         fontfamily="monospace")
    upd_objs.append((t1, t2))

# live w/b display
live_wb = ax_update.text(0.03, 0.06, "", transform=ax_update.transAxes,
                          fontsize=8, color=ACCENT3, va="bottom", alpha=0)

# ── [4] Boundary trails ────────────────────────────────────────────────────────
ax_bound.set_title("[4] Boundary Evolution across Epochs",
                    fontsize=9, color=ACCENT4, pad=5)
ax_bound.set_xlabel("Feature 0", fontsize=8)
ax_bound.set_ylabel("Feature 1", fontsize=8)
ax_bound.scatter(X_train[:, 0], X_train[:, 1], c=c_pts, s=12, alpha=0.4, zorder=2)
ax_bound.set_ylim(ymin_d, ymax_d)
trail_lines = []

# ── [5] Mistake curve ─────────────────────────────────────────────────────────
ax_err.set_title("[5] Mistakes per Epoch  (should reach 0)",
                  fontsize=9, color=ACCENT4, pad=5)
ax_err.set_xlabel("Epoch", fontsize=8)
ax_err.set_ylabel("Mistakes", fontsize=8)
ax_err.set_xlim(0, n_iters)
ax_err.set_ylim(-1, max(mistake_log) + 3)
err_line, = ax_err.plot([], [], color=ACCENT3, lw=2.5)
err_dot   = ax_err.scatter([], [], color=ACCENT4, s=70, zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────
TOTAL = 160

def ease(v): return max(0.0, min(1.0, float(v)))

def update(frame):
    artists = []

    # Phase 1 (0-20): Draw activation function
    if frame <= 20:
        n = max(1, int(len(z_vals) * ease(frame / 20)))
        act_line.set_data(z_vals[:n], step_out[:n])
        artists.append(act_line)

    # Phase 2 (21-40): Reveal update equations one by one
    elif frame <= 40:
        sub = frame - 21
        for i, (t1, t2) in enumerate(upd_objs):
            a = ease((sub - i * 3) / 3)
            t1.set_alpha(a); t2.set_alpha(a)
            artists += [t1, t2]

    # Phase 3 (41-140): Animate training epoch by epoch
    elif frame <= 140:
        sub   = frame - 41
        epoch = min(int(sub / 2.5), n_iters - 1)
        w_e, b_e, mistakes = snapshots[epoch]

        # Update decision boundary on main plot
        xr, yr = decision_line(w_e, b_e, x_range)
        if xr is not None:
            db_line.set_data(xr, yr)
        epoch_txt.set_text(f"Epoch {epoch+1:02d} / {n_iters}  |  mistakes: {mistakes}")

        # Highlight a sample point being "processed"
        sample_idx = (epoch * 7) % len(X_train)
        highlight_dot.set_offsets([X_train[sample_idx]])

        # Show z on activation plot for this sample
        z_val = np.dot(X_train[sample_idx], w_e) + b_e
        live_z_dot.set_offsets([[np.clip(z_val, -5.9, 5.9), unit_step(z_val)]])
        live_z_label.set_text(f"z={z_val:.2f}")
        live_z_label.set_position((np.clip(z_val, -5.5, 4.5), unit_step(z_val) + 0.08))

        # Live w/b text
        live_wb.set_text(f"w=[{w_e[0]:.3f}, {w_e[1]:.3f}]  b={b_e:.3f}")
        live_wb.set_alpha(1)

        # Add boundary trail (one per epoch)
        if xr is not None and epoch < len(trail_lines) + 1:
            alpha_trail = max(0.05, 0.35 - len(trail_lines) * 0.008)
            tl, = ax_bound.plot(xr, yr, color=ACCENT1,
                                 lw=1, alpha=alpha_trail, zorder=3)
            trail_lines.append(tl)

        # Mistake curve up to current epoch
        err_line.set_data(range(1, epoch + 2), mistake_log[:epoch + 1])
        err_dot.set_offsets([[epoch + 1, mistake_log[epoch]]])

        artists += [db_line, epoch_txt, highlight_dot, live_z_dot,
                    live_z_label, live_wb, err_line, err_dot] + trail_lines

    # Phase 4 (141-160): Final state — show test accuracy
    elif frame <= 160:
        sub = frame - 141
        # Solidify final boundary
        xr, yr = decision_line(final_w, final_b, x_range)
        if xr is not None:
            db_line.set_data(xr, yr)
            db_line.set_color(ACCENT4)
            db_line.set_linewidth(3)

        preds  = unit_step(X_test @ final_w + final_b)
        acc    = np.sum(preds == np.where(y_test > 0, 1, 0)) / len(y_test)
        epoch_txt.set_text(f"FINAL  |  Test Acc: {acc*100:.1f}%")
        highlight_dot.set_offsets(np.empty((0, 2)))

        artists += [db_line, epoch_txt, err_line, err_dot]

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL,
                    interval=65, blit=False, repeat=False)

writer = PillowWriter(fps=18)
out = "/mnt/user-data/outputs/perceptron_animation.gif"
ani.save(out, writer=writer, dpi=100)
print(f"GIF saved -> {out}")
plt.close(fig)
