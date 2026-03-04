import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split
from sklearn import datasets

# ── Data ──────────────────────────────────────────────────────────────────────
x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
x_flat = x_train.flatten()

# ── Run gradient descent, snapshot weights every few iters ───────────────────
LEARNING_RATE = 0.01
N_ITERS       = 200
SNAPSHOT_EVERY = 5   # record a frame every N iters

n_samples  = x_train.shape[0]
weight     = 0.0
bias       = 0.0
snapshots  = []   # list of (iter, w, b, mse)

for i in range(N_ITERS):
    y_pred = x_flat * weight + bias
    error  = y_pred - y_train

    dw = (1 / n_samples) * np.dot(x_flat, error)
    db = (1 / n_samples) * np.sum(error)

    weight -= LEARNING_RATE * dw
    bias   -= LEARNING_RATE * db

    if i % SNAPSHOT_EVERY == 0 or i == N_ITERS - 1:
        mse = np.mean(error ** 2)
        snapshots.append((i + 1, weight, bias, mse))

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, (ax_main, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in (ax_main, ax_loss):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# ── Left: scatter + regression line ──────────────────────────────────────────
cmap = plt.get_cmap('viridis')
ax_main.scatter(x_train, y_train, color=cmap(0.9), s=50, alpha=0.7, label='Train')
ax_main.scatter(x_test,  y_test,  color=cmap(0.5), s=50, alpha=0.7, label='Test')
ax_main.set_xlabel('x',        color='#aaa', fontsize=11)
ax_main.set_ylabel('y',        color='#aaa', fontsize=11)
ax_main.set_title('Gradient Descent – Fitting the Line', color='white', pad=10)
ax_main.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)

x_line = np.linspace(x.min(), x.max(), 200)
reg_line, = ax_main.plot([], [], color='#e74c3c', linewidth=2.5, label='Prediction')
ax_main.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)

info_box = ax_main.text(
    0.03, 0.97, '', transform=ax_main.transAxes,
    color='white', fontsize=9, va='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.85)
)

# residual lines (drawn fresh each frame)
residual_lines = []

# ── Right: loss curve ─────────────────────────────────────────────────────────
all_iters = [s[0] for s in snapshots]
all_mses  = [s[3] for s in snapshots]

ax_loss.plot(all_iters, all_mses, color='#ffffff22', linewidth=1.5)
ax_loss.set_xlabel('Iteration', color='#aaa', fontsize=11)
ax_loss.set_ylabel('MSE',       color='#aaa', fontsize=11)
ax_loss.set_title('Loss Curve (MSE)', color='white', pad=10)
ax_loss.set_xlim(0, N_ITERS)
ax_loss.set_ylim(0, max(all_mses) * 1.05)

loss_dot,  = ax_loss.plot([], [], 'o', color='#e74c3c', markersize=8, zorder=5)
loss_trail,= ax_loss.plot([], [], color='#e74c3c', linewidth=2)

plt.tight_layout()

# ── Update function ───────────────────────────────────────────────────────────
def update(frame_idx):
    global residual_lines
    itr, w, b, mse = snapshots[frame_idx]

    # regression line
    y_line = x_line * w + b
    reg_line.set_data(x_line, y_line)

    # residuals
    for ln in residual_lines:
        ln.remove()
    residual_lines = []
    y_train_pred = x_flat * w + b
    for xi, yi_true, yi_pred in zip(x_flat, y_train, y_train_pred):
        ln, = ax_main.plot([xi, xi], [yi_true, yi_pred],
                           color='#e74c3c', alpha=0.25, linewidth=1)
        residual_lines.append(ln)

    # info box
    info_box.set_text(
        f'Iteration : {itr}/{N_ITERS}\n'
        f'Weight    : {w:.3f}\n'
        f'Bias      : {b:.3f}\n'
        f'MSE       : {mse:.2f}'
    )

    # loss dot + trail
    trail_iters = [s[0] for s in snapshots[:frame_idx + 1]]
    trail_mses  = [s[3] for s in snapshots[:frame_idx + 1]]
    loss_trail.set_data(trail_iters, trail_mses)
    loss_dot.set_data([itr], [mse])

    return [reg_line, info_box, loss_dot, loss_trail] + residual_lines


anim = FuncAnimation(fig, update, frames=len(snapshots),
                     interval=120, blit=False, repeat=False)

# ── Save as GIF ───────────────────────────────────────────────────────────────
GIF_PATH = 'linear_regression_animation.gif'
print(f'Saving animation to {GIF_PATH} …')
writer = PillowWriter(fps=10)
anim.save(GIF_PATH, writer=writer, dpi=100)
print('Saved!')

plt.show()
