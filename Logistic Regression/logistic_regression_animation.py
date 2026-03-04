import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split
from sklearn import datasets

# ── Data ──────────────────────────────────────────────────────────────────────
bc = datasets.load_breast_cancer()
X_full, Y_full = bc.data, bc.target
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=1234
)

# Train on ALL features (for >90% accuracy), plot first 2 for visualization
FEAT_A, FEAT_B = 0, 1   # Mean Radius, Mean Texture (for scatter plot only)

# Standardize all features — critical for gradient descent convergence
mean_all = X_train.mean(axis=0)
std_all  = X_train.std(axis=0)
X_tr_scaled  = (X_train - mean_all) / std_all
X_te_scaled  = (X_test  - mean_all) / std_all

# 2-feature version just for the scatter plot
X2_train = X_tr_scaled[:, [FEAT_A, FEAT_B]]
X2_test  = X_te_scaled[:, [FEAT_A, FEAT_B]]

# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ── Run gradient descent, snapshot every few iters ───────────────────────────
LEARNING_RATE  = 0.001   # matches LogisticRegression default
N_ITERS        = 1000    # matches LogisticRegression default
SNAPSHOT_EVERY = 20

n_samples  = X_tr_scaled.shape[0]
n_features = X_tr_scaled.shape[1]
w = np.zeros(n_features)   # self.weights = np.zeros(n_features)
b = 0.0                    # self.bias = 0
snapshots = []   # (iter, w2d, b, loss, acc)  w2d = first 2 weights for plotting

for i in range(N_ITERS):
    # exact equations from LogisticRegression.fit()
    linear_pred = np.dot(X_tr_scaled, w) + b
    y_predicted = sigmoid(linear_pred)

    dw = (1 / n_samples) * np.dot(X_tr_scaled.T, (y_predicted - Y_train))
    db = (1 / n_samples) * np.sum(y_predicted - Y_train)
    w = w - LEARNING_RATE * dw
    b = b - LEARNING_RATE * db

    if i % SNAPSHOT_EVERY == 0 or i == N_ITERS - 1:
        loss = binary_cross_entropy(Y_train, sigmoid(np.dot(X_tr_scaled, w) + b))
        y_pred = sigmoid(np.dot(X_tr_scaled, w) + b)
        preds = np.array([1 if v > 0.5 else 0 for v in y_pred])
        acc   = accuracy(Y_train, preds)
        snapshots.append((i + 1, w[[FEAT_A, FEAT_B]].copy(), b, loss, acc))

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, (ax_main, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in (ax_main, ax_loss):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')

COLORS = {0: '#e74c3c', 1: '#2ecc71'}
LABELS = {0: 'Malignant', 1: 'Benign'}

# scatter (static)
for cls in [0, 1]:
    mask = Y_train == cls
    ax_main.scatter(X2_train[mask, 0], X2_train[mask, 1],
                    color=COLORS[cls], alpha=0.45, s=30, label=LABELS[cls])

ax_main.set_xlabel(bc.feature_names[FEAT_A], color='#aaa', fontsize=9)
ax_main.set_ylabel(bc.feature_names[FEAT_B], color='#aaa', fontsize=9)
ax_main.set_title('Logistic Regression – Decision Boundary', color='white', pad=10)
ax_main.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)

x_min, x_max = X2_train[:, 0].min() - 1, X2_train[:, 0].max() + 1
y_min, y_max = X2_train[:, 1].min() - 1, X2_train[:, 1].max() + 1

# decision boundary line
db_line, = ax_main.plot([], [], color='#f39c12', linewidth=2.5, label='Decision boundary')
ax_main.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)

info_box = ax_main.text(
    0.03, 0.97, '', transform=ax_main.transAxes,
    color='white', fontsize=9, va='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.85)
)

# sigmoid curve inset
ax_sig = ax_main.inset_axes([0.68, 0.03, 0.30, 0.28])
ax_sig.set_facecolor('#0f3460')
ax_sig.tick_params(colors='#777', labelsize=6)
for sp in ax_sig.spines.values():
    sp.set_edgecolor('#445')
z_vals = np.linspace(-6, 6, 200)
ax_sig.plot(z_vals, sigmoid(z_vals), color='#f39c12', linewidth=1.5)
ax_sig.axhline(0.5, color='#ffffff33', linewidth=1, linestyle='--')
ax_sig.set_title('σ(z)', color='#aaa', fontsize=7, pad=2)
sig_dot, = ax_sig.plot([], [], 'o', color='#e74c3c', markersize=5)

# ── Right: loss + accuracy curves ────────────────────────────────────────────
all_iters  = [s[0]  for s in snapshots]
all_losses = [s[3]  for s in snapshots]
all_accs   = [s[4]  for s in snapshots]

ax_loss.plot(all_iters, all_losses, color='#ffffff18', linewidth=1.5, label='Loss')
ax2 = ax_loss.twinx()
ax2.plot(all_iters, all_accs, color='#2ecc7130', linewidth=1.5, label='Accuracy')
ax2.tick_params(colors='#aaa')
ax2.set_ylabel('Accuracy', color='#2ecc71', fontsize=9)
ax2.set_ylim(0, 1.05)

ax_loss.set_xlabel('Iteration', color='#aaa', fontsize=11)
ax_loss.set_ylabel('Loss (BCE)',  color='#e74c3c', fontsize=9)
ax_loss.set_title('Loss & Accuracy', color='white', pad=10)
ax_loss.set_xlim(0, N_ITERS)
ax_loss.set_ylim(0, max(all_losses) * 1.1)

loss_dot,  = ax_loss.plot([], [], 'o', color='#e74c3c', markersize=8, zorder=5)
loss_trail,= ax_loss.plot([], [], color='#e74c3c', linewidth=2)
acc_dot,   = ax2.plot([], [], 'o', color='#2ecc71', markersize=8, zorder=5)
acc_trail, = ax2.plot([], [], color='#2ecc71', linewidth=2)

plt.tight_layout()

# ── Update function ───────────────────────────────────────────────────────────
fill_artist = [None]

def update(fi):
    itr, w, b, loss, acc = snapshots[fi]

    # decision boundary: w[0]*x + w[1]*y + b = 0  →  y = -(w[0]*x + b) / w[1]
    if abs(w[1]) > 1e-8:
        x_vals = np.array([x_min, x_max])
        y_vals = -(w[0] * x_vals + b) / w[1]
        db_line.set_data(x_vals, y_vals)
    else:
        db_line.set_data([], [])

    # sigmoid dot (mean linear prediction)
    z_mean = np.mean(X2_train @ snapshots[fi][1] + b)
    sig_dot.set_data([np.clip(z_mean, -6, 6)], [sigmoid(z_mean)])

    # info box
    info_box.set_text(
        f'Iteration : {itr}/{N_ITERS}\n'
        f'Weights   : [{w[0]:.3f}, {w[1]:.3f}]\n'
        f'Bias      : {b:.3f}\n'
        f'Loss(BCE) : {loss:.4f}\n'
        f'Accuracy  : {acc*100:.1f}%'
    )

    # trails
    t_iters  = [s[0] for s in snapshots[:fi + 1]]
    t_losses = [s[3] for s in snapshots[:fi + 1]]
    t_accs   = [s[4] for s in snapshots[:fi + 1]]
    loss_trail.set_data(t_iters, t_losses)
    loss_dot.set_data([itr], [loss])
    acc_trail.set_data(t_iters, t_accs)
    acc_dot.set_data([itr], [acc])

    return [db_line, info_box, sig_dot, loss_dot, loss_trail, acc_dot, acc_trail]


anim = FuncAnimation(fig, update, frames=len(snapshots),
                     interval=150, blit=False, repeat=False)

# ── Save GIF ──────────────────────────────────────────────────────────────────
GIF_PATH = 'logistic_regression_animation.gif'
print(f'Saving {GIF_PATH} …')
anim.save(GIF_PATH, writer=PillowWriter(fps=8), dpi=100)
print('Saved!')

plt.show()
