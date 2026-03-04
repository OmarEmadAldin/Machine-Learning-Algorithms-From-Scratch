import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Data ──────────────────────────────────────────────────────────────────────
iris = datasets.load_iris()
X = iris.data[:, :2]   # use only first 2 features for 2‑D plot
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

COLORS   = ['#e74c3c', '#2ecc71', '#3498db']   # class colours
K        = 3
TEST_IDX = 0   # which test point to animate (change 0‑29 to try others)

query    = X_test[TEST_IDX]
true_lbl = y_test[TEST_IDX]

# pre‑compute distances & sorted order
dists   = np.sqrt(np.sum((X_train - query) ** 2, axis=1))
order   = np.argsort(dists)            # indices sorted by distance
k_idx   = order[:K]                    # K nearest indices
k_lbls  = [y_train[i] for i in k_idx]
pred    = Counter(k_lbls).most_common(1)[0][0]

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

ax_main, ax_bar = axes

# static training scatter
for cls in range(3):
    mask = y_train == cls
    ax_main.scatter(X_train[mask, 0], X_train[mask, 1],
                    color=COLORS[cls], alpha=0.4, s=40, edgecolors='none')

# query point (always visible)
ax_main.scatter(*query, color='yellow', s=180, zorder=10,
                marker='*', label='Query point')

ax_main.set_xlabel('Sepal Length', color='#aaa')
ax_main.set_ylabel('Sepal Width',  color='#aaa')
ax_main.set_title('KNN – Finding Nearest Neighbors', color='white', pad=10)

legend_handles = [mpatches.Patch(color=COLORS[i], label=iris.target_names[i])
                  for i in range(3)]
legend_handles.append(mpatches.Patch(color='yellow', label='Query point'))
ax_main.legend(handles=legend_handles, facecolor='#0f3460',
               labelcolor='white', fontsize=8)

# bar chart (vote counter)
ax_bar.set_xlim(-0.5, 2.5)
ax_bar.set_ylim(0, K + 0.5)
ax_bar.set_xticks([0, 1, 2])
ax_bar.set_xticklabels(iris.target_names, color='white')
ax_bar.set_ylabel('Votes', color='#aaa')
ax_bar.set_title('Neighbor Votes', color='white', pad=10)
bars = ax_bar.bar([0, 1, 2], [0, 0, 0],
                  color=COLORS, width=0.5, alpha=0.8)
vote_text = ax_bar.text(1, K * 0.85, '', ha='center',
                        color='white', fontsize=11, fontweight='bold')

# dynamic artists updated per frame
circle_artists   = []   # distance rings
neighbor_scatter = [None]
line_artists     = []
info_text        = ax_main.text(
    0.02, 0.97, '', transform=ax_main.transAxes,
    color='white', fontsize=9, va='top',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.8)
)

# ── Animation frames ──────────────────────────────────────────────────────────
# Frame plan:
#   0          : show query point only
#   1..N       : reveal each training point one by one with a distance ring
#   N+1..N+K   : highlight the K nearest one by one + draw line
#   N+K+1      : show final prediction

N_TRAIN  = len(X_train)
FRAMES   = 1 + N_TRAIN + K + 2   # total

def update(frame):
    # ── clean up previous circles / lines ────────────────────────────────────
    for art in circle_artists:
        art.remove()
    circle_artists.clear()
    for art in line_artists:
        art.remove()
    line_artists.clear()
    if neighbor_scatter[0] is not None:
        neighbor_scatter[0].remove()
        neighbor_scatter[0] = None

    vote_counts = [0, 0, 0]

    # ── Frame 0: intro ────────────────────────────────────────────────────────
    if frame == 0:
        info_text.set_text('Click Play to start\nQuery point placed ★')

    # ── Frames 1..N: scan training points ────────────────────────────────────
    elif 1 <= frame <= N_TRAIN:
        revealed = frame   # how many points revealed so far
        # dim ring growing outward – show current search radius concept
        idx      = order[revealed - 1]   # the revealed point
        pt       = X_train[idx]
        radius   = dists[idx]
        circle   = plt.Circle(query, radius,
                               fill=False, color='#ffffff22',
                               linewidth=1, linestyle='--')
        ax_main.add_patch(circle)
        circle_artists.append(circle)

        info_text.set_text(
            f'Scanning point {revealed}/{N_TRAIN}\n'
            f'Distance: {dists[idx]:.2f}'
        )

    # ── Frames N+1..N+K: highlight K nearest ─────────────────────────────────
    elif N_TRAIN < frame <= N_TRAIN + K:
        ki         = frame - N_TRAIN - 1   # 0‑based neighbor index
        shown_k    = ki + 1
        # draw lines to all shown neighbors
        for j in range(shown_k):
            idx  = k_idx[j]
            line, = ax_main.plot([query[0], X_train[idx, 0]],
                                  [query[1], X_train[idx, 1]],
                                  '--', color=COLORS[y_train[idx]],
                                  linewidth=1.5, alpha=0.8)
            line_artists.append(line)

        # highlight shown neighbors
        nh_x = [X_train[k_idx[j], 0] for j in range(shown_k)]
        nh_y = [X_train[k_idx[j], 1] for j in range(shown_k)]
        nh_c = [COLORS[y_train[k_idx[j]]] for j in range(shown_k)]
        neighbor_scatter[0] = ax_main.scatter(
            nh_x, nh_y, s=160, c=nh_c,
            edgecolors='white', linewidths=2, zorder=9
        )

        # update vote bar
        for j in range(shown_k):
            vote_counts[y_train[k_idx[j]]] += 1
        for i, bar in enumerate(bars):
            bar.set_height(vote_counts[i])
        vote_text.set_text('')

        info_text.set_text(
            f'Neighbor {shown_k}/{K}\n'
            f'Class: {iris.target_names[y_train[k_idx[ki]]]}\n'
            f'Dist: {dists[k_idx[ki]]:.2f}'
        )

    # ── Final frame: prediction ───────────────────────────────────────────────
    else:
        for j in range(K):
            idx  = k_idx[j]
            line, = ax_main.plot([query[0], X_train[idx, 0]],
                                  [query[1], X_train[idx, 1]],
                                  '--', color=COLORS[y_train[idx]],
                                  linewidth=1.5, alpha=0.8)
            line_artists.append(line)

        nh_x = X_train[k_idx, 0]
        nh_y = X_train[k_idx, 1]
        nh_c = [COLORS[y_train[i]] for i in k_idx]
        neighbor_scatter[0] = ax_main.scatter(
            nh_x, nh_y, s=160, c=nh_c,
            edgecolors='white', linewidths=2, zorder=9
        )

        for j in range(K):
            vote_counts[y_train[k_idx[j]]] += 1
        for i, bar in enumerate(bars):
            bar.set_height(vote_counts[i])

        result = ('✓ Correct' if pred == true_lbl else '✗ Wrong')
        vote_text.set_text(
            f'Predicted: {iris.target_names[pred]}\n{result}'
        )
        vote_text.set_color('#2ecc71' if pred == true_lbl else '#e74c3c')

        info_text.set_text(
            f'Prediction: {iris.target_names[pred]}\n'
            f'True label: {iris.target_names[true_lbl]}\n'
            f'{result}'
        )

    return circle_artists + line_artists + [info_text, vote_text] + \
           ([neighbor_scatter[0]] if neighbor_scatter[0] else []) + list(bars)


anim = FuncAnimation(fig, update, frames=FRAMES,
                     interval=300, blit=False, repeat=False)

# ── Save as GIF ───────────────────────────────────────────────────────────────
writer = PillowWriter(fps=10)
anim.save("Knn_Animation.gif", writer=writer, dpi=100)
print('Saved!')

plt.tight_layout()
plt.show()
