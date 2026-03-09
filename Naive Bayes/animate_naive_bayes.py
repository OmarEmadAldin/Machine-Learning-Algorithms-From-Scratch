import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT1 = "#7c6aff"
ACCENT2 = "#00d4aa"
ACCENT3 = "#ff6b6b"
ACCENT4 = "#ffd166"
WHITE   = "#e8eaf6"
GRAY    = "#4a4e69"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRAY, "text.color": WHITE,
    "axes.labelcolor": WHITE, "xtick.color": WHITE,
    "ytick.color": WHITE, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "monospace",
})

# ── Synthetic 1-D data (2 classes) ───────────────────────────────────────────
n_each = 60
X0 = np.random.randn(n_each) * 1.0 + (-2.0)
X1 = np.random.randn(n_each) * 1.2 + ( 2.0)
X  = np.concatenate([X0, X1])
y  = np.array([0]*n_each + [1]*n_each)

mu0, var0 = X0.mean(), X0.var()
mu1, var1 = X1.mean(), X1.var()
prior0 = n_each / len(y)
prior1 = n_each / len(y)

def gaussian(x, mu, var):
    return np.exp(-((x - mu)**2) / (2*var)) / np.sqrt(2 * np.pi * var)

x_range = np.linspace(-7, 7, 400)
pdf0    = gaussian(x_range, mu0, var0)
pdf1    = gaussian(x_range, mu1, var1)

# posterior over range (log scale, unnormalised)
log_post0 = np.log(prior0) + np.log(gaussian(x_range, mu0, var0) + 1e-12)
log_post1 = np.log(prior1) + np.log(gaussian(x_range, mu1, var1) + 1e-12)
decision  = x_range[np.argmin(np.abs(log_post0 - log_post1))]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         left=0.05, right=0.97,
                         top=0.89, bottom=0.07,
                         hspace=0.55, wspace=0.38)

ax_data  = fig.add_subplot(gs[0, 0])   # raw data histogram
ax_prior = fig.add_subplot(gs[1, 0])   # prior bars
ax_pdf   = fig.add_subplot(gs[0, 1])   # Gaussian PDFs
ax_post  = fig.add_subplot(gs[1, 1])   # log-posteriors
ax_eq    = fig.add_subplot(gs[0, 2])   # equation panel
ax_pred  = fig.add_subplot(gs[1, 2])   # single prediction walkthrough

for ax in [ax_data, ax_prior, ax_pdf, ax_post, ax_eq, ax_pred]:
    ax.set_facecolor(PANEL)

fig.text(0.5, 0.95, "Naive Bayes Classifier — Step-by-Step",
         ha="center", fontsize=18, color=WHITE,
         fontweight="bold", fontfamily="monospace")

# ── [1] Data histogram ────────────────────────────────────────────────────────
ax_data.set_title("[1] Training Data (1-D feature)", fontsize=9, color=ACCENT4, pad=5)
ax_data.set_xlabel("Feature value", fontsize=8)
ax_data.set_ylabel("Count", fontsize=8)
hist0_bars = ax_data.hist(X0, bins=20, color=ACCENT1, alpha=0, label="Class 0")[2]
hist1_bars = ax_data.hist(X1, bins=20, color=ACCENT2, alpha=0, label="Class 1")[2]
ax_data.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)

# ── [2] Prior bars ────────────────────────────────────────────────────────────
ax_prior.set_title("[2] Prior:  P(y=c) = n_c / n", fontsize=9, color=ACCENT4, pad=5)
ax_prior.set_xticks([0, 1])
ax_prior.set_xticklabels(["Class 0", "Class 1"], fontsize=8)
ax_prior.set_ylim(0, 0.8)
ax_prior.set_ylabel("Probability", fontsize=8)
prior_bar0 = ax_prior.bar(0, 0, color=ACCENT1, width=0.4, edgecolor=WHITE, lw=0.8)
prior_bar1 = ax_prior.bar(1, 0, color=ACCENT2, width=0.4, edgecolor=WHITE, lw=0.8)
prior_txt0 = ax_prior.text(0, 0.01, "", ha="center", fontsize=8, color=WHITE)
prior_txt1 = ax_prior.text(1, 0.01, "", ha="center", fontsize=8, color=WHITE)

# ── [3] Gaussian PDFs ─────────────────────────────────────────────────────────
ax_pdf.set_title("[3] Likelihood:  P(x|y) = Gaussian(mu, sigma^2)",
                  fontsize=8, color=ACCENT4, pad=5)
ax_pdf.set_xlabel("Feature value", fontsize=8)
ax_pdf.set_ylabel("Density", fontsize=8)
pdf_line0, = ax_pdf.plot([], [], color=ACCENT1, lw=2.5, label="Class 0")
pdf_line1, = ax_pdf.plot([], [], color=ACCENT2, lw=2.5, label="Class 1")
mu0_line   = ax_pdf.axvline(x=mu0, color=ACCENT1, lw=1, ls=":", alpha=0)
mu1_line   = ax_pdf.axvline(x=mu1, color=ACCENT2, lw=1, ls=":", alpha=0)
ax_pdf.set_xlim(-7, 7)
ax_pdf.set_ylim(0, max(pdf0.max(), pdf1.max()) * 1.25)
ax_pdf.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)
pdf_mu_txt0 = ax_pdf.text(mu0, pdf0.max() * 1.05, "", color=ACCENT1,
                            fontsize=7, ha="center", alpha=0)
pdf_mu_txt1 = ax_pdf.text(mu1, pdf1.max() * 1.05, "", color=ACCENT2,
                            fontsize=7, ha="center", alpha=0)

# ── [4] Log-posteriors ────────────────────────────────────────────────────────
ax_post.set_title("[4] Log-Posterior = log P(y) + sum log P(x_j|y)",
                   fontsize=8, color=ACCENT4, pad=5)
ax_post.set_xlabel("Feature value", fontsize=8)
ax_post.set_ylabel("Log-posterior", fontsize=8)
post_line0, = ax_post.plot([], [], color=ACCENT1, lw=2, label="Class 0")
post_line1, = ax_post.plot([], [], color=ACCENT2, lw=2, label="Class 1")
post_fill0  = None
post_fill1  = None
dec_line    = ax_post.axvline(x=decision, color=ACCENT3, lw=1.8,
                               ls="--", alpha=0, label="Decision boundary")
ax_post.set_xlim(-7, 7)
ymin_p = min(log_post0.min(), log_post1.min())
ymax_p = max(log_post0.max(), log_post1.max())
ax_post.set_ylim(ymin_p - 1, ymax_p + 1)
ax_post.legend(fontsize=7, facecolor=PANEL, edgecolor=GRAY, labelcolor=WHITE)

# ── [5] Equation panel ────────────────────────────────────────────────────────
ax_eq.axis("off")
ax_eq.set_title("[5] Equations Summary", fontsize=9, color=ACCENT4, pad=5)
equations = [
    ("Bayes Theorem:",   "P(y|X)  ∝  P(X|y) · P(y)",        ACCENT4),
    ("Prior:",           "P(y=c) = n_c / n",                  ACCENT1),
    ("Gaussian PDF:",    "P(x|y) = exp(-(x-mu)^2 / 2s^2)\n"
                         "         / sqrt(2·pi·s^2)",         ACCENT2),
    ("Independence:",    "P(X|y) = prod_j P(x_j|y)",          ACCENT5 := "#a8dadc"),
    ("Prediction:",      "y = argmax [ log P(y)\n"
                         "           + sum log P(x_j|y) ]",   ACCENT3),
]
eq_texts = []
for i, (label, eq, col) in enumerate(equations):
    y_pos = 0.92 - i * 0.19
    t1 = ax_eq.text(0.02, y_pos, label, transform=ax_eq.transAxes,
                     fontsize=8, color=GRAY, va="top", alpha=0)
    t2 = ax_eq.text(0.02, y_pos - 0.045, eq, transform=ax_eq.transAxes,
                     fontsize=8.5, color=col, va="top", alpha=0,
                     fontfamily="monospace")
    eq_texts.append((t1, t2))

# ── [6] Single prediction walkthrough ────────────────────────────────────────
ax_pred.axis("off")
ax_pred.set_title("[6] Predict  x = 1.0", fontsize=9, color=ACCENT4, pad=5)
x_test_val = 1.0
lp0 = np.log(prior0) + np.log(gaussian(x_test_val, mu0, var0) + 1e-12)
lp1 = np.log(prior1) + np.log(gaussian(x_test_val, mu1, var1) + 1e-12)
pred_class = 0 if lp0 > lp1 else 1

pred_lines = [
    f"x = {x_test_val}",
    f"",
    f"log P(C0|x) =",
    f"  log({prior0:.2f}) + log(pdf0(x))",
    f"  = {lp0:.3f}",
    f"",
    f"log P(C1|x) =",
    f"  log({prior1:.2f}) + log(pdf1(x))",
    f"  = {lp1:.3f}",
    f"",
    f"=> predict Class {pred_class}",
]
pred_text_objs = []
for i, line in enumerate(pred_lines):
    col = ACCENT3 if "predict" in line else (ACCENT1 if "C0" in line else
          (ACCENT2 if "C1" in line else WHITE))
    t = ax_pred.text(0.05, 0.95 - i * 0.082, line,
                      transform=ax_pred.transAxes,
                      fontsize=8.5, color=col, va="top", alpha=0,
                      fontfamily="monospace")
    pred_text_objs.append(t)

# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────
TOTAL = 140

def ease(v): return max(0.0, min(1.0, v))

def update(frame):
    artists = []

    # Phase 1 (0-20): Histogram bars fade in
    if frame <= 20:
        a = ease(frame / 20)
        for bar in hist0_bars: bar.set_alpha(a * 0.8)
        for bar in hist1_bars: bar.set_alpha(a * 0.8)
        artists += list(hist0_bars) + list(hist1_bars)

    # Phase 2 (21-40): Prior bars grow
    elif frame <= 40:
        a = ease((frame - 21) / 19)
        prior_bar0[0].set_height(prior0 * a)
        prior_bar1[0].set_height(prior1 * a)
        prior_txt0.set_text(f"{prior0 * a:.2f}")
        prior_txt1.set_text(f"{prior1 * a:.2f}")
        prior_txt0.set_position((0, prior0 * a + 0.01))
        prior_txt1.set_position((1, prior1 * a + 0.01))
        artists += [prior_bar0[0], prior_bar1[0], prior_txt0, prior_txt1]

    # Phase 3 (41-70): Gaussian PDFs draw
    elif frame <= 70:
        prog = ease((frame - 41) / 29)
        n    = max(1, int(len(x_range) * prog))
        pdf_line0.set_data(x_range[:n], pdf0[:n])
        pdf_line1.set_data(x_range[:n], pdf1[:n])
        a_mu = ease((frame - 60) / 10)
        mu0_line.set_alpha(a_mu * 0.7)
        mu1_line.set_alpha(a_mu * 0.7)
        pdf_mu_txt0.set_alpha(a_mu)
        pdf_mu_txt1.set_alpha(a_mu)
        pdf_mu_txt0.set_text(f"mu={mu0:.2f}")
        pdf_mu_txt1.set_text(f"mu={mu1:.2f}")
        artists += [pdf_line0, pdf_line1, mu0_line, mu1_line,
                    pdf_mu_txt0, pdf_mu_txt1]

    # Phase 4 (71-95): Log-posterior curves + decision boundary
    elif frame <= 95:
        prog = ease((frame - 71) / 24)
        n    = max(1, int(len(x_range) * prog))
        post_line0.set_data(x_range[:n], log_post0[:n])
        post_line1.set_data(x_range[:n], log_post1[:n])
        a_dec = ease((frame - 88) / 7)
        dec_line.set_alpha(a_dec)
        artists += [post_line0, post_line1, dec_line]

    # Phase 5 (96-115): Equation panel reveals line by line
    elif frame <= 115:
        sub = frame - 96
        for i, (t1, t2) in enumerate(eq_texts):
            a = ease((sub - i * 4) / 4)
            t1.set_alpha(a); t2.set_alpha(a)
            artists += [t1, t2]

    # Phase 6 (116-140): Prediction walkthrough
    elif frame <= 140:
        sub = frame - 116
        for i, t in enumerate(pred_text_objs):
            a = ease((sub - i * 2) / 3)
            t.set_alpha(a)
            artists.append(t)

    return artists

ani = FuncAnimation(fig, update, frames=TOTAL,
                    interval=65, blit=False, repeat=False)

writer = PillowWriter(fps=18)
out = "/mnt/user-data/outputs/naive_bayes_animation.gif"
ani.save(out, writer=writer, dpi=100)
print(f"GIF saved -> {out}")
plt.close(fig)
