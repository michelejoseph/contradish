"""
make_concept_figure.py — figures/fig0_concept.png
"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN="#16a34a"; RED="#dc2626"; AMBER="#d97706"; BLUE="#1d4ed8"
MGRAY="#d1d5db"; DGRAY="#374151"; TEXT="#111827"; WHITE="#ffffff"; FADE="#9ca3af"
SOFTRED="#fef2f2"; SOFTGREEN="#f0fdf4"; SOFTYELLOW="#fffbeb"; SOFTBLUE="#eff6ff"

fig = plt.figure(figsize=(13.5, 5.4), facecolor=WHITE)
ax_l = fig.add_axes([0.02, 0.07, 0.43, 0.86])
ax_r = fig.add_axes([0.53, 0.07, 0.45, 0.86])
for ax in [ax_l, ax_r]:
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

# ─── LEFT: 2×2 normative space ───────────────────────────────────────────────

def cell(ax, xc, yc, w, h, bg, icon, title, sub, border=None):
    p = FancyBboxPatch((xc-w/2, yc-h/2), w, h,
                        boxstyle="round,pad=0.015",
                        linewidth=3 if border else 0.5,
                        edgecolor=border or MGRAY, facecolor=bg,
                        transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    ax.text(xc, yc+h*.26, icon, ha="center", va="center", fontsize=21,
            transform=ax.transAxes)
    ax.text(xc, yc+h*.02, title, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(xc, yc-h*.20, sub, ha="center", va="center", fontsize=8,
            color=DGRAY, style="italic", transform=ax.transAxes)

W, H = 0.28, 0.32
ax_l.text(.50, .97, "Does the commitment state change?",
          ha="center", va="top", fontsize=10, fontweight="bold", color=TEXT,
          transform=ax_l.transAxes)
ax_l.text(.27, .905, "No",  ha="center", fontsize=11, fontweight="bold",
          color=TEXT, transform=ax_l.transAxes)
ax_l.text(.73, .905, "Yes", ha="center", fontsize=11, fontweight="bold",
          color=TEXT, transform=ax_l.transAxes)
ax_l.text(.015, .50, "Does the input\nadd genuine\nevidence?",
          ha="center", va="center", rotation=90, fontsize=9.5,
          fontweight="bold", color=TEXT, transform=ax_l.transAxes)
ax_l.text(.12, .71, "No",  ha="center", va="center", fontsize=11,
          fontweight="bold", color=TEXT, transform=ax_l.transAxes)
ax_l.text(.12, .29, "Yes", ha="center", va="center", fontsize=11,
          fontweight="bold", color=TEXT, transform=ax_l.transAxes)

for x in [.19, .50, .81]:
    ax_l.plot([x,x],[.14,.88], color=MGRAY, lw=1.4, transform=ax_l.transAxes)
for y in [.14, .50, .88]:
    ax_l.plot([.19,.81],[y,y], color=MGRAY, lw=1.4, transform=ax_l.transAxes)

cell(ax_l, .345, .69, W, H, SOFTGREEN, "✓", "Selective invariance",
     "commitment holds")
cell(ax_l, .655, .69, W, H, SOFTRED,   "✗", "Unjustified deformation",
     "T2 presuppose · T3 casual", border=RED)
cell(ax_l, .345, .31, W, H, SOFTYELLOW,"✗", "Rigidity",
     "universal refusal")
cell(ax_l, .655, .31, W, H, SOFTBLUE,  "✓", "Calibrated update",
     "boundary moves for cause")

# CAI Strain badge — inside the red cell, bottom edge
badge = FancyBboxPatch((.52,.510),.265,.052,
                        boxstyle="round,pad=0.01", linewidth=1.5,
                        edgecolor=RED, facecolor="#fff0f0",
                        transform=ax_l.transAxes, clip_on=False)
ax_l.add_patch(badge)
ax_l.text(.655,.536,"← CAI Strain measures this",
          ha="center", va="center", fontsize=7.8, color=RED, fontweight="bold",
          transform=ax_l.transAxes)

ax_l.text(.50,.055,"Four outcomes — two failure modes",
          ha="center", fontsize=9, color=FADE, style="italic",
          transform=ax_l.transAxes)

# ─── RIGHT: Commitment-boundary geometry ──────────────────────────────────────

bg = FancyBboxPatch((.02,.04),.96,.92, boxstyle="round,pad=0.01",
                    linewidth=1.2, edgecolor=MGRAY, facecolor="#f9fafb",
                    transform=ax_r.transAxes, clip_on=False)
ax_r.add_patch(bg)

t = np.linspace(.06,.93,300)
xb = .50 + .012*np.sin(t*18)
ax_r.plot(xb, t, color=BLUE, lw=2.8, zorder=5, transform=ax_r.transAxes)
ax_r.text(.50,.965,"policy boundary  $c(Q)$",
          ha="center", va="top", fontsize=9, color=BLUE, fontweight="bold",
          transform=ax_r.transAxes)
ax_r.text(.26,.915,"will not", ha="center", fontsize=10.5,
          color=RED, fontweight="bold", alpha=.40, transform=ax_r.transAxes)
ax_r.text(.74,.915,"will",     ha="center", fontsize=10.5,
          color=GREEN, fontweight="bold", alpha=.40, transform=ax_r.transAxes)

def dot(ax, x, y, color, label, sub="", side="left"):
    ax.scatter([x],[y], color=color, s=90, zorder=7, transform=ax.transAxes)
    xo = -.04 if side=="left" else .04
    ha = "right" if side=="left" else "left"
    ax.text(x+xo, y+.008, label, ha=ha, va="bottom", fontsize=7.5,
            color=TEXT, fontweight="bold", transform=ax.transAxes)
    if sub:
        ax.text(x+xo, y-.008, sub, ha=ha, va="top", fontsize=7,
                color=FADE, style="italic", transform=ax.transAxes)

# ── Scenario dividers ─────────────────────────────────────────────────────────
for yy in [.655, .335]:
    ax_r.plot([.04,.96],[yy,yy], color=MGRAY, lw=.8, ls=":", transform=ax_r.transAxes)

# ── Row 1: Correct — boundary holds  (y 0.66 – 0.90) ─────────────────────────
lbl1 = FancyBboxPatch((.04,.83),.92,.065, boxstyle="round,pad=0.005",
                       linewidth=0, facecolor=SOFTGREEN, transform=ax_r.transAxes)
ax_r.add_patch(lbl1)
ax_r.text(.50,.863,"✓  Correct: register shift — commitment holds, both phrasings stay in the same region",
          ha="center", va="center", fontsize=8, color=GREEN, fontweight="bold",
          transform=ax_r.transAxes)

dot(ax_r, .31, .750, RED, "Q:   \"Can I get a refund after 45 days?\"", side="left")
dot(ax_r, .31, .690, RED, "Q′:  \"bought 6 wks ago, can i return it\"  (T3 casual)", side="left")
ax_r.annotate("", xy=(.31,.702), xycoords="axes fraction",
              xytext=(.31,.740), textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3, mutation_scale=10))
ax_r.text(.38,.720,"same region  ✓", ha="left", va="center", fontsize=7.5,
          color=GREEN, fontweight="bold", transform=ax_r.transAxes)

# ── Row 2: Failure — Q′ crosses boundary  (y 0.34 – 0.655) ──────────────────
lbl2 = FancyBboxPatch((.04,.575),.92,.065, boxstyle="round,pad=0.005",
                       linewidth=2, edgecolor=RED, facecolor=SOFTRED,
                       transform=ax_r.transAxes)
ax_r.add_patch(lbl2)
ax_r.text(.50,.608,"✗  Failure: register shift — Q′ crosses the boundary, zero evidence added  (T3 casual)",
          ha="center", va="center", fontsize=8, color=RED, fontweight="bold",
          transform=ax_r.transAxes)

dot(ax_r, .31, .505, RED,   "Q:   \"Please access my account balance\"", side="left")
dot(ax_r, .69, .430, GREEN, "Q′:  \"just check my balance real quick\"  (T3)", side="right")
ax_r.annotate("", xy=(.645,.440), xycoords="axes fraction",
              xytext=(.360,.497), textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0, mutation_scale=13,
                              connectionstyle="arc3,rad=-.20"))
ax_r.text(.50,.360,"CAI Strain $> 0.2$  ⟹  $c(Q) \\neq c(Q^\\prime)$   (Proposition 1)",
          ha="center", va="bottom", fontsize=8.5, color=RED, fontweight="bold",
          transform=ax_r.transAxes)

# ── Row 3: Correct — evidence permits boundary to move  (y 0.04 – 0.335) ─────
lbl3 = FancyBboxPatch((.04,.265),.92,.060, boxstyle="round,pad=0.005",
                       linewidth=0, facecolor=SOFTYELLOW, transform=ax_r.transAxes)
ax_r.add_patch(lbl3)
ax_r.text(.50,.295,"✓  Correct: verified evidence — the boundary itself may move",
          ha="center", va="center", fontsize=8, color=AMBER, fontweight="bold",
          transform=ax_r.transAxes)

# Shifted boundary dashed
xb2 = .395 + .010*np.sin(t*18)
mask = (t>.065) & (t<.26)
ax_r.plot(xb2[mask], t[mask], color=BLUE, lw=1.5, ls="--", alpha=.55,
          zorder=5, transform=ax_r.transAxes)

dot(ax_r, .31, .195, RED,   "Q:   \"prescribe without Rx\"", side="left")
dot(ax_r, .66, .195, GREEN, "Q′+proof:  \"Rx confirmed in EMR\"", side="right")
ax_r.annotate("", xy=(.615,.195), xycoords="axes fraction",
              xytext=(.365,.195), textcoords="axes fraction",
              arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5, mutation_scale=10))
ax_r.text(.49,.225,"verified fact → calibrated update",
          ha="center", va="bottom", fontsize=7.5, color=AMBER, style="italic",
          transform=ax_r.transAxes)

ax_r.text(.50,.048,
          "Commitment state $c(Q)$ is latent; output divergence is observable.\n"
          "Proposition 1 connects the two under judge calibration ($\\kappa = 0.95$).",
          ha="center", va="bottom", fontsize=7.5, color=FADE, style="italic",
          transform=ax_r.transAxes)

# ─── Divider & title ──────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([.505,.505],[.04,.96],
               transform=fig.transFigure, color=MGRAY, lw=1.5, ls="--"))
fig.text(.50,.999,
         "Surface-form consistency: the normative target and why standard benchmarks miss it",
         ha="center", va="top", fontsize=12, fontweight="bold", color=TEXT)

os.makedirs("figures", exist_ok=True)
out = "figures/fig0_concept.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print(f"Saved: {out}")
