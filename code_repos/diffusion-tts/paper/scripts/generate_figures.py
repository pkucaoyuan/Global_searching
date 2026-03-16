#!/usr/bin/env python3
"""Generate paper figures as PDF with CMU Serif font."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Global font setup: CMU Serif ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,      # TrueType in PDF
    'ps.fonttype': 42,
})

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTDIR, exist_ok=True)


def fig_stopping_decision():
    """Figure 1: Online stopping decision in the (sigma_t, a_K') plane."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # Boundaries
    sig_th = 0.38   # beta_sigma * bar_sigma  (normalized)
    ak_th  = 0.35   # gain threshold (normalized)

    # Shaded STOP region
    ax.fill_between([0, sig_th], 0, ak_th, color='#c8e6c9', alpha=0.7, zorder=1)

    # Threshold lines
    ax.axvline(sig_th, color='#1565c0', ls='--', lw=1.5, zorder=3)
    ax.axhline(ak_th, color='#c62828', ls='--', lw=1.5, zorder=3)

    # Threshold labels
    ax.text(sig_th, 1.02, r'$\beta_\sigma \bar{\sigma}$',
            ha='center', va='bottom', fontsize=11, color='#1565c0',
            transform=ax.get_xaxis_transform())
    ax.text(1.02, ak_th, r'$\frac{\beta_g \bar{g}}{\sigma_t}$',
            ha='left', va='center', fontsize=11, color='#c62828',
            transform=ax.get_yaxis_transform())

    # Hyperbola: sigma * a_K' = lambda*
    sig_vals = np.linspace(0.08, 0.98, 300)
    lam_star = 0.14
    ak_vals = lam_star / sig_vals
    mask = ak_vals < 0.98
    ax.plot(sig_vals[mask], ak_vals[mask], color='#757575', lw=1.5, zorder=2)
    # Label the hyperbola
    ax.text(0.72, 0.22, r'$\sigma_t \cdot a_K^\prime = \lambda^*$',
            fontsize=9, color='#757575', rotation=-18, ha='center')

    # ── Region labels ──
    # Bottom-left: STOP
    ax.text(sig_th/2, ak_th/2 + 0.04, 'STOP', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#2e7d32', zorder=5)
    ax.text(sig_th/2, ak_th/2 - 0.07, 'low sensitivity\nlow marginal gain',
            fontsize=7.5, ha='center', va='center', color='#388e3c', zorder=5)

    # Top-left: Continue
    ax.text(sig_th/2, (1+ak_th)/2 + 0.06, 'Continue', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#1565c0', zorder=5)
    ax.text(sig_th/2, (1+ak_th)/2 - 0.07, 'insensitive, but\nsearch still productive',
            fontsize=7.5, ha='center', va='center', color='#1976d2', zorder=5)

    # Bottom-right: Continue
    ax.text((1+sig_th)/2, ak_th/2 + 0.04, 'Continue', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#c62828', zorder=5)
    ax.text((1+sig_th)/2, ak_th/2 - 0.07, 'responsive but exhausted;\nlucky draw still possible',
            fontsize=7.5, ha='center', va='center', color='#d32f2f', zorder=5)

    # Top-right: Continue
    ax.text((1+sig_th)/2, (1+ak_th)/2 + 0.06, 'Continue', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#424242', zorder=5)
    ax.text((1+sig_th)/2, (1+ak_th)/2 - 0.07, 'high sensitivity &\nhigh marginal gain',
            fontsize=7.5, ha='center', va='center', color='#616161', zorder=5)

    # Axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\sigma_t$ (sensitivity)', fontsize=11)
    ax.set_ylabel(r"$a_K'$ (exhaustion)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout(pad=0.3)
    out = os.path.join(OUTDIR, 'fig_stopping_decision.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


def fig_architecture():
    """Figure 2: Two-level view — global scheduler allocates per-step budget to local search."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.set_xlim(-0.8, 11.2)
    ax.set_ylim(-1.0, 4.8)
    ax.axis('off')

    # ── Colors ──
    c_global = '#1565c0'
    c_local  = '#e65100'
    c_traj   = '#424242'
    c_bg_local = '#fff3e0'
    c_bar    = '#90caf9'
    c_bar_edge = '#1565c0'

    # ── Global scheduler box (top, simple) ──
    g_box = FancyBboxPatch((1.5, 3.2), 8.0, 1.2, boxstyle="round,pad=0.15",
                            facecolor='#e3f2fd', edgecolor=c_global, lw=1.8)
    ax.add_patch(g_box)
    ax.text(5.5, 3.8, 'Global Scheduler  $\\mathcal{G}$',
            fontsize=12, fontweight='bold', ha='center', va='center', color=c_global)

    # Budget B entering from left
    ax.annotate('', xy=(1.45, 3.8), xytext=(0.3, 3.8),
                arrowprops=dict(arrowstyle='->', color=c_global, lw=1.5))
    ax.text(0.15, 3.8, '$B$', fontsize=12, ha='center', va='center',
            color=c_global, fontweight='bold')

    # ── Denoising trajectory (bottom row) ──
    x_positions = [1.5, 3.5, 5.5, 7.5, 9.5]
    t_labels = ['$t_T$', '$t_{T\\!-\\!1}$', '$\\cdots$', '$t_2$', '$t_1$']
    # Non-uniform budget heights — key visual message
    bar_heights = [0.45, 1.05, 0, 0.75, 0.30]
    k_labels = ['$K_T$', '$K_{T-1}$', '', '$K_2$', '$K_1$']

    for i, (xp, tl, kl, bh) in enumerate(zip(x_positions, t_labels, k_labels, bar_heights)):
        if i == 2:
            ax.text(xp, 0.5, '$\\cdots$', fontsize=16, ha='center', va='center',
                    color=c_traj)
            continue

        # Local search box
        bw = 1.3
        local_box = FancyBboxPatch((xp - bw/2, -0.15), bw, 1.15,
                                    boxstyle="round,pad=0.07",
                                    facecolor=c_bg_local, edgecolor=c_local, lw=1.2)
        ax.add_patch(local_box)
        ax.text(xp, 0.42, r'$\mathcal{L}_t$', fontsize=12, ha='center',
                va='center', color=c_local)
        ax.text(xp, -0.0, tl, fontsize=9, ha='center', va='center', color=c_traj)

        # Budget bar above local box (non-uniform height = key message)
        bar_bottom = 1.15
        bar = FancyBboxPatch((xp - 0.3, bar_bottom), 0.6, bh,
                              boxstyle="round,pad=0.02",
                              facecolor=c_bar, edgecolor=c_bar_edge,
                              lw=0.8, alpha=0.8)
        ax.add_patch(bar)
        ax.text(xp, bar_bottom + bh + 0.08, kl, fontsize=8, ha='center',
                va='bottom', color=c_global)

        # Dashed arrow from global down to bar
        ax.annotate('', xy=(xp, bar_bottom + bh), xytext=(xp, 3.15),
                    arrowprops=dict(arrowstyle='->', color=c_global,
                                    lw=1.0, ls='--', alpha=0.5))

    # ── Trajectory arrows (horizontal) ──
    arrow_y = 0.42
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for (a, b) in pairs:
        xa = x_positions[a]
        xb = x_positions[b]
        gap = 0.7 if a != 1 and b != 3 else 0.5
        x_s = xa + gap
        x_e = xb - gap
        if a == 1:  # before dots
            x_s = xa + 0.7
            x_e = xb - 0.35
        if a == 2:  # after dots
            x_s = xa + 0.35
            x_e = xb - 0.7
        ax.annotate('', xy=(x_e, arrow_y), xytext=(x_s, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=c_traj, lw=1.0))

    # x_T and x_0 labels at ends
    ax.text(0.5, 0.42, '$x_T$', fontsize=10, ha='center', va='center', color=c_traj)
    ax.annotate('', xy=(0.8, arrow_y), xytext=(0.65, arrow_y),
                arrowprops=dict(arrowstyle='->', color=c_traj, lw=1.0))
    ax.text(10.5, 0.42, '$x_0$', fontsize=10, ha='center', va='center', color=c_traj)
    ax.annotate('', xy=(10.35, arrow_y), xytext=(10.2, arrow_y),
                arrowprops=dict(arrowstyle='->', color=c_traj, lw=1.0))

    # ── Level labels on the sides ──
    ax.text(-0.6, 3.8, 'Global', fontsize=9, ha='center', va='center',
            color=c_global, fontstyle='italic', rotation=90)
    ax.text(-0.6, 0.5, 'Local', fontsize=9, ha='center', va='center',
            color=c_local, fontstyle='italic', rotation=90)

    fig.tight_layout(pad=0.2)
    out = os.path.join(OUTDIR, 'fig_architecture.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


if __name__ == '__main__':
    print('Generating figures with CMU Serif...')
    fig_stopping_decision()
    fig_architecture()
    print('Done.')
