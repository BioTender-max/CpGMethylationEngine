"""
CpGMethylationEngine: Whole-Genome Bisulfite Sequencing Analysis Pipeline
- CpG island detection (obs/exp ratio, GC content, length filters)
- Differentially methylated region (DMR) calling (t-test + BH FDR)
- Epigenetic clock (Horvath-style: weighted CpG methylation → age prediction)
- Methylation entropy (epiallele heterogeneity per CpG)
- Tissue-specific methylation signatures
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# ─── DARK THEME DEFAULTS ────────────────────────────────────────────────────
plt.rcParams.update({
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#444444',
    'grid.color': '#333333',
    'font.size': 8,
})

# ─── DATA SIMULATION ────────────────────────────────────────────────────────
N_SAMPLES = 500
N_CPG = 5000
N_NORMAL = 250
N_CANCER = 250

ages = np.concatenate([
    np.random.uniform(20, 80, N_NORMAL),
    np.random.uniform(20, 80, N_CANCER)
])

# Simulate beta values (0-1): cancer has shifted methylation
beta_normal = np.random.beta(2, 5, (N_NORMAL, N_CPG))
beta_cancer = np.random.beta(3, 3, (N_CANCER, N_CPG))

# Inject DMR signal
beta_cancer[:, :150] = np.random.beta(7, 2, (N_CANCER, 150))   # hypermethylated
beta_cancer[:, 150:300] = np.random.beta(1, 7, (N_CANCER, 150))  # hypomethylated

# Age-correlated CpGs
age_weights = np.random.randn(50) * 0.01
for i, w in enumerate(age_weights):
    beta_normal[:, 4000 + i] = np.clip(
        beta_normal[:, 4000 + i] + w * ages[:N_NORMAL], 0, 1
    )

beta_all = np.vstack([beta_normal, beta_cancer])
labels = np.array([0] * N_NORMAL + [1] * N_CANCER)

print("=" * 60)
print("CpGMethylationEngine: WGBS Analysis Pipeline")
print("=" * 60)
print(f"Samples: {N_SAMPLES} ({N_NORMAL} normal, {N_CANCER} cancer)")
print(f"CpG sites: {N_CPG}")

# ─── 1. CpG ISLAND DETECTION ────────────────────────────────────────────────
cpg_positions = np.sort(np.random.randint(0, 3_000_000, N_CPG))
gc_content = np.random.beta(5, 3, N_CPG)
obs_exp_ratio = np.random.beta(3, 2, N_CPG)

island_mask = (obs_exp_ratio > 0.6) & (gc_content > 0.5)
n_islands = island_mask.sum()

n_windows = 30
window_size = 100_000
island_density = np.zeros(n_windows)
for w in range(n_windows):
    w_start = w * window_size
    w_end = (w + 1) * window_size
    in_window = (cpg_positions[island_mask] >= w_start) & (cpg_positions[island_mask] < w_end)
    island_density[w] = in_window.sum()

print(f"\n[CpG Islands]")
print(f"  Detected islands: {n_islands} / {N_CPG} sites ({100*n_islands/N_CPG:.1f}%)")
print(f"  Mean GC content (islands): {gc_content[island_mask].mean():.3f}")
print(f"  Mean obs/exp ratio (islands): {obs_exp_ratio[island_mask].mean():.3f}")

# ─── 2. DMR CALLING ─────────────────────────────────────────────────────────
t_stats = np.zeros(N_CPG)
p_values = np.zeros(N_CPG)
delta_beta = np.zeros(N_CPG)

for j in range(N_CPG):
    t_stat, p_val = stats.ttest_ind(
        beta_cancer[:, j], beta_normal[:, j], equal_var=False
    )
    t_stats[j] = t_stat
    p_values[j] = p_val
    delta_beta[j] = beta_cancer[:, j].mean() - beta_normal[:, j].mean()

def bh_fdr(p_vals):
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    q_vals = p_vals * n / ranks
    q_sorted = q_vals[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_vals[order] = q_sorted
    return np.clip(q_vals, 0, 1)

fdr = bh_fdr(p_values)
dmr_mask = (fdr < 0.05) & (np.abs(delta_beta) > 0.2)
hyper_mask = dmr_mask & (delta_beta > 0)
hypo_mask = dmr_mask & (delta_beta < 0)

n_dmr = dmr_mask.sum()
n_hyper = hyper_mask.sum()
n_hypo = hypo_mask.sum()

print(f"\n[DMR Calling]")
print(f"  Total DMRs (FDR<0.05, |Δβ|>0.2): {n_dmr}")
print(f"  Hypermethylated: {n_hyper}")
print(f"  Hypomethylated: {n_hypo}")
if n_dmr > 0:
    print(f"  Max |Δβ|: {np.abs(delta_beta[dmr_mask]).max():.3f}")

chrom_labels = np.random.choice(np.arange(1, 23), N_CPG)
hyper_per_chrom = np.array([hyper_mask[chrom_labels == c].sum() for c in range(1, 23)])
hypo_per_chrom = np.array([hypo_mask[chrom_labels == c].sum() for c in range(1, 23)])

# ─── 3. EPIGENETIC CLOCK ────────────────────────────────────────────────────
clock_cpg_idx = np.arange(4000, 4050)
X_clock = beta_normal[:, clock_cpg_idx]
A = np.column_stack([X_clock, np.ones(N_NORMAL)])
coeffs, _, _, _ = np.linalg.lstsq(A, ages[:N_NORMAL], rcond=None)
weights = coeffs[:-1]
intercept = coeffs[-1]

X_all_clock = beta_all[:, clock_cpg_idx]
age_pred = X_all_clock @ weights + intercept
age_pred = np.clip(age_pred, 0, 100)

mae = np.mean(np.abs(age_pred - ages))
r_clock, _ = stats.pearsonr(ages, age_pred)

print(f"\n[Epigenetic Clock]")
print(f"  Clock CpGs: 50 (age-correlated)")
print(f"  MAE: {mae:.2f} years")
print(f"  Pearson r: {r_clock:.3f}")

# ─── 4. METHYLATION ENTROPY ─────────────────────────────────────────────────
def methylation_entropy(beta_col):
    bins = np.array([0, 0.25, 0.5, 0.75, 1.0])
    counts, _ = np.histogram(beta_col, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

entropy = np.array([methylation_entropy(beta_all[:, j]) for j in range(N_CPG)])

print(f"\n[Methylation Entropy]")
print(f"  Mean entropy: {entropy.mean():.3f} bits")
print(f"  High-entropy CpGs (>1.5 bits): {(entropy > 1.5).sum()}")

# ─── 5. TISSUE-SPECIFIC SIGNATURES ─────────────────────────────────────────
var_normal = beta_normal.var(axis=0)
var_cancer = beta_cancer.var(axis=0)
top20_normal = np.argsort(var_normal)[-20:]
top20_cancer = np.argsort(var_cancer)[-20:]

# ─── DASHBOARD ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]
for ax in axes:
    ax.set_facecolor('#111111')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

# Panel 1: Beta value distribution violin
ax = axes[0]
data_n = beta_normal[:, :500].flatten()[::10]
data_c = beta_cancer[:, :500].flatten()[::10]
parts = ax.violinplot([data_n, data_c], positions=[1, 2], showmedians=True)
colors = ['#4fc3f7', '#ef5350']
for pc, col in zip(parts['bodies'], colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.7)
for key in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
    if key in parts:
        parts[key].set_color('white')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Normal', 'Cancer'], color='white')
ax.set_ylabel('Beta Value', color='white')
ax.set_title('Beta Value Distribution', color='white', fontsize=9, fontweight='bold')
ax.set_ylim(0, 1)

# Panel 2: DMR Volcano
ax = axes[1]
log10_fdr = -np.log10(fdr + 1e-300)
non_dmr = ~dmr_mask
ax.scatter(delta_beta[non_dmr], log10_fdr[non_dmr], c='#555555', s=1, alpha=0.4)
ax.scatter(delta_beta[hyper_mask], log10_fdr[hyper_mask], c='#ef5350', s=3, alpha=0.8, label=f'Hyper ({n_hyper})')
ax.scatter(delta_beta[hypo_mask], log10_fdr[hypo_mask], c='#4fc3f7', s=3, alpha=0.8, label=f'Hypo ({n_hypo})')
ax.axhline(-np.log10(0.05), color='#ffeb3b', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axvline(0.2, color='#aaaaaa', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(-0.2, color='#aaaaaa', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Δβ (Cancer - Normal)', color='white')
ax.set_ylabel('-log10(FDR)', color='white')
ax.set_title('DMR Volcano Plot', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white', framealpha=0.8)

# Panel 3: CpG island methylation heatmap (top 30 DMRs)
ax = axes[2]
top30_dmr = np.argsort(np.abs(delta_beta))[-30:]
sample_idx = np.linspace(0, N_SAMPLES - 1, 50, dtype=int)
hm = ax.imshow(beta_all[sample_idx, :][:, top30_dmr].T, aspect='auto',
               cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
ax.axvline(24.5, color='white', linewidth=1, alpha=0.7)
ax.set_xlabel('Samples (Normal | Cancer)', color='white')
ax.set_ylabel('Top 30 DMRs', color='white')
ax.set_title('DMR Methylation Heatmap', color='white', fontsize=9, fontweight='bold')
plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)

# Panel 4: Epigenetic clock scatter
ax = axes[3]
ax.scatter(ages[:N_NORMAL], age_pred[:N_NORMAL], c='#4fc3f7', s=8, alpha=0.6, label='Normal')
ax.scatter(ages[N_NORMAL:], age_pred[N_NORMAL:], c='#ef5350', s=8, alpha=0.6, label='Cancer')
lims = [15, 85]
ax.plot(lims, lims, 'w--', linewidth=1, alpha=0.5)
ax.set_xlabel('Chronological Age', color='white')
ax.set_ylabel('Predicted Age', color='white')
ax.set_title(f'Epigenetic Clock (r={r_clock:.3f}, MAE={mae:.1f}y)', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white')
ax.set_xlim(lims); ax.set_ylim(lims)

# Panel 5: Methylation entropy distribution
ax = axes[4]
ax.hist(entropy, bins=50, color='#ab47bc', alpha=0.8, edgecolor='none')
ax.axvline(entropy.mean(), color='#ffeb3b', linestyle='--', linewidth=1.5,
           label=f'Mean={entropy.mean():.2f}')
ax.set_xlabel('Entropy (bits)', color='white')
ax.set_ylabel('Count', color='white')
ax.set_title('Methylation Entropy Distribution', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 6: Hyper/Hypo DMR by chromosome
ax = axes[5]
chroms = np.arange(1, 23)
x = np.arange(22)
width = 0.4
ax.bar(x - width/2, hyper_per_chrom, width, color='#ef5350', alpha=0.8, label='Hyper')
ax.bar(x + width/2, hypo_per_chrom, width, color='#4fc3f7', alpha=0.8, label='Hypo')
ax.set_xticks(x[::2])
ax.set_xticklabels([f'chr{c}' for c in chroms[::2]], rotation=45, fontsize=6)
ax.set_ylabel('DMR Count', color='white')
ax.set_title('DMRs by Chromosome', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 7: Top 20 tissue-specific CpGs heatmap
ax = axes[6]
combined_idx = np.concatenate([top20_normal, top20_cancer])
combined_data = beta_all[:, combined_idx]
sample_idx2 = np.concatenate([
    np.linspace(0, N_NORMAL - 1, 25, dtype=int),
    np.linspace(N_NORMAL, N_SAMPLES - 1, 25, dtype=int)
])
hm2 = ax.imshow(combined_data[sample_idx2, :].T, aspect='auto', cmap='coolwarm',
                vmin=0, vmax=1, interpolation='nearest')
ax.axvline(24.5, color='white', linewidth=1, alpha=0.7)
ax.set_xlabel('Samples', color='white')
ax.set_ylabel('Variable CpGs', color='white')
ax.set_title('Tissue-Specific CpG Signatures', color='white', fontsize=9, fontweight='bold')
plt.colorbar(hm2, ax=ax, fraction=0.046, pad=0.04)

# Panel 8: CpG island density plot
ax = axes[7]
window_centers = np.arange(n_windows) * window_size / 1e6
ax.fill_between(window_centers, island_density, color='#26a69a', alpha=0.7)
ax.plot(window_centers, island_density, color='#80cbc4', linewidth=1)
ax.set_xlabel('Genomic Position (Mb)', color='white')
ax.set_ylabel('CpG Island Count', color='white')
ax.set_title('CpG Island Density', color='white', fontsize=9, fontweight='bold')

# Panel 9: Summary text
ax = axes[8]
ax.axis('off')
max_db = np.abs(delta_beta[dmr_mask]).max() if n_dmr > 0 else 0.0
summary_lines = [
    "CpGMethylationEngine Summary",
    "─" * 32,
    f"Samples: {N_SAMPLES} ({N_NORMAL}N / {N_CANCER}C)",
    f"CpG Sites: {N_CPG:,}",
    f"CpG Islands: {n_islands} ({100*n_islands/N_CPG:.1f}%)",
    f"Total DMRs: {n_dmr}",
    f"  Hyper: {n_hyper}  |  Hypo: {n_hypo}",
    f"Max |delta_beta|: {max_db:.3f}",
    f"Epigenetic Clock MAE: {mae:.2f} yrs",
    f"Clock Pearson r: {r_clock:.3f}",
    f"Mean Entropy: {entropy.mean():.3f} bits",
    f"High-entropy CpGs: {(entropy > 1.5).sum()}",
]
y_pos = 0.95
for line in summary_lines:
    color = '#4fc3f7' if line.startswith('CpG') else 'white'
    ax.text(0.05, y_pos, line, transform=ax.transAxes,
            fontsize=8, color=color,
            verticalalignment='top', fontfamily='monospace')
    y_pos -= 0.075

fig.suptitle('CpGMethylationEngine: WGBS Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/cpg_methylation_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0a')
plt.close()

print(f"\n[Dashboard] Saved: /workspace/cpg_methylation_dashboard.png")
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  CpG Islands detected:     {n_islands} ({100*n_islands/N_CPG:.1f}%)")
print(f"  Total DMRs:               {n_dmr}")
print(f"  Hypermethylated DMRs:     {n_hyper}")
print(f"  Hypomethylated DMRs:      {n_hypo}")
print(f"  Max |delta_beta|:         {max_db:.4f}")
print(f"  Epigenetic Clock MAE:     {mae:.2f} years")
print(f"  Epigenetic Clock r:       {r_clock:.4f}")
print(f"  Mean Methylation Entropy: {entropy.mean():.4f} bits")
print(f"  High-entropy CpGs:        {(entropy > 1.5).sum()}")
print("=" * 60)
