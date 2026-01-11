"""
AD-DIFFI Chapter 5 Simulation: RSO DIFFI vs AD-DIFFI (Z-Standardization)
Synthetic mixed-type data (5 signal + 3 noise features) evaluation.

Reproduces Chapter 5 figures and tables.
Requires: ad_diffi.py (core functions)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ad_diffi import (
    N_ITER, N_ITER_NOISE, FEATURE_DEFINITIONS, FEATURE_NAMES, FEATURE_TYPES,
    run_noise_only_rso_simulation, diffi_score_collection_and_standardize_rso
)

def plot_rso_standardization_comparison(combined_df: pd.DataFrame, feature_defs: List):
    """Generate comparison boxplot: Raw DIFFI RSO vs AD-DIFFI RSO (Z-scores)."""
    # Feature labels for plot
    feature_labels = {
        'X1_Strong_Cont': 'X1 (S Cont.)', 'X2_Weak_Cont_A': 'X2 (W Cont. A)',
        'X3_Weak_Cont_B': 'X3 (W Cont. B)', 'X4_Strong_Bin': 'X4 (S Bin.)',
        'X5_Weak_Bin': 'X5 (W Bin.)', 'X6_Cont_Noise': 'X6 (Cont. Noise)',
        'X7_Bin_Noise_A': 'X7 (Bin. Noise A)', 'X8_Bin_Noise_B': 'X8 (Bin. Noise B)',
    }
    df_plot = combined_df[['Scenario'] + FEATURE_NAMES].melt(id_vars='Scenario', var_name='Feature', value_name='Score')
    df_plot['Feature_Label'] = df_plot['Feature'].map(feature_labels)
    feature_order = [feature_labels[f['name']] for f in feature_defs]
    scenario_order = ['1_Raw_Score (DIFFI RSO)', '2_Z_Score (AD-DIFFI RSO)']

    plt.figure(figsize=(16, 8))

    sns.boxplot(
        x='Feature_Label', y='Score', hue='Scenario',
        data=df_plot, order=feature_order, hue_order=scenario_order,
        palette={'1_Raw_Score (DIFFI RSO)': '#1f77b4', '2_Z_Score (AD-DIFFI RSO)': '#ff7f0e'},
        width=0.7, showfliers=False, ax=plt.gca()
    )

    # Reference lines
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Threshold (1.0 for Raw)')
    plt.axhline(0.0, color='k', linestyle=':', linewidth=1.5, label='Threshold (0.0 for Z/AD-DIFFI)')

    plt.title(f'Chapter 5: DIFFI RSO vs AD-DIFFI RSO (N={N_ITER} Iterations)', fontsize=14)
    plt.ylabel('Score Value', fontsize=12)
    plt.xlabel('Feature (Type & Strength)', fontsize=12)
    plt.xticks(rotation=20, ha='right')

    # Dynamic y-limits
    z_score_min = df_plot[df_plot['Scenario'] == '2_Z_Score (AD-DIFFI RSO)']['Score'].min()
    score_max = df_plot['Score'].max()
    y_min_adjusted = z_score_min * 1.1 if z_score_min < 0 else z_score_min * 0.9
    y_max_adjusted = score_max * 1.1
    plt.ylim(y_min_adjusted, y_max_adjusted)

    plt.legend(title='Evaluation Metric', loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

# --- Main Execution (Chapter 5 Reproduction) ---
if __name__ == '__main__':
    print("=== AD-DIFFI Chapter 5 Simulation: RSO + Z-Normalization ===")
    
    # Step 1: Establish noise baselines (type-specific)
    cont_mean, cont_sd, bin_mean, bin_sd = run_noise_only_rso_simulation(
        N_ITER_NOISE, FEATURE_TYPES
    )
    
    # Step 2: Collect signal scores and compute AD-DIFFI (Z-scores)
    df_results = diffi_score_collection_and_standardize_rso(
        N_ITER, FEATURE_TYPES, cont_mean, cont_sd, bin_mean, bin_sd
    )
    
    # Step 3: Generate and save plot
    plot_rso_standardization_comparison(df_results, FEATURE_DEFINITIONS)
    plt.savefig("ch5_ad_diffi_rso_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Step 4: Summary table
    avg_scores = df_results.groupby(['Scenario', 'Feature_Label'])['Score'].agg(['mean', 'std']).reset_index()
    # Note: Feature_Label mapping inside plot func; recompute here for table
    feature_labels = {
        'X1_Strong_Cont': 'X1 (S Cont.)', 'X2_Weak_Cont_A': 'X2 (W Cont. A)',
        'X3_Weak_Cont_B': 'X3 (W Cont. B)', 'X4_Strong_Bin': 'X4 (S Bin.)',
        'X5_Weak_Bin': 'X5 (W Bin.)', 'X6_Cont_Noise': 'X6 (Cont. Noise)',
        'X7_Bin_Noise_A': 'X7 (Bin. Noise A)', 'X8_Bin_Noise_B': 'X8 (Bin. Noise B)',
    }
    df_results['Feature_Label'] = df_results['Feature'].map(feature_labels) if 'Feature' in df_results else None  # Adjust if needed
    avg_scores = df_results.groupby(['Scenario', df_results['Feature'].map(feature_labels)])['Score'].agg(['mean', 'std']).reset_index()
    avg_scores = avg_scores.pivot(index=avg_scores.columns[1], columns='Scenario', values=['mean', 'std'])
    feature_sort_order = [feature_labels[f['name']] for f in FEATURE_DEFINITIONS]
    avg_scores = avg_scores.reindex(feature_sort_order)

    print("\n### Chapter 5 Summary: DIFFI RSO Raw vs AD-DIFFI RSO (Z-Score) ###")
    print(avg_scores.round(4).to_markdown())
    
    print("\nExecution complete. Figure saved: ch5_ad_diffi_rso_comparison.png")
    print("Ready for paper supplementary materials.")
