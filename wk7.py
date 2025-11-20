"""
COMPAS Bias Audit using IBM AIF360
Saves visualizations: fpr_by_race.png and rates_by_race.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# AIF360 imports
from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# Silence warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def load_compas():
    # Load the COMPAS dataset included in AIF360
    compas = CompasDataset()
    return compas

def prepare_data(aif_dataset):
    # Convert AIF360 dataset to pandas DataFrame (features + labels + protected attribute)
    df = aif_dataset.convert_to_dataframe()[0]
    return df

def train_baseline(X_train, y_train):
    # Simple baseline pipeline: scaling + logistic regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def main(save_plots=True):
    # 1. Load dataset
    compas = load_compas()
    # compas.features - features names, compas.protected_attribute_names etc.
    print("AIF360 COMPAS dataset loaded.")
    print("Available protected attributes:", compas.protected_attribute_names)
    print("Label names:", compas.favorable_label, compas.unfavorable_label)

    # 2. Convert to pandas and inspect
    df = prepare_data(compas)
    # The dataset includes columns: attributes, label, and protected attributes (e.g., race)
    # We'll treat 'race' as protected and 'two_year_recid' or the target label provided.
    # AIF360 typically uses 'sex' and 'race' as protected; label column name is compas.label_names[0]
    label_name = compas.label_names[0]  # e.g., 'two_year_recid'
    protected_attr = 'race'  # field contained in dataframe

    # 3. Feature selection: use AIF360-provided features (drop label and protected columns)
    # Filter out meta columns from compas.metadata if present
    cols_to_exclude = [label_name, 'race', 'sex', 'age', 'age_cat', 'is_recid', 'c_charge_degree', 'decile_score', 'id']
    # Build feature list as numeric columns excluding label and protected attributes
    feature_cols = [c for c in df.columns if c not in cols_to_exclude and df[c].dtype in [np.int64, np.float64, np.int32, np.float32]]
    if not feature_cols:
        # fallback: use all columns except label and protected
        feature_cols = [c for c in df.columns if c not in [label_name, 'race', 'sex']]

    # 4. Train/test split
    X = df[feature_cols]
    y = df[label_name].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df.index, test_size=0.3, random_state=42, stratify=y)

    clf = train_baseline(X_train, y_train)

    # 5. Predictions
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # 6. Build AIF360 BinaryLabelDataset for test set (required for metrics)
    # Reconstruct the matrix for AIF360: the dataset expects numpy arrays for features and protected attributes
    # We'll create a small helper to map the pandas split back into an AIF360 BinaryLabelDataset
    # Start by taking the rows from original aif_dataset corresponding to idx_test
    test_bld = compas.subset(idx_test.tolist())

    # Replace labels in test_bld with our model's predictions so we can compute classification metrics
    # Create a copy to hold predictions as labels
    pred_bld = test_bld.copy()
    pred_bld.labels = y_pred.reshape(-1, 1)

    # 7. Compute classification metrics using AIF360
    metric = ClassificationMetric(test_bld, pred_bld,
                                  unprivileged_groups=[{'race': 0}],  # NOTE: mapping depends on how race is encoded
                                  privileged_groups=[{'race': 1}])

    # Because race encoding may not be 0/1 in dataset, compute rates by grouping on the original DataFrame:
    test_df = df.loc[idx_test].copy()
    test_df['y_true'] = y_test.values
    test_df['y_pred'] = y_pred

    # Map race values — show unique race values
    print("Unique race values in test set:", test_df['race'].unique())

    # Compute FPR, TPR, FNR by race
    group_stats = []
    for grp in sorted(test_df['race'].unique()):
        grp_df = test_df[test_df['race'] == grp]
        tp = ((grp_df['y_true'] == 1) & (grp_df['y_pred'] == 1)).sum()
        tn = ((grp_df['y_true'] == 0) & (grp_df['y_pred'] == 0)).sum()
        fp = ((grp_df['y_true'] == 0) & (grp_df['y_pred'] == 1)).sum()
        fn = ((grp_df['y_true'] == 1) & (grp_df['y_pred'] == 0)).sum()
        pos = (grp_df['y_true'] == 1).sum()
        neg = (grp_df['y_true'] == 0).sum()

        fpr = fp / neg if neg > 0 else np.nan
        tpr = tp / pos if pos > 0 else np.nan
        fnr = fn / pos if pos > 0 else np.nan
        fdr = fp / (fp + tp) if (fp + tp) > 0 else np.nan
        group_stats.append({
            'race': grp,
            'count': len(grp_df),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'TPR': tpr, 'FPR': fpr, 'FNR': fnr, 'FDR': fdr
        })

    stats_df = pd.DataFrame(group_stats).sort_values('race')
    print("\nPer-race confusion stats:\n", stats_df)

    # 8. Visualizations
    sns.set(style="whitegrid")
    # FPR by race
    plt.figure(figsize=(8,5))
    sns.barplot(data=stats_df, x='race', y='FPR')
    plt.title('False Positive Rate (FPR) by Race')
    plt.xlabel('Race')
    plt.ylabel('False Positive Rate')
    plt.ylim(0, stats_df['FPR'].max()*1.2 if stats_df['FPR'].max() < 1 else 1)
    plt.tight_layout()
    if save_plots:
        plt.savefig('fpr_by_race.png', dpi=300)
    plt.show()

    # Grouped rates: TPR / FPR / FNR
    rates_df = stats_df.melt(id_vars=['race'], value_vars=['TPR','FPR','FNR'], var_name='rate_type', value_name='rate')
    plt.figure(figsize=(10,6))
    sns.barplot(data=rates_df, x='race', y='rate', hue='rate_type')
    plt.title('TPR / FPR / FNR by Race')
    plt.xlabel('Race')
    plt.ylabel('Rate')
    plt.legend(title='Rate Type')
    plt.tight_layout()
    if save_plots:
        plt.savefig('rates_by_race.png', dpi=300)
    plt.show()

    # 9. Disparate impact and statistical parity difference using BinaryLabelDatasetMetric
    bld_test = test_bld  # original test label dataset
    metric_bld = BinaryLabelDatasetMetric(bld_test, unprivileged_groups=[{'race': 0}], privileged_groups=[{'race': 1}])
    # Note: these metrics use AIF360's encoding assumptions; handle them cautiously
    try:
        stat_par_diff = metric_bld.statistical_parity_difference()
        disp_impact = metric_bld.disparate_impact()
        print(f"\nStatistical parity difference (unpriv - priv): {stat_par_diff:.4f}")
        print(f"Disparate impact (unpriv/priv): {disp_impact:.4f}")
    except Exception as e:
        print("Unable to compute AIF360 group-level metrics due to encoding assumptions. Error:", e)

    # Print interpretation assistance
    print("\nInterpretation guidance:")
    print("- Higher FPR for a group means they are more likely to be falsely labeled as 'at-risk' (false positives).")
    print("- Compare FPR across races to detect disparate impact in policing / pre-trial decisions.")
    print("- Statistical parity difference near 0 indicates parity; negative values show disadvantage for unprivileged group.")
    print("- Disparate impact < 0.8 is often considered a sign of adverse impact (80% rule) but context matters.")

if __name__ == "__main__":
    main()
