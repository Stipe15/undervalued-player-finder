"""
Explains the XGBoost market-value model's predictions with SHAP.

SHAP values are computed out-of-fold, using the same 5-fold split as
main.py, so a player's explanation always comes from a model that never
saw that player - matching how the undervalued/overvalued rankings work.

Outputs (outputs/shap/):
  shap_values_all_players.csv   one row per player, one column per feature
  global_importance_bar.png     mean |SHAP| per feature, top 20
  global_beeswarm.png           direction + magnitude of each feature's effect
  top10_undervalued.csv         biggest (predicted - actual) / actual
  top10_overvalued.csv          smallest (predicted - actual) / actual
  undervalued/NN_<name>.png     waterfall chart per undervalued player
  overvalued/NN_<name>.png      waterfall chart per overvalued player

Run with: python shap_analysis.py
Draw one extra player's waterfall on demand:
  python shap_analysis.py --player "Kylian Mbappe"
"""

import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import KFold

from main import RANDOM_STATE, load_data, make_xgb_model

OUT_DIR = "outputs/shap"
N_TOP = 10


def safe_filename(name):
    name = name.replace(" ", "_").replace("/", "-")
    return re.sub(r"[^\w\-.]", "", name, flags=re.UNICODE)


def compute_oof_shap(X, y):
    """Fit XGBoost per fold and explain only that fold's held-out players.

    Returns (shap_values [n_players, n_features], base_values [n_players],
    oof_pred_log [n_players]).
    """
    n = len(X)
    n_features = X.shape[1]
    shap_values = np.zeros((n, n_features))
    base_values = np.zeros(n)
    oof_pred_log = np.zeros(n)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        model = make_xgb_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        explainer = shap.TreeExplainer(model)
        explanation = explainer(X.iloc[test_idx])

        shap_values[test_idx] = explanation.values
        base_values[test_idx] = explanation.base_values
        oof_pred_log[test_idx] = model.predict(X.iloc[test_idx])

        print(f"Fold {fold}: explained {len(test_idx)} held-out players")

    return shap_values, base_values, oof_pred_log


def save_global_plots(shap_values, X):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/global_importance_bar.png", dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/global_beeswarm.png", dpi=150)
    plt.close()


def save_player_waterfall(path, name, actual, predicted, shap_row, base_value, feature_names, feature_values):
    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=feature_values,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(f"{name}\nActual €{actual:.1f}m vs predicted €{predicted:.1f}m", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def top_contributions(shap_row, feature_names, k=3):
    order = np.argsort(shap_row)
    top_pos = [(feature_names[i], shap_row[i]) for i in order[::-1][:k]]
    top_neg = [(feature_names[i], shap_row[i]) for i in order[:k]]
    return top_pos, top_neg


def build_ranked_table(df, oof_pred_log):
    actual = df["Market Value (m/€)"].to_numpy()
    predicted = np.expm1(oof_pred_log)
    gap = (predicted - actual) / actual

    table = pd.DataFrame({
        "NAME": df["NAME"],
        "POSITION": df["POSITION"],
        "LEAGUE": df["LEAGUE"],
        "actual_eur_m": actual,
        "predicted_eur_m": predicted,
        "gap_pct": gap * 100,
    })
    return table, gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", help="Draw an on-demand waterfall chart for this player name")
    args = parser.parse_args()

    import os
    os.makedirs(f"{OUT_DIR}/undervalued", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/overvalued", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/players", exist_ok=True)

    df, X, y = load_data()
    feature_names = list(X.columns)

    print("\nComputing out-of-fold SHAP values (5 folds)...")
    shap_values, base_values, oof_pred_log = compute_oof_shap(X, y)

    # Additivity check: base + sum(shap) must equal the raw model output (log scale).
    reconstructed = base_values + shap_values.sum(axis=1)
    max_diff = np.abs(reconstructed - oof_pred_log).max()
    assert max_diff < 1e-4, f"SHAP additivity check failed: max diff {max_diff}"
    print(f"Additivity check passed (max diff {max_diff:.2e})")

    ranked, gap = build_ranked_table(df, oof_pred_log)

    # --- CSV with every player's SHAP values -------------------------------
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    all_players = pd.concat([ranked, pd.Series(base_values, name="base_value_log"), shap_df], axis=1)
    all_players.to_csv(f"{OUT_DIR}/shap_values_all_players.csv", index=False)
    print(f"Wrote {len(all_players)} rows -> {OUT_DIR}/shap_values_all_players.csv")

    # --- Global plots --------------------------------------------------------
    save_global_plots(shap_values, X)
    print(f"Wrote global_importance_bar.png and global_beeswarm.png -> {OUT_DIR}/")

    # --- Top 10 undervalued / overvalued -------------------------------------
    undervalued_idx = np.argsort(gap)[::-1][:N_TOP]
    overvalued_idx = np.argsort(gap)[:N_TOP]

    for label, idx_list, subdir in [
        ("undervalued", undervalued_idx, "undervalued"),
        ("overvalued", overvalued_idx, "overvalued"),
    ]:
        rows = []
        for rank, idx in enumerate(idx_list, start=1):
            name = df["NAME"].iloc[idx]
            actual = df["Market Value (m/€)"].iloc[idx]
            predicted = np.expm1(oof_pred_log[idx])
            top_pos, top_neg = top_contributions(shap_values[idx], feature_names)

            fname = f"{OUT_DIR}/{subdir}/{rank:02d}_{safe_filename(name)}.png"
            save_player_waterfall(
                fname, name, actual, predicted,
                shap_values[idx], base_values[idx], feature_names, X.iloc[idx].values,
            )

            rows.append({
                "rank": rank,
                "NAME": name,
                "actual_eur_m": actual,
                "predicted_eur_m": predicted,
                "gap_pct": gap[idx] * 100,
                "top_positive_1": f"{top_pos[0][0]}={top_pos[0][1]:.3f}",
                "top_positive_2": f"{top_pos[1][0]}={top_pos[1][1]:.3f}",
                "top_positive_3": f"{top_pos[2][0]}={top_pos[2][1]:.3f}",
                "top_negative_1": f"{top_neg[0][0]}={top_neg[0][1]:.3f}",
                "top_negative_2": f"{top_neg[1][0]}={top_neg[1][1]:.3f}",
                "top_negative_3": f"{top_neg[2][0]}={top_neg[2][1]:.3f}",
            })

        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/top10_{label}.csv", index=False)
        print(f"Wrote top10_{label}.csv and {N_TOP} waterfall charts -> {OUT_DIR}/{subdir}/")

    # --- On-demand single-player chart ---------------------------------------
    if args.player:
        matches = df.index[df["NAME"].str.lower() == args.player.lower()]
        if len(matches) == 0:
            print(f"\nNo player named '{args.player}' found.", file=sys.stderr)
            sys.exit(1)
        idx = matches[0]
        name = df["NAME"].iloc[idx]
        actual = df["Market Value (m/€)"].iloc[idx]
        predicted = np.expm1(oof_pred_log[idx])
        fname = f"{OUT_DIR}/players/{safe_filename(name)}.png"
        save_player_waterfall(
            fname, name, actual, predicted,
            shap_values[idx], base_values[idx], feature_names, X.iloc[idx].values,
        )
        print(f"\nWrote {fname}")


if __name__ == "__main__":
    main()
