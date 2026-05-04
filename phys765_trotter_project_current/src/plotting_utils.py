"""Plotting helpers for the Physics 765 Trotter study."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _coerce_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    if "resolved" in plot_df.columns:
        plot_df["resolved"] = plot_df["resolved"].astype("boolean")
    for col in [
        "Delta",
        "r_star",
        "r_value_for_plot",
        "t_count_estimate_total",
        "worst_case_rule_r",
        "low_energy_rule_r",
    ]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    return plot_df


def plot_rstar_vs_delta(df: pd.DataFrame, ax=None):
    plot_df = _coerce_plot_columns(df)
    if "resolved" not in plot_df.columns or "Delta" not in plot_df.columns:
        raise ValueError("plot_rstar_vs_delta requires 'resolved' and 'Delta' columns")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    for family, sub in plot_df.groupby("family"):
        resolved = sub[(sub["resolved"] == True) & sub["Delta"].notna() & sub["r_star"].notna()]
        censored = sub[(sub["resolved"] == False) & sub["Delta"].notna() & sub["r_value_for_plot"].notna()]
        if not resolved.empty:
            ax.scatter(resolved["Delta"], resolved["r_star"], label=f"{family} (resolved)", alpha=0.85)
        if not censored.empty:
            ax.scatter(
                censored["Delta"],
                censored["r_value_for_plot"],
                label=f"{family} (censored)",
                alpha=0.75,
                marker="x",
            )
    ax.set_xlabel(r"$\Delta = \langle \psi_0 | H | \psi_0 \rangle$")
    ax.set_ylabel(r"Empirical threshold $r^*$ (censored points marked)")
    ax.set_title(r"Empirical threshold scan vs initial energy")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_error_curves(curves: Sequence[dict], ax=None, yscale: str = "log"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    for row in curves:
        ax.plot(row["r_values"], row["errors"], marker="o", markersize=3, label=row["label"])
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$1 - |\langle \psi_{\mathrm{exact}} | \psi_{\mathrm{trot}} \rangle|^2$")
    ax.set_title("Representative fidelity-error curves")
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_resource_vs_delta(df: pd.DataFrame, resource_col: str = "t_count_estimate_total", ax=None):
    plot_df = _coerce_plot_columns(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    family_order = list(dict.fromkeys(plot_df["family"].tolist())) if "family" in plot_df.columns else []
    usable = plot_df[plot_df[resource_col].notna() & plot_df["Delta"].notna()].copy()
    omitted = []
    for family in family_order:
        sub = usable[usable["family"] == family]
        if sub.empty:
            omitted.append(family)
            ax.scatter([], [], label=f"{family} (no resolved totals)")
        else:
            ax.scatter(sub["Delta"], sub[resource_col], label=family, alpha=0.8)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(resource_col.replace("_", " "))
    title = f"{resource_col.replace('_', ' ')} for resolved empirical thresholds"
    if omitted:
        title += "\nFamilies without resolved totals are shown in legend only"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_family_boxplot(df: pd.DataFrame, value_col: str = "r_star", ax=None):
    plot_df = _coerce_plot_columns(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if plot_df.empty or "family" not in plot_df.columns:
        ax.set_title(f"No data available for {value_col}")
        return fig, ax
    family_order = list(dict.fromkeys(plot_df["family"].tolist()))
    groups = []
    labels = []
    for family in family_order:
        vals = pd.to_numeric(plot_df.loc[plot_df["family"] == family, value_col], errors="coerce").dropna().values
        if len(vals) == 0:
            groups.append(np.array([np.nan]))
            labels.append(f"{family}\n(n=0)")
        else:
            groups.append(vals)
            labels.append(f"{family}\n(n={len(vals)})")
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel(value_col.replace("_", " "))
    ax.set_title(f"Distribution of resolved {value_col.replace('_', ' ')} by family")
    ax.grid(True, axis="y")
    return fig, ax


def plot_random_instance_variation(random_instance_df: pd.DataFrame, axes=None):
    plot_df = random_instance_df.copy()
    for col in ["resolution_rate", "mean_r_star_resolved", "num_resolved", "num_states"]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        if not isinstance(axes, (list, tuple, np.ndarray)) or len(axes) != 2:
            raise ValueError("plot_random_instance_variation requires two axes when axes is provided")
        fig = axes[0].figure
    ax0, ax1 = axes
    ax0.bar(plot_df["instance_id"], plot_df["resolution_rate"])
    ax0.set_title("Random-instance resolution rate")
    ax0.set_ylabel("Resolution rate")
    ax0.tick_params(axis="x", rotation=45)
    ax0.grid(True, axis="y")

    resolved_only = plot_df[plot_df["num_resolved"] > 0].copy() if "num_resolved" in plot_df.columns else plot_df.iloc[0:0].copy()
    if not resolved_only.empty and "mean_r_star_resolved" in resolved_only.columns:
        ax1.bar(resolved_only["instance_id"], resolved_only["mean_r_star_resolved"])
    ax1.set_title("Mean resolved empirical r*")
    ax1.set_ylabel("Mean resolved r*")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y")
    return fig, axes


def plot_random_instance_resolution_rate(random_instance_df: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    plot_df = random_instance_df.copy()
    plot_df["resolution_rate"] = pd.to_numeric(plot_df["resolution_rate"], errors="coerce")
    ax.bar(plot_df["instance_id"], plot_df["resolution_rate"])
    ax.set_title("Random-instance resolution rate")
    ax.set_ylabel("Resolution rate")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y")
    return fig, ax


def plot_rule_comparison(df: pd.DataFrame, family: str, ax=None):
    plot_df = _coerce_plot_columns(df)
    sub = plot_df[plot_df["family"] == family].copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
    else:
        fig = ax.figure
    if sub.empty:
        ax.set_title(f"No rows for family: {family}")
        return fig, ax

    resolved = sub[sub["resolved"] == True]
    censored = sub[sub["resolved"] == False]
    if not resolved.empty:
        ax.scatter(resolved["Delta"], resolved["r_star"], label="empirical resolved", alpha=0.9)
    if not censored.empty:
        ax.scatter(censored["Delta"], censored["r_value_for_plot"], label="empirical censored", marker="x", alpha=0.8)
    if "worst_case_rule_r" in sub.columns and sub["worst_case_rule_r"].notna().any():
        line = sub[["Delta", "worst_case_rule_r"]].dropna().sort_values("Delta")
        ax.plot(line["Delta"], line["worst_case_rule_r"], label="worst-case proxy rule", linewidth=2)
    if "low_energy_rule_r" in sub.columns and sub["low_energy_rule_r"].notna().any():
        line = sub[["Delta", "low_energy_rule_r"]].dropna().sort_values("Delta")
        ax.plot(line["Delta"], line["low_energy_rule_r"], label="low-energy rule", linewidth=2, linestyle="--")
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"Rule-recommended or empirical $r$")
    ax.set_title(f"Rule comparison: {family}")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_largeN_scaling(df: pd.DataFrame, y_col: str = "t_count_estimate_per_step", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    for family, sub in df.groupby("family"):
        ax.plot(sub["N"], sub[y_col], marker="o", label=family)
    ax.set_xlabel("N")
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_title(f"Reference scaling of {y_col.replace('_', ' ')}")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def make_dashboard(
    exact_df: pd.DataFrame,
    random_instance_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    plot_rstar_vs_delta(exact_df, ax=axes[0])
    plot_resource_vs_delta(exact_df, resource_col="t_count_estimate_total", ax=axes[1])
    plot_family_boxplot(exact_df, value_col="r_star", ax=axes[2])
    plot_family_boxplot(exact_df, value_col="t_count_estimate_total", ax=axes[3])
    plot_random_instance_resolution_rate(random_instance_df, ax=axes[4])
    plot_largeN_scaling(scaling_df, y_col="t_count_estimate_per_step", ax=axes[5])
    fig.tight_layout()
    return fig, axes


def plot_rstar_vs_energy_variance(df: pd.DataFrame, ax=None):
    """Scatter r* vs energy variance σ_H², highlighting endpoint thetas (θ ≈ 0 or π)."""
    plot_df = _coerce_plot_columns(df)
    if "energy_variance" not in plot_df.columns:
        raise ValueError("plot_rstar_vs_energy_variance requires 'energy_variance' column")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    palette = [p["color"] for p in prop_cycle]
    for color_idx, (family, sub) in enumerate(plot_df.groupby("family")):
        color = palette[color_idx % len(palette)]
        resolved = sub[
            (sub["resolved"] == True)
            & sub["r_star"].notna()
            & sub["energy_variance"].notna()
        ].copy()
        if resolved.empty:
            continue
        resolved["energy_variance"] = pd.to_numeric(resolved["energy_variance"], errors="coerce")
        is_endpoint = (resolved["theta"].abs() < 0.05) | (
            (resolved["theta"] - np.pi).abs() < 0.05
        )
        interior = resolved[~is_endpoint]
        endpoint = resolved[is_endpoint]
        if not interior.empty:
            ax.scatter(interior["energy_variance"], interior["r_star"],
                       color=color, label=family, alpha=0.8, s=40)
        if not endpoint.empty:
            ax.scatter(endpoint["energy_variance"], endpoint["r_star"],
                       color=color, marker="*", s=160, zorder=5,
                       label=f"{family} (θ endpoint)")
    ax.set_xlabel(r"Energy variance $\sigma_H^2 = \langle H^2\rangle - \langle H\rangle^2$")
    ax.set_ylabel(r"Empirical threshold $r^*$")
    ax.set_title(r"$r^*$ vs energy variance  (★ = endpoint $\theta$)")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_validation_eps_sweep(validation_df: pd.DataFrame, ax=None):
    """Plot mean r* vs ε for each family; checks that family ordering is preserved."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if validation_df.empty or "eps" not in validation_df.columns:
        ax.set_title("No validation data available")
        return fig, ax
    for family, sub in validation_df.groupby("family"):
        sub_sorted = sub.sort_values("eps")
        mask = sub_sorted["mean_r_star_resolved"].notna()
        if mask.any():
            ax.plot(
                sub_sorted.loc[mask, "eps"],
                sub_sorted.loc[mask, "mean_r_star_resolved"],
                marker="o", label=f"{family} (resolved mean)",
            )
        mask_lb = sub_sorted["mean_r_star_lower_bound"].notna()
        if mask_lb.any():
            ax.plot(
                sub_sorted.loc[mask_lb, "eps"],
                sub_sorted.loc[mask_lb, "mean_r_star_lower_bound"],
                marker="s", linestyle="--", alpha=0.6,
                label=f"{family} (lower-bound mean)",
            )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Error threshold $\varepsilon$")
    ax.set_ylabel(r"Mean $r^*$")
    ax.set_title(r"Validation: mean $r^*$ vs $\varepsilon$ — is family ordering preserved?")
    ax.grid(True)
    ax.legend(fontsize=8)
    return fig, ax


def plot_ordering_comparison(ordering_df: pd.DataFrame, ax=None):
    """Box plot of r* by term ordering, showing sensitivity to ordering choice."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if ordering_df.empty or "ordering" not in ordering_df.columns:
        ax.set_title("No ordering comparison data available")
        return fig, ax
    ordering_names = list(dict.fromkeys(ordering_df["ordering"].tolist()))
    groups = []
    labels = []
    for ord_name in ordering_names:
        sub = ordering_df[ordering_df["ordering"] == ord_name]
        vals = pd.to_numeric(sub["r_star"], errors="coerce").dropna().values
        groups.append(vals if len(vals) > 0 else np.array([np.nan]))
        n = int((sub["resolved"] == True).sum()) if "resolved" in sub.columns else len(vals)
        labels.append(f"{ord_name}\n(n_resolved={n})")
    ax.boxplot(groups, tick_labels=labels)
    ax.set_ylabel(r"Empirical $r^*$")
    ax.set_title("r* stability under different term orderings (structured dipolar)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y")
    return fig, ax


def save_figure(fig, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
