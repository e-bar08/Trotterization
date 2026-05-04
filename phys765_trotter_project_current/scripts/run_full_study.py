from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_utils import (
    aggregate_instance_summaries_by_family,
    annotate_resource_estimates,
    build_error_curve_rows,
    build_family_level_regression_summary,
    build_rule_comparison_table,
    export_dataframe,
    rows_to_dataframe,
    run_random_instance_study,
    run_theta_sweep_for_family,
    sweep_summary,
)
from src.hamiltonians import (
    build_H,
    dipolar_terms_xxyy_m2zz,
    matched_random_two_local_pauli_terms,
    shifted_tfim_psd_terms,
)
from src.plotting_utils import (
    make_dashboard,
    plot_error_curves,
    plot_family_boxplot,
    plot_largeN_scaling,
    plot_random_instance_variation,
    plot_resource_vs_delta,
    plot_rstar_vs_delta,
    plot_rule_comparison,
    save_figure,
)
from src.resource_estimation import estimate_step_resources, export_stim_skeleton
from src.states import basis_state, plus_state
from src.trotter_utils import compare_trotter_implementations, sweep_r_values


def profile_params(profile: str) -> dict:
    if profile == "quick":
        return {
            "N_exact": 4,
            "t": 1.0,
            "eps": 1e-3,
            "r_max": 30,
            "num_thetas": 7,
            "num_random_instances": 1,
            "r_curve_max": 20,
            "rotation_precision": 1e-10,
            "scaling_N_values": [4, 5, 6],
            "reference_r_for_scaling": 24,
            "J0": 1.0,
            "alpha": 3.0,
            "local_J": 1.0,
            "local_h": 0.7,
            "seed": 0,
        }
    if profile == "paper":
        return {
            "N_exact": 6,
            "t": 1.0,
            "eps": 1e-3,
            "r_max": 120,
            "num_thetas": 21,
            "num_random_instances": 6,
            "r_curve_max": 80,
            "rotation_precision": 1e-10,
            "scaling_N_values": [4, 5, 6, 7, 8, 9, 10, 11, 12],
            "reference_r_for_scaling": 48,
            "J0": 1.0,
            "alpha": 3.0,
            "local_J": 1.0,
            "local_h": 0.7,
            "seed": 0,
        }
    raise ValueError("profile must be 'quick' or 'paper'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["quick", "paper"], default="quick")
    args = parser.parse_args()
    cfg = profile_params(args.profile)

    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    table_dir = results_dir / "tables"
    circuit_dir = results_dir / "circuits"
    for path in [fig_dir, table_dir, circuit_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    N = cfg["N_exact"]
    t = cfg["t"]
    eps = cfg["eps"]
    r_max = cfg["r_max"]
    num_thetas = cfg["num_thetas"]
    order = "forward"
    method = "pauli"

    structured_terms = dipolar_terms_xxyy_m2zz(N, J0=cfg["J0"], alpha=cfg["alpha"])
    structured_H = build_H(structured_terms)
    random_terms = matched_random_two_local_pauli_terms(N, structured_terms, rng=rng, unique=True)
    random_H = build_H(random_terms)
    local_terms = shifted_tfim_psd_terms(N, J=cfg["local_J"], h=cfg["local_h"], periodic=False)
    local_H = build_H(local_terms)

    validation_rows = []
    validation_rows.append(compare_trotter_implementations(structured_terms, plus_state(N), t=0.7, r=8, order=order))
    zero_state_curve = sweep_r_values(
        structured_terms,
        structured_H,
        basis_state(N, "0" * N),
        t=t,
        r_values=list(range(1, min(12, r_max) + 1)),
        order=order,
        method=method,
    )
    validation_rows.append({
        "check": "|0...0> eigenstate-like exactness for dipolar family",
        "max_error_over_first_rs": float(max(row["error"] for row in zero_state_curve)),
    })
    export_dataframe(pd.DataFrame(validation_rows), table_dir / "validation_summary.csv")

    structured_rows = run_theta_sweep_for_family(
        family_name="structured dipolar",
        terms=structured_terms,
        H=structured_H,
        N=N,
        t=t,
        eps=eps,
        r_max=r_max,
        num_thetas=num_thetas,
        order=order,
        method=method,
        instance_id="structured_0",
        family_kind="structured_long_range",
    )
    structured_rows = annotate_resource_estimates(structured_rows, structured_terms, rotation_precision=cfg["rotation_precision"])

    random_rows = run_theta_sweep_for_family(
        family_name="random two-local",
        terms=random_terms,
        H=random_H,
        N=N,
        t=t,
        eps=eps,
        r_max=r_max,
        num_thetas=num_thetas,
        order=order,
        method=method,
        instance_id="random_0",
        family_kind="random_baseline",
    )
    random_rows = annotate_resource_estimates(random_rows, random_terms, rotation_precision=cfg["rotation_precision"])

    local_rows = run_theta_sweep_for_family(
        family_name="local shifted TFIM",
        terms=local_terms,
        H=local_H,
        N=N,
        t=t,
        eps=eps,
        r_max=r_max,
        num_thetas=num_thetas,
        order=order,
        method=method,
        instance_id="local_0",
        family_kind="structured_local_psd",
    )
    local_rows = annotate_resource_estimates(local_rows, local_terms, rotation_precision=cfg["rotation_precision"])

    extra_random_rows, random_instance_summaries = run_random_instance_study(
        N=N,
        structured_terms=structured_terms,
        t=t,
        eps=eps,
        num_instances=cfg["num_random_instances"],
        num_thetas=num_thetas,
        r_max=r_max,
        rotation_precision=cfg["rotation_precision"],
        rng=rng,
        order=order,
        method=method,
        start_index=1,
    )

    exact_df = rows_to_dataframe(structured_rows + random_rows + local_rows)
    all_random_df = rows_to_dataframe(extra_random_rows)
    all_state_df = rows_to_dataframe(structured_rows + random_rows + local_rows + extra_random_rows)

    instance_summary_df = pd.DataFrame([
        sweep_summary(structured_rows),
        sweep_summary(random_rows),
        sweep_summary(local_rows),
    ] + random_instance_summaries)
    family_summary_df = aggregate_instance_summaries_by_family(instance_summary_df)
    random_instance_df = pd.DataFrame(random_instance_summaries)
    correlation_df = build_family_level_regression_summary(all_state_df)
    rule_comparison_df = build_rule_comparison_table(exact_df)

    export_dataframe(exact_df, table_dir / "exact_family_state_results.csv")
    export_dataframe(all_random_df, table_dir / "random_instance_state_results.csv")
    export_dataframe(all_state_df, table_dir / "all_state_results.csv")
    export_dataframe(instance_summary_df, table_dir / "instance_summary.csv")
    export_dataframe(family_summary_df, table_dir / "family_summary.csv")
    export_dataframe(random_instance_df, table_dir / "random_instance_summary.csv")
    export_dataframe(correlation_df, table_dir / "correlation_summary.csv")
    export_dataframe(rule_comparison_df, table_dir / "rule_comparison.csv")

    theta_values = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi]
    r_values = list(range(1, cfg["r_curve_max"] + 1))
    structured_curves = build_error_curve_rows(structured_terms, structured_H, theta_values, N, t, r_values, "structured dipolar", order=order, method=method)
    random_curves = build_error_curve_rows(random_terms, random_H, theta_values, N, t, r_values, "random two-local", order=order, method=method)
    local_curves = build_error_curve_rows(local_terms, local_H, theta_values, N, t, r_values, "local shifted TFIM", order=order, method=method)

    fig, _ = plot_rstar_vs_delta(exact_df)
    save_figure(fig, fig_dir / "rstar_vs_delta.png")
    plt.close(fig)

    fig, _ = plot_error_curves(structured_curves)
    save_figure(fig, fig_dir / "structured_error_curves.png")
    plt.close(fig)

    fig, _ = plot_error_curves(random_curves)
    save_figure(fig, fig_dir / "random_error_curves.png")
    plt.close(fig)

    fig, _ = plot_error_curves(local_curves)
    save_figure(fig, fig_dir / "local_error_curves.png")
    plt.close(fig)

    fig, _ = plot_resource_vs_delta(exact_df, resource_col="t_count_estimate_total")
    save_figure(fig, fig_dir / "tcount_vs_delta.png")
    plt.close(fig)

    fig, _ = plot_family_boxplot(exact_df, value_col="r_star")
    save_figure(fig, fig_dir / "rstar_boxplot.png")
    plt.close(fig)

    fig, _ = plot_family_boxplot(exact_df, value_col="t_count_estimate_total")
    save_figure(fig, fig_dir / "tcount_boxplot.png")
    plt.close(fig)

    fig, _ = plot_random_instance_variation(random_instance_df)
    save_figure(fig, fig_dir / "random_instance_variation.png")
    plt.close(fig)

    rule_fig, rule_axes = plt.subplots(1, 3, figsize=(18, 4.5))
    plot_rule_comparison(exact_df, family="structured dipolar", ax=rule_axes[0])
    plot_rule_comparison(exact_df, family="random two-local", ax=rule_axes[1])
    plot_rule_comparison(exact_df, family="local shifted TFIM", ax=rule_axes[2])
    rule_fig.tight_layout()
    save_figure(rule_fig, fig_dir / "rule_comparison.png")
    plt.close(rule_fig)

    scaling_rows = []
    for N_scale in cfg["scaling_N_values"]:
        dip_terms = dipolar_terms_xxyy_m2zz(N_scale, J0=cfg["J0"], alpha=cfg["alpha"])
        rand_terms = matched_random_two_local_pauli_terms(N_scale, dip_terms, rng=rng, unique=True)
        local_scale_terms = shifted_tfim_psd_terms(N_scale, J=cfg["local_J"], h=cfg["local_h"], periodic=False)
        for family_name, terms in [
            ("structured dipolar", dip_terms),
            ("random two-local", rand_terms),
            ("local shifted TFIM", local_scale_terms),
        ]:
            row = estimate_step_resources(
                terms,
                dt=cfg["t"] / cfg["reference_r_for_scaling"],
                rotation_precision=cfg["rotation_precision"],
            )
            row.update({
                "family": family_name,
                "N": int(N_scale),
                "reference_r": int(cfg["reference_r_for_scaling"]),
                "t": float(cfg["t"]),
            })
            scaling_rows.append(row)
    scaling_df = pd.DataFrame(scaling_rows)
    export_dataframe(scaling_df, table_dir / "reference_scaling_resources.csv")

    fig, _ = plot_largeN_scaling(scaling_df, y_col="t_count_estimate_per_step")
    save_figure(fig, fig_dir / "reference_scaling_tcount.png")
    plt.close(fig)

    dashboard_fig, _ = make_dashboard(exact_df, random_instance_df, scaling_df)
    save_figure(dashboard_fig, fig_dir / "summary_dashboard.png")
    plt.close(dashboard_fig)

    structured_stim_path = export_stim_skeleton(structured_terms, circuit_dir / "structured_step_skeleton.stim")
    random_stim_path = export_stim_skeleton(random_terms, circuit_dir / "random_step_skeleton.stim")
    local_stim_path = export_stim_skeleton(local_terms, circuit_dir / "local_step_skeleton.stim")

    print({
        "profile": args.profile,
        "results_dir": str(results_dir),
        "num_exact_rows": len(exact_df),
        "num_all_rows": len(all_state_df),
        "num_resolved_exact_rows": int(exact_df["resolved"].fillna(False).sum()),
        "figures": len(list(fig_dir.glob("*.png"))),
        "tables": len(list(table_dir.glob("*.csv"))),
        "structured_stim_exported": structured_stim_path is not None,
        "random_stim_exported": random_stim_path is not None,
        "local_stim_exported": local_stim_path is not None,
    })


if __name__ == "__main__":
    main()
