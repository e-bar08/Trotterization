"""Study runners, summaries, and export helpers for the Physics 765 project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from .hamiltonians import (
    Term,
    build_H,
    coeff_l2_from_terms,
    dipolar_terms_xxyy_m2zz,
    family_summary,
    matched_random_two_local_pauli_terms,
    random_two_local_generation_info,
    shifted_tfim_psd_terms,
)
from .resource_estimation import estimate_total_resources_for_r
from .selection_rules import (
    empirical_rule_summary,
    low_energy_rule_placeholder,
    worst_case_commutator_proxy_rule,
)
from .states import rotated_zero_product_state, theta_family
from .trotter_utils import energy_expectation, energy_variance, exact_propagator, find_r_star_linear, sweep_r_values


LowEnergyRuleCallable = Callable[[int, float, float, float], int | float | None]


def censored_r_value_for_plot(r_star: int | None, r_max: int) -> int:
    return int(r_star) if r_star is not None else int(r_max + 1)


def run_single_state_experiment(
    terms: Sequence[Term],
    H,
    psi0,
    t: float,
    eps: float,
    r_max: int = 120,
    order: str = "forward",
    method: str = "pauli",
    U_exact=None,
) -> Dict[str, object]:
    Delta = energy_expectation(H, psi0)
    sigma2 = energy_variance(H, psi0)
    empirical = find_r_star_linear(
        terms,
        H,
        psi0,
        t,
        eps,
        r_max=r_max,
        order=order,
        method=method,
        U_exact=U_exact,
    )
    row: Dict[str, object] = {
        "Delta": float(Delta),
        "energy_variance": float(sigma2),
        "resolved": bool(empirical["resolved"]),
        "r_star": empirical["r_star"],
        "error_at_r_star": empirical["error_at_r_star"],
        "fidelity_at_r_star": empirical["fidelity_at_r_star"],
        "error_at_r_max": float(empirical["error_at_r_max"]),
        "fidelity_at_r_max": float(empirical["fidelity_at_r_max"]),
        "best_r_in_scan": empirical["best_r_in_scan"],
        "best_error_in_scan": float(empirical["best_error_in_scan"]),
    }
    row.update(empirical_rule_summary(row))
    return row


def run_theta_sweep_for_family(
    family_name: str,
    terms: Sequence[Term],
    H,
    N: int,
    t: float,
    eps: float,
    r_max: int = 120,
    num_thetas: int = 21,
    order: str = "forward",
    method: str = "pauli",
    instance_id: str = "instance_0",
    low_energy_rule_callable: LowEnergyRuleCallable | None = None,
    family_kind: str = "structured",
) -> List[Dict[str, object]]:
    static_summary = family_summary(terms)
    U_exact = exact_propagator(H, t)
    worst_case_rule = worst_case_commutator_proxy_rule(terms, t=t, eps=eps)
    rows: List[Dict[str, object]] = []
    for theta, psi0 in theta_family(N, num=num_thetas):
        row = run_single_state_experiment(
            terms,
            H,
            psi0,
            t,
            eps,
            r_max=r_max,
            order=order,
            method=method,
            U_exact=U_exact,
        )
        row.update({
            "family": family_name,
            "family_kind": family_kind,
            "instance_id": instance_id,
            "N": int(N),
            "t": float(t),
            "eps": float(eps),
            "theta": float(theta),
            "r_max": int(r_max),
            "r_value_for_plot": censored_r_value_for_plot(row["r_star"], r_max),
            "censored_for_plot": not bool(row["resolved"]),
        })
        row.update(static_summary)
        row.update(worst_case_rule)
        row.update(low_energy_rule_placeholder(N=N, Delta=float(row["Delta"]), t=t, eps=eps, rule_callable=low_energy_rule_callable))
        rows.append(row)
    return rows


def annotate_resource_estimates(
    rows: Sequence[Dict[str, object]],
    terms: Sequence[Term],
    rotation_precision: float = 1e-10,
) -> List[Dict[str, object]]:
    new_rows: List[Dict[str, object]] = []
    if not rows:
        return new_rows
    t_value = float(rows[0]["t"])
    for row in rows:
        row_copy = dict(row)
        if bool(row_copy.get("resolved", False)) and row_copy.get("r_star") is not None:
            resource_r = int(row_copy["r_star"])
            resource_summary = estimate_total_resources_for_r(
                terms,
                t=t_value,
                r=resource_r,
                rotation_precision=rotation_precision,
            )
            row_copy.update(resource_summary)
            row_copy["resource_totals_reported"] = True
            row_copy["resource_totals_status"] = "reported_at_empirical_r_star"
        else:
            row_copy["resource_r"] = None
            row_copy["resource_t"] = float(t_value)
            row_copy["resource_totals_reported"] = False
            row_copy["resource_totals_status"] = "not_reported_unresolved_threshold"
        new_rows.append(row_copy)
    return new_rows


def build_error_curve_rows(
    terms: Sequence[Term],
    H,
    theta_values: Sequence[float],
    N: int,
    t: float,
    r_values: Sequence[int],
    family_label: str,
    order: str = "forward",
    method: str = "pauli",
) -> List[Dict[str, object]]:
    U_exact = exact_propagator(H, t)
    curves: List[Dict[str, object]] = []
    for theta in theta_values:
        psi0 = rotated_zero_product_state(N, float(theta))
        Delta = energy_expectation(H, psi0)
        sweep = sweep_r_values(
            terms,
            H,
            psi0,
            t,
            r_values,
            order=order,
            method=method,
            U_exact=U_exact,
        )
        curves.append({
            "family": family_label,
            "theta": float(theta),
            "Delta": float(Delta),
            "r_values": [int(row["r"]) for row in sweep],
            "errors": [float(row["error"]) for row in sweep],
            "label": fr"{family_label}, $\theta={theta:.2f}$, $\Delta={Delta:+.2f}$",
        })
    return curves


def sweep_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        raise ValueError("sweep_summary requires at least one row")
    first = rows[0]
    deltas = np.array([float(row["Delta"]) for row in rows], dtype=float)
    resolved_rows = [row for row in rows if bool(row.get("resolved", False))]
    summary: Dict[str, object] = {
        "family": str(first["family"]),
        "family_kind": str(first.get("family_kind", "unknown")),
        "instance_id": str(first["instance_id"]),
        "num_states": int(len(rows)),
        "num_resolved": int(len(resolved_rows)),
        "num_unresolved": int(len(rows) - len(resolved_rows)),
        "resolution_rate": float(len(resolved_rows) / len(rows)),
        "min_Delta": float(np.min(deltas)),
        "max_Delta": float(np.max(deltas)),
        "n_terms": int(first["num_terms"]),
        "num_identity_terms": int(first.get("num_identity_terms", 0)),
        "num_nonidentity_terms": int(first.get("num_nonidentity_terms", 0)),
        "num_single_site_terms": int(first.get("num_single_site_terms", 0)),
        "num_two_site_terms": int(first.get("num_two_site_terms", 0)),
        "distinct_qubit_pairs": int(first["distinct_qubit_pairs"]),
        "noncommuting_pairs": int(first["noncommuting_pairs"]),
        "coeff_rms": float(first["coeff_rms"]),
        "coeff_l1": float(first["coeff_l1"]),
        "coeff_l2": float(first["coeff_l2"]),
        "worst_case_rule_r": int(first["worst_case_rule_r"]),
        "worst_case_rule_status": str(first["worst_case_rule_status"]),
        "low_energy_rule_status": str(first["low_energy_rule_status"]),
    }
    r_max_val = int(first.get("r_max", 120))
    if resolved_rows:
        rstars = np.array([int(row["r_star"]) for row in resolved_rows], dtype=float)
        # Lower-bound mean: unresolved states contribute r_max (conservative floor).
        all_r_values = np.concatenate([rstars, np.full(len(rows) - len(resolved_rows), r_max_val)])
        summary.update({
            "mean_r_star_resolved": float(np.mean(rstars)),
            "median_r_star_resolved": float(np.median(rstars)),
            "min_r_star_resolved": float(np.min(rstars)),
            "max_r_star_resolved": float(np.max(rstars)),
            "mean_r_star_lower_bound": float(np.mean(all_r_values)),
        })
    else:
        summary.update({
            "mean_r_star_resolved": np.nan,
            "median_r_star_resolved": np.nan,
            "min_r_star_resolved": np.nan,
            "max_r_star_resolved": np.nan,
            "mean_r_star_lower_bound": float(r_max_val),
        })

    numeric_total_cols = [
        "t_count_estimate_total",
        "cnot_count_total",
        "single_qubit_clifford_count_total",
        "rotation_gate_count_total",
        "sequential_layer_proxy_total",
    ]
    resolved_resource_rows = [row for row in resolved_rows if bool(row.get("resource_totals_reported", False))]
    for col in numeric_total_cols:
        if resolved_resource_rows and col in resolved_resource_rows[0]:
            vals = np.array([float(row[col]) for row in resolved_resource_rows], dtype=float)
            summary[f"mean_{col}"] = float(np.mean(vals))
            summary[f"median_{col}"] = float(np.median(vals))
        else:
            summary[f"mean_{col}"] = np.nan
            summary[f"median_{col}"] = np.nan
    return summary


def censored_summary_stats(rows: Sequence[Dict[str, object]], r_max: int) -> Dict[str, object]:
    """Return resolved mean r* and a lower-bound mean treating unresolved states as r_max.

    Args:
        rows: experiment rows, each with 'resolved' and 'r_star' keys.
        r_max: scan ceiling used as the lower-bound substitute for unresolved states.

    Returns dict with keys: num_states, num_resolved, num_unresolved,
        mean_r_star_resolved, mean_r_star_lower_bound.
    """
    resolved = [row for row in rows if bool(row.get("resolved", False))]
    unresolved = [row for row in rows if not bool(row.get("resolved", False))]
    r_stars = [int(row["r_star"]) for row in resolved]
    all_vals = r_stars + [r_max] * len(unresolved)
    return {
        "num_states": int(len(rows)),
        "num_resolved": int(len(resolved)),
        "num_unresolved": int(len(unresolved)),
        "mean_r_star_resolved": float(np.mean(r_stars)) if r_stars else float("nan"),
        "mean_r_star_lower_bound": float(np.mean(all_vals)) if all_vals else float("nan"),
    }


def rows_to_dataframe(rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    sort_cols = [col for col in ["family", "instance_id", "theta", "Delta"] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "resolved" in df.columns:
        df["resolved"] = df["resolved"].astype("boolean")
    for col in ["r_star", "r_value_for_plot", "Delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def export_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def build_family_level_regression_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for family, sub in df.groupby("family"):
        resolved_sub = sub[sub["resolved"] == True].copy()
        if resolved_sub.empty:
            rows.append({
                "family": family,
                "num_rows": int(len(sub)),
                "num_resolved": 0,
                "resolution_rate": float(sub["resolved"].fillna(False).mean()),
                "corr_rstar_Delta_resolved": np.nan,
                "corr_rstar_theta_resolved": np.nan,
                "mean_r_star_resolved": np.nan,
                "mean_t_count_estimate_total_resolved": np.nan,
            })
            continue
        corr_delta = float(resolved_sub["Delta"].corr(resolved_sub["r_star"])) if (resolved_sub["Delta"].nunique() > 1 and resolved_sub["r_star"].nunique() > 1) else np.nan
        corr_theta = float(resolved_sub["theta"].corr(resolved_sub["r_star"])) if (resolved_sub["theta"].nunique() > 1 and resolved_sub["r_star"].nunique() > 1) else np.nan
        t_mean = float(resolved_sub["t_count_estimate_total"].mean()) if "t_count_estimate_total" in resolved_sub.columns else np.nan
        rows.append({
            "family": family,
            "num_rows": int(len(sub)),
            "num_resolved": int(len(resolved_sub)),
            "resolution_rate": float(len(resolved_sub) / len(sub)),
            "corr_rstar_Delta_resolved": corr_delta,
            "corr_rstar_theta_resolved": corr_theta,
            "mean_r_star_resolved": float(resolved_sub["r_star"].mean()),
            "mean_t_count_estimate_total_resolved": t_mean,
        })
    return pd.DataFrame(rows)


def aggregate_instance_summaries_by_family(instance_summary_df: pd.DataFrame) -> pd.DataFrame:
    if instance_summary_df.empty:
        return pd.DataFrame()
    grouped = (
        instance_summary_df.groupby("family", dropna=False)
        .agg(
            num_instances=("instance_id", "nunique"),
            total_states=("num_states", "sum"),
            total_resolved=("num_resolved", "sum"),
            total_unresolved=("num_unresolved", "sum"),
            mean_resolution_rate=("resolution_rate", "mean"),
            mean_of_instance_mean_r_star=("mean_r_star_resolved", "mean"),
            median_of_instance_mean_r_star=("mean_r_star_resolved", "median"),
            mean_num_nonidentity_terms=("num_nonidentity_terms", "mean"),
            mean_num_two_site_terms=("num_two_site_terms", "mean"),
        )
        .reset_index()
    )
    grouped["overall_resolution_rate"] = grouped["total_resolved"] / grouped["total_states"]
    return grouped


def build_rule_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "family",
        "instance_id",
        "theta",
        "Delta",
        "resolved",
        "r_star",
        "r_value_for_plot",
        "worst_case_rule_r",
        "worst_case_rule_status",
        "low_energy_rule_r",
        "low_energy_rule_status",
    ]
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    if "resolved" in out.columns:
        out["empirical_rule_display_r"] = np.where(out["resolved"] == True, out["r_star"], out.get("r_value_for_plot"))
    return out


def run_random_instance_study(
    N: int,
    structured_terms: Sequence[Term],
    t: float,
    eps: float,
    num_instances: int,
    num_thetas: int,
    r_max: int,
    rotation_precision: float,
    rng: np.random.Generator | None = None,
    order: str = "forward",
    method: str = "pauli",
    start_index: int = 1,
) -> tuple[list[dict], list[dict]]:
    if rng is None:
        rng = np.random.default_rng()
    all_rows: List[dict] = []
    summaries: List[dict] = []
    for instance_idx in range(start_index, start_index + num_instances):
        terms = matched_random_two_local_pauli_terms(N, structured_terms, rng=rng, unique=True)
        H = build_H(terms)
        rows = run_theta_sweep_for_family(
            family_name="random two-local",
            terms=terms,
            H=H,
            N=N,
            t=t,
            eps=eps,
            r_max=r_max,
            num_thetas=num_thetas,
            order=order,
            method=method,
            instance_id=f"random_{instance_idx}",
            family_kind="random_baseline",
        )
        rows = annotate_resource_estimates(rows, terms, rotation_precision=rotation_precision)
        all_rows.extend(rows)
        summaries.append(sweep_summary(rows))
    return all_rows, summaries


def run_validation_eps_sweep(
    families: Sequence[tuple],
    N: int,
    t: float,
    eps_values: Sequence[float],
    r_max: int,
    num_thetas: int,
    order: str = "forward",
    method: str = "pauli",
) -> pd.DataFrame:
    """Sweep epsilon over eps_values for each family and report r* statistics.

    Args:
        families: list of (family_name, terms, H) triples.
        eps_values: thresholds to sweep, e.g. [1e-2, 1e-3, 1e-4].

    Returns DataFrame with columns: eps, family, num_states, num_resolved,
        num_unresolved, mean_r_star_resolved, mean_r_star_lower_bound.
    """
    rows: List[Dict[str, object]] = []
    for eps in eps_values:
        for family_name, terms, H in families:
            sweep_rows = run_theta_sweep_for_family(
                family_name=family_name,
                terms=terms,
                H=H,
                N=N,
                t=t,
                eps=float(eps),
                r_max=r_max,
                num_thetas=num_thetas,
                order=order,
                method=method,
                instance_id="validation",
                family_kind="validation",
            )
            stats = censored_summary_stats(sweep_rows, r_max)
            rows.append({
                "eps": float(eps),
                "family": family_name,
                **stats,
            })
    return pd.DataFrame(rows)


def run_ordering_comparison(
    terms: Sequence[Term],
    H,
    N: int,
    t: float,
    eps: float,
    r_max: int,
    num_thetas: int = 11,
    num_random_orderings: int = 2,
    seed: int | None = None,
    method: str = "pauli",
) -> pd.DataFrame:
    """Compare r* under forward, reverse, and random term orderings.

    Random orderings are fixed permutations (not re-shuffled per Trotter step),
    so each trial is fully reproducible given the seed.

    Returns DataFrame with columns: ordering, theta, r_star, resolved, r_value_for_plot.
    """
    rng = np.random.default_rng(seed)
    U_exact = exact_propagator(H, t)

    term_list = list(terms)
    orderings: List[tuple] = [
        ("forward", term_list),
        ("reverse", list(reversed(term_list))),
    ]
    for trial in range(num_random_orderings):
        shuffled = list(term_list)
        rng.shuffle(shuffled)
        orderings.append((f"random_{trial + 1}", shuffled))

    rows: List[Dict[str, object]] = []
    for theta, psi0 in theta_family(N, num=num_thetas):
        for ordering_name, ordered_terms in orderings:
            result = find_r_star_linear(
                ordered_terms, H, psi0, t, eps,
                r_max=r_max, order="forward", method=method, U_exact=U_exact,
            )
            rows.append({
                "ordering": ordering_name,
                "theta": float(theta),
                "r_star": result["r_star"],
                "resolved": bool(result["resolved"]),
                "r_value_for_plot": censored_r_value_for_plot(result["r_star"], r_max),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level reproducibility entry point
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, object] = {
    "N": 4,
    "t": 1.0,
    "eps": 1e-3,
    "r_max": 60,
    "num_thetas": 11,
    "num_instances": 3,
    "seed": 0,
    "J0": 1.0,
    "alpha": 3.0,
    "local_J": 1.0,
    "local_h": 0.7,
    "rotation_precision": 1e-10,
    "method": "pauli",
    "order": "forward",
    "validation_eps_values": [1e-2, 1e-3, 1e-4],
    "num_ordering_trials": 2,
}


def run_full_study(config: Dict[str, object] | None = None) -> Dict[str, object]:
    """Run all study experiments and return a results dictionary.

    Missing keys in *config* are filled from DEFAULT_CONFIG.

    Returns a dict with keys:
        main_df           – exact-family theta-sweep rows (structured, random, local)
        all_rows_df       – main_df plus all random-instance rows
        instance_summary_df – per-instance sweep summaries
        family_summary_df   – per-family aggregated summaries
        validation_df       – eps-sweep validation results
        ordering_df         – term-ordering comparison results
        generation_info     – dict documenting random baseline generation
        config              – the resolved config dict used for the run
    """
    cfg: Dict[str, object] = {**DEFAULT_CONFIG, **(config or {})}
    N = int(cfg["N"])
    t = float(cfg["t"])
    eps = float(cfg["eps"])
    r_max = int(cfg["r_max"])
    num_thetas = int(cfg["num_thetas"])
    num_instances = int(cfg["num_instances"])
    seed = cfg["seed"]
    method = str(cfg["method"])
    order = str(cfg["order"])
    rotation_precision = float(cfg["rotation_precision"])

    rng = np.random.default_rng(seed)

    # Build Hamiltonians
    structured_terms = dipolar_terms_xxyy_m2zz(N, J0=float(cfg["J0"]), alpha=float(cfg["alpha"]))
    structured_H = build_H(structured_terms)

    target_l2 = coeff_l2_from_terms(structured_terms)
    random_terms = matched_random_two_local_pauli_terms(N, structured_terms, rng=rng, unique=True)
    random_H = build_H(random_terms)

    local_terms = shifted_tfim_psd_terms(N, J=float(cfg["local_J"]), h=float(cfg["local_h"]), periodic=False)
    local_H = build_H(local_terms)

    generation_info = random_two_local_generation_info(
        N=N, M=len(structured_terms), target_l2=target_l2, seed=seed, unique=True
    )

    # Theta sweeps for the three families
    structured_rows = run_theta_sweep_for_family(
        "structured dipolar", structured_terms, structured_H, N, t, eps, r_max, num_thetas,
        order=order, method=method, instance_id="structured_0", family_kind="structured_long_range",
    )
    structured_rows = annotate_resource_estimates(structured_rows, structured_terms, rotation_precision=rotation_precision)

    random_rows = run_theta_sweep_for_family(
        "random two-local", random_terms, random_H, N, t, eps, r_max, num_thetas,
        order=order, method=method, instance_id="random_0", family_kind="random_baseline",
    )
    random_rows = annotate_resource_estimates(random_rows, random_terms, rotation_precision=rotation_precision)

    local_rows = run_theta_sweep_for_family(
        "local shifted TFIM", local_terms, local_H, N, t, eps, r_max, num_thetas,
        order=order, method=method, instance_id="local_0", family_kind="structured_local_psd",
    )
    local_rows = annotate_resource_estimates(local_rows, local_terms, rotation_precision=rotation_precision)

    # Additional random instances
    extra_rows, random_summaries = run_random_instance_study(
        N=N, structured_terms=structured_terms, t=t, eps=eps,
        num_instances=num_instances, num_thetas=num_thetas, r_max=r_max,
        rotation_precision=rotation_precision, rng=rng, order=order, method=method,
        start_index=1,
    )

    main_df = rows_to_dataframe(structured_rows + random_rows + local_rows)
    all_rows_df = rows_to_dataframe(structured_rows + random_rows + local_rows + extra_rows)

    instance_summary_df = pd.DataFrame([
        sweep_summary(structured_rows),
        sweep_summary(random_rows),
        sweep_summary(local_rows),
    ] + random_summaries)
    family_summary_df = aggregate_instance_summaries_by_family(instance_summary_df)

    # Validation: eps sweep
    families_for_validation = [
        ("structured dipolar", structured_terms, structured_H),
        ("random two-local", random_terms, random_H),
        ("local shifted TFIM", local_terms, local_H),
    ]
    validation_df = run_validation_eps_sweep(
        families=families_for_validation,
        N=N, t=t,
        eps_values=list(cfg["validation_eps_values"]),
        r_max=r_max, num_thetas=num_thetas,
        order=order, method=method,
    )

    # Ordering comparison on the structured family
    ordering_df = run_ordering_comparison(
        terms=structured_terms, H=structured_H,
        N=N, t=t, eps=eps, r_max=r_max, num_thetas=num_thetas,
        num_random_orderings=int(cfg["num_ordering_trials"]),
        seed=seed, method=method,
    )

    return {
        "main_df": main_df,
        "all_rows_df": all_rows_df,
        "instance_summary_df": instance_summary_df,
        "family_summary_df": family_summary_df,
        "validation_df": validation_df,
        "ordering_df": ordering_df,
        "generation_info": generation_info,
        "config": dict(cfg),
    }


def export_clean_outputs(
    results: Dict[str, object],
    out_dir,
) -> Dict[str, object]:
    """Export main dataset, summaries, and validation results to CSV files.

    Args:
        results: dict returned by run_full_study.
        out_dir: directory to write CSV files (created if absent).

    Returns dict mapping logical key to the written Path.
    """
    from pathlib import Path as _Path
    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, object] = {}
    for key, filename in [
        ("main_df", "main_dataset.csv"),
        ("all_rows_df", "all_rows.csv"),
        ("instance_summary_df", "instance_summaries.csv"),
        ("family_summary_df", "family_summaries.csv"),
        ("validation_df", "validation_eps_sweep.csv"),
        ("ordering_df", "ordering_comparison.csv"),
    ]:
        df = results.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            written[key] = export_dataframe(df, out_dir / filename)
    return written
