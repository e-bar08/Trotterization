"""Rule interfaces for the Physics 765 Trotter study.

This module separates three notions that appear in the project proposal:

1. empirical rule:
   the actual threshold scan over r used in the exact-validation regime;
2. worst-case rule:
   a state-independent proxy rule based on a first-order commutator bound;
3. low-energy rule:
   a placeholder interface for a theorem-specific low-energy bound.

Only the empirical rule is treated as a numerically validated quantity in this repository.
The worst-case rule implemented here is a transparent proxy, not a theorem-certified
fidelity guarantee. The low-energy rule requires additional theorem-specific constants
or a callable supplied by the user.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from .hamiltonians import Term, pauli_strings_commute


LowEnergyRuleCallable = Callable[[int, float, float, float], int | float | None]


def absolute_commutator_sum(terms: Sequence[Term]) -> float:
    """Return sum_{a<b} ||[H_a, H_b]|| for Pauli-string terms H_a = c_a P_a.

    For Pauli strings, ||[c_a P_a, c_b P_b]|| is either 0 (if they commute) or
    2 |c_a c_b| (if they anticommute on an odd number of sites).
    """
    total = 0.0
    for idx_a, term_a in enumerate(terms):
        coeff_a = float(term_a["coeff"])
        pauli_a = str(term_a["pauli_string"])
        for term_b in terms[idx_a + 1 :]:
            coeff_b = float(term_b["coeff"])
            pauli_b = str(term_b["pauli_string"])
            if not pauli_strings_commute(pauli_a, pauli_b):
                total += 2.0 * abs(coeff_a * coeff_b)
    return float(total)


def empirical_rule_summary(row: Dict[str, object]) -> Dict[str, object]:
    """Extract the empirical threshold information from one experiment row."""
    resolved = bool(row.get("resolved", False))
    return {
        "empirical_rule_resolved": resolved,
        "empirical_rule_r": int(row["r_star"]) if resolved and row.get("r_star") is not None else None,
        "empirical_rule_error": float(row["error_at_r_star"]) if resolved and row.get("error_at_r_star") is not None else None,
        "empirical_rule_status": "resolved" if resolved else "censored_at_r_max",
    }


def worst_case_commutator_proxy_rule(
    terms: Sequence[Term],
    t: float,
    eps: float,
) -> Dict[str, object]:
    """Return a state-independent first-order commutator proxy for recommended r.

    This is *not* presented as a theorem-certified fidelity threshold. It is the
    cleanest transparent rule available from the current repository alone.

    We use the linearized first-order operator-norm proxy

        || U(t) - U_Trotter^(r)(t) ||  ≲  (t^2 / (2 r)) sum_{a<b} ||[H_a, H_b]||,

    and solve the right-hand side <= eps for r.

    Because the project's reported accuracy metric is state fidelity error rather than
    operator norm, this rule is explicitly labeled as a *proxy*.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    commutator_sum = absolute_commutator_sum(terms)
    if commutator_sum == 0.0:
        recommended_r = 1
    else:
        recommended_r = int(np.ceil((float(t) ** 2) * commutator_sum / (2.0 * float(eps))))
        recommended_r = max(recommended_r, 1)
    return {
        "worst_case_rule_name": "first_order_commutator_proxy",
        "worst_case_rule_r": recommended_r,
        "worst_case_rule_status": "proxy_not_certified_for_fidelity",
        "worst_case_rule_commutator_sum": float(commutator_sum),
        "worst_case_rule_formula": "(t^2 / (2 r)) * sum_{a<b} ||[H_a, H_b]|| <= eps",
        "worst_case_rule_rigorous_for_current_metric": False,
    }


def low_energy_rule_placeholder(
    N: int,
    Delta: float,
    t: float,
    eps: float,
    rule_callable: LowEnergyRuleCallable | None = None,
) -> Dict[str, object]:
    """Placeholder interface for a theorem-specific low-energy rule.

    What is missing from the uploaded materials is the explicit theorem-to-code map:
    which bound from the source paper you want to use, what constants appear in the
    bound, what Hamiltonian normalization/energy shift is assumed, and how the bound's
    epsilon should be related to the notebook's fidelity error threshold.

    If you later provide a callable with signature (N, Delta, t, eps) -> r, this
    function will wrap it into the common reporting format.
    """
    if rule_callable is None:
        return {
            "low_energy_rule_name": "placeholder_requires_theorem_specific_input",
            "low_energy_rule_r": None,
            "low_energy_rule_status": "missing_theorem_formula_or_constants",
            "low_energy_rule_missing_inputs": (
                "Need the exact low-energy bound to implement, including constants, "
                "normalization/energy-shift conventions, and how its accuracy parameter "
                "maps onto the notebook's fidelity-error threshold."
            ),
            "low_energy_rule_rigorous_for_current_metric": False,
        }
    r_value = rule_callable(int(N), float(Delta), float(t), float(eps))
    return {
        "low_energy_rule_name": getattr(rule_callable, "__name__", "user_supplied_low_energy_rule"),
        "low_energy_rule_r": None if r_value is None else int(np.ceil(float(r_value))),
        "low_energy_rule_status": "user_supplied",
        "low_energy_rule_rigorous_for_current_metric": False,
    }
