"""Exact evolution, first-order Trotter evolution, and empirical threshold helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.linalg import expm

Term = Dict[str, object]


def energy_expectation(H: np.ndarray, psi: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, H @ psi)))


def energy_variance(H: np.ndarray, psi: np.ndarray) -> float:
    """Return σ_H² = ⟨H²⟩ - ⟨H⟩² for the state psi."""
    exp_H2 = float(np.real(np.vdot(psi, H @ (H @ psi))))
    exp_H = energy_expectation(H, psi)
    return float(exp_H2 - exp_H ** 2)


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    value = float(np.abs(np.vdot(psi, phi)) ** 2)
    if value < 0 and abs(value) < 1e-12:
        return 0.0
    if value > 1 and abs(value - 1.0) < 1e-12:
        return 1.0
    return value


def exact_propagator(H: np.ndarray, t: float) -> np.ndarray:
    return expm(-1j * H * t)


def evolve_exact(H: np.ndarray, psi: np.ndarray, t: float, U_exact: np.ndarray | None = None) -> np.ndarray:
    if U_exact is None:
        U_exact = exact_propagator(H, t)
    return U_exact @ psi


def apply_pauli_rotation_to_state(P: np.ndarray, theta: float, psi: np.ndarray) -> np.ndarray:
    return np.cos(theta) * psi - 1j * np.sin(theta) * (P @ psi)


def _ordered_terms(terms: Sequence[Term], order: str = "forward") -> List[Term]:
    if order == "forward":
        return list(terms)
    if order == "reverse":
        return list(reversed(terms))
    raise ValueError("order must be 'forward' or 'reverse'")


def evaluate_trotter_error(
    terms: Sequence[Term],
    H: np.ndarray,
    psi0: np.ndarray,
    t: float,
    r: int,
    order: str = "forward",
    method: str = "pauli",
    U_exact: np.ndarray | None = None,
) -> Dict[str, float | int]:
    if r <= 0:
        raise ValueError("r must be positive")
    psi_exact = evolve_exact(H, psi0, t, U_exact=U_exact)
    psi_trot = evolve_trotter(terms, psi0, t, r, method=method, order=order)
    fidelity = state_fidelity(psi_exact, psi_trot)
    return {"r": int(r), "fidelity": float(fidelity), "error": float(1.0 - fidelity)}


def evolve_trotter_1st_pauli(
    terms: Sequence[Term],
    psi: np.ndarray,
    t: float,
    r: int,
    order: str = "forward",
) -> np.ndarray:
    if r <= 0:
        raise ValueError("r must be positive")
    dt = t / r
    vec = psi.copy()
    term_list = _ordered_terms(terms, order)
    for _ in range(r):
        for term in term_list:
            coeff = float(term["coeff"])
            P = term["matrix"]
            vec = apply_pauli_rotation_to_state(P, coeff * dt, vec)
    return vec


def evolve_trotter_1st_expm(
    terms: Sequence[Term],
    psi: np.ndarray,
    t: float,
    r: int,
    order: str = "forward",
) -> np.ndarray:
    if r <= 0:
        raise ValueError("r must be positive")
    dt = t / r
    vec = psi.copy()
    term_list = _ordered_terms(terms, order)
    for _ in range(r):
        for term in term_list:
            coeff = float(term["coeff"])
            P = term["matrix"]
            vec = expm(-1j * coeff * P * dt) @ vec
    return vec


def evolve_trotter(
    terms: Sequence[Term],
    psi: np.ndarray,
    t: float,
    r: int,
    method: str = "pauli",
    order: str = "forward",
) -> np.ndarray:
    if method == "pauli":
        return evolve_trotter_1st_pauli(terms, psi, t, r, order=order)
    if method == "expm":
        return evolve_trotter_1st_expm(terms, psi, t, r, order=order)
    raise ValueError("method must be 'pauli' or 'expm'")


def sweep_r_values(
    terms: Sequence[Term],
    H: np.ndarray,
    psi0: np.ndarray,
    t: float,
    r_values: Sequence[int],
    order: str = "forward",
    method: str = "pauli",
    U_exact: np.ndarray | None = None,
) -> List[Dict[str, float | int]]:
    psi_exact = evolve_exact(H, psi0, t, U_exact=U_exact)
    rows: List[Dict[str, float | int]] = []
    for r in r_values:
        psi_trot = evolve_trotter(terms, psi0, t, int(r), method=method, order=order)
        fidelity = state_fidelity(psi_exact, psi_trot)
        rows.append({"r": int(r), "fidelity": float(fidelity), "error": float(1.0 - fidelity)})
    return rows


def find_r_star_linear(
    terms: Sequence[Term],
    H: np.ndarray,
    psi0: np.ndarray,
    t: float,
    eps: float,
    r_max: int = 120,
    order: str = "forward",
    method: str = "pauli",
    U_exact: np.ndarray | None = None,
) -> Dict[str, float | int | None | bool]:
    """Empirically scan r = 1, ..., r_max and report threshold/censoring information.

    Returns both the threshold result and the endpoint information at r_max so unresolved
    cases remain explicit instead of being silently replaced by r_max + 1.
    """
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    psi_exact = evolve_exact(H, psi0, t, U_exact=U_exact)
    best_error = np.inf
    best_r = None
    endpoint_fidelity = None
    endpoint_error = None

    for r in range(1, r_max + 1):
        psi_trot = evolve_trotter(terms, psi0, t, r, method=method, order=order)
        fidelity = state_fidelity(psi_exact, psi_trot)
        error = float(1.0 - fidelity)
        if error < best_error:
            best_error = error
            best_r = r
        if r == r_max:
            endpoint_fidelity = float(fidelity)
            endpoint_error = float(error)
        if error <= eps:
            if endpoint_error is None:
                endpoint = evaluate_trotter_error(
                    terms, H, psi0, t, r_max, order=order, method=method, U_exact=U_exact
                )
                endpoint_fidelity = float(endpoint["fidelity"])
                endpoint_error = float(endpoint["error"])
            return {
                "resolved": True,
                "r_star": int(r),
                "fidelity_at_r_star": float(fidelity),
                "error_at_r_star": float(error),
                "fidelity_at_r_max": float(endpoint_fidelity),
                "error_at_r_max": float(endpoint_error),
                "best_r_in_scan": int(best_r),
                "best_error_in_scan": float(best_error),
            }

    return {
        "resolved": False,
        "r_star": None,
        "fidelity_at_r_star": None,
        "error_at_r_star": None,
        "fidelity_at_r_max": float(endpoint_fidelity),
        "error_at_r_max": float(endpoint_error),
        "best_r_in_scan": int(best_r) if best_r is not None else None,
        "best_error_in_scan": float(best_error),
    }


def compare_trotter_implementations(
    terms: Sequence[Term],
    psi0: np.ndarray,
    t: float,
    r: int,
    order: str = "forward",
) -> Dict[str, float | int]:
    psi_pauli = evolve_trotter(terms, psi0, t, r, method="pauli", order=order)
    psi_expm = evolve_trotter(terms, psi0, t, r, method="expm", order=order)
    return {
        "r": int(r),
        "state_difference_norm": float(np.linalg.norm(psi_pauli - psi_expm)),
        "implementation_fidelity": float(state_fidelity(psi_pauli, psi_expm)),
    }
