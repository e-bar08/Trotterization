"""Hamiltonian builders and structural summaries for the Physics 765 project."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MAP: Dict[str, np.ndarray] = {"I": I2, "X": X, "Y": Y, "Z": Z}
TWO_LOCAL_CHOICES = [(p, q) for p in "XYZ" for q in "XYZ"]
ONE_LOCAL_CHOICES = list("XYZ")

Term = Dict[str, object]


def kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    if len(mats) == 0:
        raise ValueError("kron_all requires at least one matrix")
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


def pauli_string_matrix(pauli_string: str) -> np.ndarray:
    mats = []
    for p in pauli_string:
        if p not in PAULI_MAP:
            raise ValueError(f"invalid Pauli letter {p!r} in {pauli_string!r}")
        mats.append(PAULI_MAP[p])
    return kron_all(mats)


def make_term(coeff: float, pauli_string: str) -> Term:
    return {
        "coeff": float(coeff),
        "pauli_string": str(pauli_string),
        "matrix": pauli_string_matrix(pauli_string),
    }


def build_H(terms: Sequence[Term]) -> np.ndarray:
    if not terms:
        raise ValueError("build_H requires at least one term")
    dim = int(terms[0]["matrix"].shape[0])
    H = np.zeros((dim, dim), dtype=complex)
    for term in terms:
        H += float(term["coeff"]) * term["matrix"]
    return H


def coeff_array(terms: Sequence[Term]) -> np.ndarray:
    return np.array([float(term["coeff"]) for term in terms], dtype=float)


def coeff_rms_from_terms(terms: Sequence[Term]) -> float:
    coeffs = coeff_array(terms)
    return float(np.sqrt(np.mean(coeffs ** 2)))


def coeff_l1_from_terms(terms: Sequence[Term]) -> float:
    return float(np.sum(np.abs(coeff_array(terms))))


def coeff_l2_from_terms(terms: Sequence[Term]) -> float:
    return float(np.linalg.norm(coeff_array(terms)))


def one_site_pauli_string(N: int, i: int, p: str) -> str:
    if not (0 <= i < N):
        raise ValueError("qubit index out of range")
    chars = ["I"] * N
    chars[i] = p
    return "".join(chars)


def two_site_pauli_string(N: int, i: int, p: str, j: int, q: str) -> str:
    if i == j:
        raise ValueError("two_site_pauli_string requires distinct indices")
    if not (0 <= i < N and 0 <= j < N):
        raise ValueError("qubit index out of range")
    chars = ["I"] * N
    chars[i] = p
    chars[j] = q
    return "".join(chars)


def pauli_support(pauli_string: str) -> List[Tuple[int, str]]:
    return [(idx, p) for idx, p in enumerate(pauli_string) if p != "I"]


def dipolar_terms_xxyy_m2zz(N: int, J0: float = 1.0, alpha: float = 3.0) -> List[Term]:
    r"""Build

    H_dip = sum_{i<j} J_ij (X_i X_j + Y_i Y_j - 2 Z_i Z_j),
    with J_ij = J0 / |i-j|^alpha.
    """
    terms: List[Term] = []
    for i in range(N):
        for j in range(i + 1, N):
            Jij = J0 / (abs(j - i) ** alpha)
            terms.append(make_term(+Jij, two_site_pauli_string(N, i, "X", j, "X")))
            terms.append(make_term(+Jij, two_site_pauli_string(N, i, "Y", j, "Y")))
            terms.append(make_term(-2.0 * Jij, two_site_pauli_string(N, i, "Z", j, "Z")))
    return terms


def shifted_tfim_psd_terms(
    N: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> List[Term]:
    r"""Build a theory-aligned positive-semidefinite local family

    H = J sum_i (I - Z_i Z_{i+1}) / 2 + h sum_i (I - X_i) / 2,

    with open boundaries by default. This Hamiltonian is a sum of positive-semidefinite
    one- and two-local terms. The identity terms only shift the spectrum and do not affect
    Trotter fidelity, but they *do* affect the initial energy Delta used in a low-energy rule.
    """
    terms: List[Term] = []

    pair_sites = [(i, i + 1) for i in range(N - 1)]
    if periodic and N > 2:
        pair_sites.append((N - 1, 0))

    for i, j in pair_sites:
        terms.append(make_term(+0.5 * J, "I" * N))
        terms.append(make_term(-0.5 * J, two_site_pauli_string(N, i, "Z", j, "Z")))

    for i in range(N):
        terms.append(make_term(+0.5 * h, "I" * N))
        terms.append(make_term(-0.5 * h, one_site_pauli_string(N, i, "X")))
    return terms


def canonical_two_local_basis_strings(N: int) -> List[str]:
    strings: List[str] = []
    for i in range(N):
        for j in range(i + 1, N):
            for p, q in TWO_LOCAL_CHOICES:
                strings.append(two_site_pauli_string(N, i, p, j, q))
    return strings


def random_two_local_pauli_terms(
    N: int,
    M: int,
    coeff_scale: float = 1.0,
    rng: np.random.Generator | None = None,
    unique: bool = True,
) -> List[Term]:
    """Build a random two-local Pauli Hamiltonian with M terms.

    When unique=True, Pauli strings are sampled without replacement whenever possible.
    """
    if rng is None:
        rng = np.random.default_rng()
    basis_strings = canonical_two_local_basis_strings(N)
    if unique and M <= len(basis_strings):
        chosen_strings = list(rng.choice(basis_strings, size=M, replace=False))
    else:
        chosen_strings = list(rng.choice(basis_strings, size=M, replace=True))
    coeffs = rng.normal(loc=0.0, scale=coeff_scale, size=M)
    return [make_term(float(c), s) for c, s in zip(coeffs, chosen_strings)]


def randomize_pauli_string_on_same_support(pauli_string: str, rng: np.random.Generator) -> str:
    support = pauli_support(pauli_string)
    N = len(pauli_string)
    if len(support) == 0:
        return "I" * N
    chars = ["I"] * N
    if len(support) == 1:
        idx, _ = support[0]
        chars[idx] = str(rng.choice(ONE_LOCAL_CHOICES))
    elif len(support) == 2:
        (i, _), (j, _) = support
        p, q = rng.choice(ONE_LOCAL_CHOICES), rng.choice(ONE_LOCAL_CHOICES)
        chars[i] = str(p)
        chars[j] = str(q)
    else:
        raise ValueError("This helper only supports support size 0, 1, or 2")
    return "".join(chars)


def matched_random_pauli_terms_same_support(
    structured_terms: Sequence[Term],
    rng: np.random.Generator | None = None,
) -> List[Term]:
    """Match term count, support profile, and coefficient values, but randomize Pauli letters.

    Identity terms are kept as identity terms so the energy shift is preserved exactly.
    """
    if rng is None:
        rng = np.random.default_rng()
    out: List[Term] = []
    for term in structured_terms:
        coeff = float(term["coeff"])
        pauli_string = str(term["pauli_string"])
        randomized = randomize_pauli_string_on_same_support(pauli_string, rng)
        out.append(make_term(coeff, randomized))
    return out


def rescale_coefficients_to_match_l2(terms: Sequence[Term], target_l2: float) -> List[Term]:
    coeffs = coeff_array(terms)
    current_l2 = float(np.linalg.norm(coeffs))
    if current_l2 == 0:
        return [make_term(0.0, str(term["pauli_string"])) for term in terms]
    scale = target_l2 / current_l2
    return [make_term(scale * float(term["coeff"]), str(term["pauli_string"])) for term in terms]


def matched_random_two_local_pauli_terms(
    N: int,
    structured_terms: Sequence[Term],
    rng: np.random.Generator | None = None,
    unique: bool = True,
) -> List[Term]:
    """Random two-local baseline matched in term count and coefficient l2 norm."""
    M = len(structured_terms)
    target_l2 = coeff_l2_from_terms(structured_terms)
    trial_terms = random_two_local_pauli_terms(N, M=M, coeff_scale=1.0, rng=rng, unique=unique)
    return rescale_coefficients_to_match_l2(trial_terms, target_l2)


def pauli_strings_commute(pauli_a: str, pauli_b: str) -> bool:
    if len(pauli_a) != len(pauli_b):
        raise ValueError("Pauli strings must have the same length")
    anticommute_count = 0
    for a, b in zip(pauli_a, pauli_b):
        if a == "I" or b == "I" or a == b:
            continue
        anticommute_count += 1
    return (anticommute_count % 2) == 0


def count_noncommuting_pairs(terms: Sequence[Term]) -> int:
    count = 0
    for term_a, term_b in combinations(terms, 2):
        if not pauli_strings_commute(str(term_a["pauli_string"]), str(term_b["pauli_string"])):
            count += 1
    return count


def count_distinct_qubit_pairs(terms: Sequence[Term]) -> int:
    pairs = set()
    for term in terms:
        support = [idx for idx, _ in pauli_support(str(term["pauli_string"]))]
        if len(support) == 2:
            pairs.add(tuple(support))
    return len(pairs)


def count_single_site_terms(terms: Sequence[Term]) -> int:
    return sum(1 for term in terms if len(pauli_support(str(term["pauli_string"]))) == 1)


def count_two_site_terms(terms: Sequence[Term]) -> int:
    return sum(1 for term in terms if len(pauli_support(str(term["pauli_string"]))) == 2)


def count_identity_terms(terms: Sequence[Term]) -> int:
    return sum(1 for term in terms if len(pauli_support(str(term["pauli_string"]))) == 0)


def count_nonidentity_terms(terms: Sequence[Term]) -> int:
    return sum(1 for term in terms if len(pauli_support(str(term["pauli_string"]))) > 0)


def random_two_local_generation_info(
    N: int,
    M: int,
    target_l2: float,
    seed: int | None = None,
    unique: bool = True,
) -> Dict[str, object]:
    """Return a dict describing how a matched random two-local baseline was generated.

    Intended for logging and reproducibility in paper exports.
    """
    basis_strings = canonical_two_local_basis_strings(N)
    sampling_mode = "without_replacement" if (unique and M <= len(basis_strings)) else "with_replacement"
    return {
        "generator": "matched_random_two_local_pauli_terms",
        "N": int(N),
        "num_terms_M": int(M),
        "pauli_pool": "all 2-local Pauli strings on N qubits (P_i Q_j, P,Q in {X,Y,Z}, i<j)",
        "num_basis_strings_available": int(len(basis_strings)),
        "coefficient_distribution": "normal(mean=0, std=1)",
        "coefficient_rescaling": f"rescaled to match target L2 norm = {target_l2:.6g}",
        "target_l2_norm": float(target_l2),
        "sampling_mode": sampling_mode,
        "unique_pauli_strings": bool(unique and M <= len(basis_strings)),
        "seed": seed,
    }


def family_summary(terms: Sequence[Term]) -> Dict[str, float | int]:
    return {
        "num_terms": len(terms),
        "num_identity_terms": count_identity_terms(terms),
        "num_nonidentity_terms": count_nonidentity_terms(terms),
        "num_single_site_terms": count_single_site_terms(terms),
        "num_two_site_terms": count_two_site_terms(terms),
        "distinct_qubit_pairs": count_distinct_qubit_pairs(terms),
        "coeff_rms": coeff_rms_from_terms(terms),
        "coeff_l1": coeff_l1_from_terms(terms),
        "coeff_l2": coeff_l2_from_terms(terms),
        "noncommuting_pairs": count_noncommuting_pairs(terms),
    }
