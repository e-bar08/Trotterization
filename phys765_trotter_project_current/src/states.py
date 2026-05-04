"""State-preparation helpers for the Physics 765 Trotter study."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def basis_state(N: int, bitstring: str) -> np.ndarray:
    if len(bitstring) != N:
        raise ValueError("bitstring length must match N")
    idx = int(bitstring, 2)
    v = np.zeros((2**N,), dtype=complex)
    v[idx] = 1.0
    return v


def plus_state(N: int) -> np.ndarray:
    q = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    out = q
    for _ in range(N - 1):
        out = np.kron(out, q)
    return out


def rotated_zero_product_state(N: int, theta: float) -> np.ndarray:
    q = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)], dtype=complex)
    out = q
    for _ in range(N - 1):
        out = np.kron(out, q)
    return out


def theta_family(N: int, num: int = 21) -> List[Tuple[float, np.ndarray]]:
    thetas = np.linspace(0.0, np.pi, num)
    return [(float(theta), rotated_zero_product_state(N, float(theta))) for theta in thetas]


def random_product_state(N: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    out = np.array([1.0], dtype=complex)
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        theta = np.arccos(1.0 - 2.0 * u)
        phi = 2.0 * np.pi * v
        qubit = np.array([
            np.cos(theta / 2.0),
            np.exp(1j * phi) * np.sin(theta / 2.0),
        ], dtype=complex)
        out = np.kron(out, qubit)
    return out
