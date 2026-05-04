"""Circuit-level resource estimation using a transparent local compilation model.

Resource-model scope:
- exact-study numerics do not depend on this module;
- counts reported here are for the chosen decomposition model
  (basis changes + parity CNOTs + synthesized Rz for non-identity local Pauli terms);
- T counts are estimates coming from pyLIQTR when available, or from a documented fallback;
- the sequential-layer quantity is a proxy, not a compiled hardware depth;
- identity terms contribute only a global phase and therefore zero circuit cost.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # optional dependency
    import cirq
    HAVE_CIRQ = True
except Exception:  # pragma: no cover - environment dependent
    cirq = None
    HAVE_CIRQ = False

try:  # optional dependency
    import stim
    HAVE_STIM = True
except Exception:  # pragma: no cover - environment dependent
    stim = None
    HAVE_STIM = False

try:  # optional dependency
    from pyLIQTR.gate_decomp.rotation_gates import T_COUNT_CONST, T_COUNT_SLOPE, rz_decomp
    HAVE_PYLIQTR = True
except Exception:  # pragma: no cover - fallback only
    rz_decomp = None
    T_COUNT_SLOPE = 3.02
    T_COUNT_CONST = 0.77
    HAVE_PYLIQTR = False

Term = Dict[str, object]

PRE_BASIS_CHANGE = {
    "I": [],
    "Z": [],
    "X": ["H"],
    "Y": ["S_DAG", "H"],
}
POST_BASIS_CHANGE = {
    "I": [],
    "Z": [],
    "X": ["H"],
    "Y": ["H", "S"],
}
SINGLE_CLIFFORD_COUNT = {"I": 0, "Z": 0, "X": 2, "Y": 4}



def local_support(pauli_string: str) -> List[Tuple[int, str]]:
    support = [(idx, p) for idx, p in enumerate(pauli_string) if p != "I"]
    if len(support) > 2:
        raise ValueError(f"expected support size <= 2, got {pauli_string!r}")
    return support


def _append_cirq_single_qubit_ops(ops, qubit, names: Sequence[str]) -> None:
    if not HAVE_CIRQ:
        raise ImportError("Cirq is not available in this environment")
    for name in names:
        if name == "H":
            ops.append(cirq.H(qubit))
        elif name == "S":
            ops.append(cirq.S(qubit))
        elif name == "S_DAG":
            ops.append(cirq.S(qubit) ** -1)
        else:
            raise ValueError(f"unknown Clifford gate name {name!r}")


def _append_stim_single_qubit_ops(circuit, qubit: int, names: Sequence[str]) -> None:
    if not HAVE_STIM:
        raise ImportError("Stim is not available in this environment")
    for name in names:
        circuit.append(name, [qubit])


@lru_cache(maxsize=None)
def _cached_t_count(angle_key: float, precision_key: float) -> int:
    if precision_key <= 0:
        raise ValueError("precision must be positive")
    if HAVE_PYLIQTR and rz_decomp is not None:
        gate = rz_decomp(rads=float(angle_key), precision=float(precision_key))
        return int(gate.get_T_count())
    return int(np.ceil(T_COUNT_SLOPE * np.log2(1.0 / precision_key) + T_COUNT_CONST))


def pyliqtr_t_count_for_rz(angle_rads: float, precision: float) -> int:
    return _cached_t_count(round(float(angle_rads), 14), float(precision))


def build_cirq_term_circuit(
    pauli_string: str,
    theta: float,
    qubits=None,
    rotation_precision: float = 1e-10,
):
    if not HAVE_CIRQ:
        raise ImportError("Cirq is required to build the explicit circuit model")
    if qubits is None:
        qubits = cirq.LineQubit.range(len(pauli_string))

    support = local_support(pauli_string)
    if len(support) == 0:
        return cirq.Circuit()

    ops = []
    if len(support) == 1:
        i, p = support[0]
        qi = qubits[i]
        _append_cirq_single_qubit_ops(ops, qi, PRE_BASIS_CHANGE[p])
        if HAVE_PYLIQTR and rz_decomp is not None:
            ops.append(rz_decomp(rads=float(2.0 * theta), precision=float(rotation_precision)).on(qi))
        else:
            ops.append(cirq.rz(2.0 * theta).on(qi))
        _append_cirq_single_qubit_ops(ops, qi, POST_BASIS_CHANGE[p])
        return cirq.Circuit(ops)

    (i, p_i), (j, p_j) = support
    qi, qj = qubits[i], qubits[j]
    _append_cirq_single_qubit_ops(ops, qi, PRE_BASIS_CHANGE[p_i])
    _append_cirq_single_qubit_ops(ops, qj, PRE_BASIS_CHANGE[p_j])
    ops.append(cirq.CNOT(qi, qj))
    if HAVE_PYLIQTR and rz_decomp is not None:
        ops.append(rz_decomp(rads=float(2.0 * theta), precision=float(rotation_precision)).on(qj))
    else:
        ops.append(cirq.rz(2.0 * theta).on(qj))
    ops.append(cirq.CNOT(qi, qj))
    _append_cirq_single_qubit_ops(ops, qi, POST_BASIS_CHANGE[p_i])
    _append_cirq_single_qubit_ops(ops, qj, POST_BASIS_CHANGE[p_j])
    return cirq.Circuit(ops)


def build_cirq_first_order_step(
    terms: Sequence[Term],
    dt: float,
    rotation_precision: float = 1e-10,
):
    if not HAVE_CIRQ:
        raise ImportError("Cirq is required to build the explicit circuit model")
    N = len(str(terms[0]["pauli_string"]))
    qubits = cirq.LineQubit.range(N)
    circuit = cirq.Circuit()
    for term in terms:
        circuit += build_cirq_term_circuit(
            str(term["pauli_string"]),
            float(term["coeff"]) * float(dt),
            qubits=qubits,
            rotation_precision=rotation_precision,
        )
    return circuit


def build_stim_clifford_skeleton_term(pauli_string: str):
    if not HAVE_STIM:
        raise ImportError("Stim is required to export the Clifford skeleton")
    support = local_support(pauli_string)
    circuit = stim.Circuit()
    if len(support) == 0:
        return circuit
    if len(support) == 1:
        i, p = support[0]
        _append_stim_single_qubit_ops(circuit, i, PRE_BASIS_CHANGE[p])
        _append_stim_single_qubit_ops(circuit, i, POST_BASIS_CHANGE[p])
        return circuit
    (i, p_i), (j, p_j) = support
    _append_stim_single_qubit_ops(circuit, i, PRE_BASIS_CHANGE[p_i])
    _append_stim_single_qubit_ops(circuit, j, PRE_BASIS_CHANGE[p_j])
    circuit.append("CX", [i, j])
    circuit.append("CX", [i, j])
    _append_stim_single_qubit_ops(circuit, i, POST_BASIS_CHANGE[p_i])
    _append_stim_single_qubit_ops(circuit, j, POST_BASIS_CHANGE[p_j])
    return circuit


def build_stim_clifford_skeleton_step(terms: Sequence[Term]):
    if not HAVE_STIM:
        raise ImportError("Stim is required to export the Clifford skeleton")
    circuit = stim.Circuit()
    for term in terms:
        circuit += build_stim_clifford_skeleton_term(str(term["pauli_string"]))
    return circuit


def count_stim_operations(circuit) -> Dict[str, int]:
    counts = Counter()
    for inst in circuit:
        counts[inst.name] += 1
    counts["num_instructions"] = len(circuit)
    counts["num_qubits"] = circuit.num_qubits
    return dict(counts)


def estimate_step_resources(
    terms: Sequence[Term],
    dt: float,
    rotation_precision: float = 1e-10,
) -> Dict[str, float | int | bool | None | str]:
    single_clifford_count = 0
    cnot_count = 0
    rotation_gate_count = 0
    t_count_estimate = 0
    identity_term_count = 0

    for term in terms:
        pauli_string = str(term["pauli_string"])
        support = local_support(pauli_string)
        if len(support) == 0:
            identity_term_count += 1
            continue
        rotation_gate_count += 1
        theta = float(term["coeff"]) * float(dt)
        if len(support) == 1:
            (_, p_i) = support[0]
            single_clifford_count += SINGLE_CLIFFORD_COUNT[p_i]
        else:
            (_, p_i), (_, p_j) = support
            single_clifford_count += SINGLE_CLIFFORD_COUNT[p_i] + SINGLE_CLIFFORD_COUNT[p_j]
            cnot_count += 2
        t_count_estimate += pyliqtr_t_count_for_rz(2.0 * theta, rotation_precision)

    sequential_layer_proxy = single_clifford_count + cnot_count + rotation_gate_count

    out: Dict[str, float | int | bool | None | str] = {
        "resource_model": "basis_change_plus_parity_cnot_plus_synthesized_rz",
        "step_dt": float(dt),
        "identity_term_count_per_step": int(identity_term_count),
        "single_qubit_clifford_count_per_step": int(single_clifford_count),
        "cnot_count_per_step": int(cnot_count),
        "rotation_gate_count_per_step": int(rotation_gate_count),
        "t_count_estimate_per_step": int(t_count_estimate),
        "sequential_layer_proxy_per_step": int(sequential_layer_proxy),
        "resource_has_stim": bool(HAVE_STIM),
        "resource_has_pyliqtr": bool(HAVE_PYLIQTR),
    }

    if HAVE_STIM:
        stim_skeleton = build_stim_clifford_skeleton_step(terms)
        out["stim_instruction_count_clifford_skeleton_per_step"] = int(len(stim_skeleton))
        out["stim_num_qubits"] = int(stim_skeleton.num_qubits)
        stim_counts = count_stim_operations(stim_skeleton)
        out.update({f"stim_{k.lower()}": int(v) for k, v in stim_counts.items() if isinstance(v, int)})
    else:
        out["stim_instruction_count_clifford_skeleton_per_step"] = None
        out["stim_num_qubits"] = None
    return out


def estimate_total_resources_for_r(
    terms: Sequence[Term],
    t: float,
    r: int,
    rotation_precision: float = 1e-10,
) -> Dict[str, float | int | bool | None | str]:
    if r <= 0:
        raise ValueError("r must be positive")
    per_step = estimate_step_resources(terms, dt=float(t) / int(r), rotation_precision=rotation_precision)
    out = dict(per_step)
    out["resource_r"] = int(r)
    out["resource_t"] = float(t)
    for key, value in list(per_step.items()):
        if key.endswith("_per_step") and isinstance(value, (int, float)):
            out[key.replace("_per_step", "_total")] = type(value)(value * int(r))
    return out


def export_stim_skeleton(terms: Sequence[Term], path: str | Path) -> Path | None:
    if not HAVE_STIM:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    circuit = build_stim_clifford_skeleton_step(terms)
    path.write_text(str(circuit), encoding="utf-8")
    return path
