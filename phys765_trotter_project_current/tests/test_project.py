from __future__ import annotations

import unittest

import numpy as np

from src.analysis_utils import annotate_resource_estimates, run_theta_sweep_for_family, run_random_instance_study
from src.hamiltonians import (
    build_H,
    dipolar_terms_xxyy_m2zz,
    make_term,
    one_site_pauli_string,
    shifted_tfim_psd_terms,
    two_site_pauli_string,
)
from src.resource_estimation import estimate_total_resources_for_r
from src.selection_rules import worst_case_commutator_proxy_rule
from src.states import basis_state, plus_state
from src.trotter_utils import compare_trotter_implementations, sweep_r_values


class Physics765ProjectTests(unittest.TestCase):
    def test_pauli_and_expm_trotter_match(self):
        N = 4
        terms = dipolar_terms_xxyy_m2zz(N)
        info = compare_trotter_implementations(terms, plus_state(N), t=0.5, r=4)
        self.assertLess(info["state_difference_norm"], 1e-10)
        self.assertGreater(info["implementation_fidelity"], 1 - 1e-10)

    def test_zero_state_is_eigenstate_like_for_dipolar(self):
        N = 4
        terms = dipolar_terms_xxyy_m2zz(N)
        H = build_H(terms)
        psi0 = basis_state(N, "0" * N)
        rows = sweep_r_values(terms, H, psi0, t=0.7, r_values=[1, 2, 3, 4])
        for row in rows:
            self.assertLess(abs(row["error"]), 1e-10)

    def test_two_local_resource_counts(self):
        term = make_term(0.5, two_site_pauli_string(2, 0, "X", 1, "X"))
        resources = estimate_total_resources_for_r([term], t=1.0, r=1, rotation_precision=1e-6)
        self.assertEqual(resources["cnot_count_per_step"], 2)
        self.assertEqual(resources["rotation_gate_count_per_step"], 1)
        self.assertEqual(resources["single_qubit_clifford_count_per_step"], 4)
        self.assertGreater(resources["t_count_estimate_per_step"], 0)

    def test_one_local_resource_counts(self):
        term = make_term(0.5, one_site_pauli_string(2, 0, "X"))
        resources = estimate_total_resources_for_r([term], t=1.0, r=1, rotation_precision=1e-6)
        self.assertEqual(resources["cnot_count_per_step"], 0)
        self.assertEqual(resources["rotation_gate_count_per_step"], 1)
        self.assertEqual(resources["single_qubit_clifford_count_per_step"], 2)
        self.assertGreater(resources["t_count_estimate_per_step"], 0)

    def test_unresolved_rows_do_not_get_total_resources(self):
        N = 5
        terms = dipolar_terms_xxyy_m2zz(N)
        H = build_H(terms)
        rows = run_theta_sweep_for_family(
            family_name="structured",
            terms=terms,
            H=H,
            N=N,
            t=1.0,
            eps=1e-12,
            r_max=2,
            num_thetas=3,
        )
        rows = annotate_resource_estimates(rows, terms, rotation_precision=1e-6)
        self.assertTrue(any(not row["resolved"] for row in rows))
        for row in rows:
            if not row["resolved"]:
                self.assertFalse(row["resource_totals_reported"])
                self.assertIsNone(row["resource_r"])
                self.assertNotIn("t_count_estimate_total", row)

    def test_worst_case_rule_returns_positive_integer(self):
        terms = dipolar_terms_xxyy_m2zz(4)
        rule = worst_case_commutator_proxy_rule(terms, t=1.0, eps=1e-3)
        self.assertGreaterEqual(rule["worst_case_rule_r"], 1)
        self.assertFalse(rule["worst_case_rule_rigorous_for_current_metric"])

    def test_shifted_tfim_builder_contains_identity_and_local_terms(self):
        terms = shifted_tfim_psd_terms(4, J=1.0, h=0.7)
        pauli_strings = [str(term["pauli_string"]) for term in terms]
        self.assertIn("IIII", pauli_strings)
        self.assertTrue(any(s.count("X") == 1 for s in pauli_strings))
        self.assertTrue(any(s.count("Z") == 2 for s in pauli_strings))

    def test_random_instance_study_starts_at_one(self):
        N = 4
        structured_terms = dipolar_terms_xxyy_m2zz(N)
        rows, summaries = run_random_instance_study(
            N=N,
            structured_terms=structured_terms,
            t=0.5,
            eps=1e-3,
            num_instances=2,
            num_thetas=3,
            r_max=5,
            rotation_precision=1e-6,
            rng=np.random.default_rng(0),
            start_index=1,
        )
        ids = {row["instance_id"] for row in rows}
        self.assertEqual(ids, {"random_1", "random_2"})
        self.assertEqual({summary["instance_id"] for summary in summaries}, {"random_1", "random_2"})


if __name__ == "__main__":
    unittest.main()
