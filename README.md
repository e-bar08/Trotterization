# First-Order Trotter Difficulty for Structured and Random Pauli-String Hamiltonians

**Physics 765: Quantum Algorithms and Error Correction — University of Wisconsin–Madison**

This repo contains the code, notebook, paper, and generated outputs for a computational study of how many first-order Trotter steps are actually needed to simulate three different families of Hamiltonians — and how much that number depends on the initial state you start from.

---

## Repository structure

```
.
├── notebooks/
│   └── 765_final_project_updated.ipynb   # main narrative notebook
├── paper/
│   └── trotter_difficulty.tex            # LaTeX paper
├── scripts/
│   └── run_full_study.py                 # reproducible batch entry point
├── src/
│   ├── hamiltonians.py                   # Hamiltonian builders and structural summaries
│   ├── trotter_utils.py                  # exact evolution, Trotter simulation, threshold scan
│   ├── analysis_utils.py                 # study runners, summaries, export helpers
│   ├── plotting_utils.py                 # all plot functions
│   ├── states.py                         # product-state preparation helpers
│   ├── selection_rules.py                # empirical and worst-case rule interfaces
│   └── resource_estimation.py            # circuit-level T-count / CNOT estimation
├── tests/
│   └── test_project.py                   # sanity tests (8 tests)
├── results/
│   ├── figures/                          # saved PNG plots
│   ├── tables/                           # saved CSV tables
│   └── circuits/                         # optional Stim circuit skeletons
└── requirements.txt
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the sanity tests
python -m unittest discover -s tests -p 'test_*.py'

# 3. Run a fast end-to-end batch study
python scripts/run_full_study.py --profile quick

# 4. Open the notebook and run top-to-bottom
jupyter notebook notebooks/765_final_project_updated.ipynb
```

---

## Run profiles

The notebook has its own `RUN_PROFILE` variable (`"quick"`, `"15min"`, or `"paper"`) in the configuration cell at the top.

---

## Programmatic entry point

All experiments can be run from a single function:

```python
from src.analysis_utils import DEFAULT_CONFIG, run_full_study, export_clean_outputs

results = run_full_study({
    **DEFAULT_CONFIG,
    "N": 6, "t": 1.0, "eps": 1e-3, "r_max": 120,
    "num_thetas": 21, "num_instances": 6, "seed": 42,
})

export_clean_outputs(results, "results/tables/")
```

`results` is a dictionary with keys: `main_df`, `all_rows_df`, `instance_summary_df`,
`family_summary_df`, `validation_df`, `ordering_df`, `generation_info`, `config`.

---

## What the study does

For each Hamiltonian and each initial product state, the code finds the minimum number of Trotter steps `r` needed to reach a target fidelity error. Specifically, it computes the exact time-evolved state and the Trotter-approximated state, then finds the smallest `r` where the fidelity error drops below the threshold `eps`. This is called the empirical threshold `r_star`.

The initial states are product states parameterized by an angle theta, sweeping from 0 to pi. The study checks how much `r_star` changes as you vary the initial state, and whether that variation looks different across Hamiltonian families.

### Hamiltonian families

- **Structured dipolar** — a long-range Hamiltonian built from XX+YY-2ZZ interactions with power-law couplings that decay as 1/|i-j|^alpha
- **Random two-local baseline** — M Pauli operators drawn at random (without replacement) from the full pool of two-site Pauli strings on N qubits; coefficients are drawn from a standard normal distribution and then rescaled so the overall coupling strength matches the dipolar instance; every instance is generated from a fixed seed for reproducibility
- **Shifted local TFIM** — a nearest-neighbor transverse-field Ising model written in positive-semidefinite form, included as a local, sparse control case

### Robustness checks

- **Epsilon sweep** — reruns the threshold scan at eps = 0.01, 0.001, and 0.0001 to check that the ordering between families holds across a range of accuracy targets
- **Ordering comparison** — compares `r_star` values under forward, reversed, and randomly permuted term orderings to see how sensitive the results are to the sequence in which terms are applied
- **Energy variance diagnostic** — computes the energy variance of each initial state and plots it against `r_star` as a state-level predictor of Trotter difficulty

---

## Generated outputs

### Figures (`results/figures/`)

| File | Description |
|------|-------------|
| `rstar_vs_delta.png` | Empirical threshold vs initial energy for all three families |
| `rstar_vs_variance.png` | Empirical threshold vs energy variance |
| `structured_error_curves.png` | Fidelity-error curves, structured dipolar |
| `random_error_curves.png` | Fidelity-error curves, random two-local |
| `local_error_curves.png` | Fidelity-error curves, local shifted TFIM |
| `rule_comparison.png` | Empirical thresholds vs worst-case proxy estimates |
| `tcount_vs_delta.png` | Total estimated T-count vs initial energy |
| `rstar_boxplot.png` | Distribution of r_star by family |
| `tcount_boxplot.png` | Total T-count distribution by family |
| `random_instance_variation.png` | Instance-to-instance variation in the random baseline |
| `reference_scaling_tcount.png` | Per-step T-count scaling with system size |
| `validation_eps_sweep.png` | Mean r_star vs epsilon across families (ordering check) |
| `ordering_comparison.png` | r_star distribution by term ordering |
| `summary_dashboard.png` | Six-panel overview dashboard |

### Tables (`results/tables/`)

| File | Description |
|------|-------------|
| `exact_family_state_results.csv` | Per-state results for the three main families |
| `all_state_results.csv` | All state results including random instances |
| `random_instance_state_results.csv` | Per-state results for extra random instances |
| `instance_summary.csv` | Per-instance sweep summaries |
| `family_summary.csv` | Aggregated per-family summaries |
| `random_instance_summary.csv` | Summary stats per random instance |
| `correlation_summary.csv` | r_star vs energy and r_star vs theta correlations |
| `rule_comparison.csv` | Empirical vs worst-case rule comparison |
| `validation_eps_sweep.csv` | Epsilon-sweep validation results |
| `ordering_comparison.csv` | Term-ordering comparison results |
| `reference_scaling_resources.csv` | Fixed-step resource scaling data |
| `validation_summary.csv` | Implementation cross-checks |

---

## Limitations

- Exact fidelity results are only valid at small system sizes where full state-vector simulation is feasible.
- States that don't reach the target within the scan range are flagged as unresolved (`resolved=False`). Summary tables report both resolved-only means and a lower-bound mean that counts unresolved states as contributing the maximum step count.
- The worst-case commutator rule is included as a rough comparison point, not as a rigorous fidelity guarantee.
- Resource estimates are model-based gate counts under a documented local compilation model, not hardware-compiled circuit depths.
- The larger system size resource section shows how per-step cost scales with N, it is not an exact fidelity study.
