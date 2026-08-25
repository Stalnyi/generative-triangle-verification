# Generative Triangle T — Symbolic and Computational Verification

This repository contains the verification scripts accompanying the work:

**Generative Triangle \(T\): A Fundamental Discrete Generative Mathematical Structure Giving Rise to Causality, Spacetime, a Spin-\(1/2\)-Like Internal Structure, Stable Particle-Like Patterns, and Gravity as an Emergent Phenomenon without Fine-Tuning**

Author: **Myroslav Stalnyi**

## Purpose

The purpose of this repository is to archive the symbolic and computational verification code used to check the formal constructions developed in the paper.

The scripts verify algebraic identities, recursive definitions, finite combinatorial structures, causal-order constructions, coordinate recursions, spin-like spectra, stable-pattern definitions, microhistorical multiplicities, combinatorial mass formulas, sector deficits, effective horizon regimes, and emergent-gravity constructions appearing in the model.

The verification code is not a replacement for the mathematical proofs in the text. Its role is to provide independent executable checks of the formal identities, bounds, examples, guards, and negative cases described in the appendices.

## Repository structure

```text
verification/
  chap1_mathematical_foundations/
  chap2_coordinate_structure/
  chap3_causality_block/
  chap4_shifted_parallel_spaces/
  chap5_continuum_limit/
  chap6_spin_half_origin/
  chap7_stable_patterns/
  chap8_emergent_gravity/
```

Each chapter directory contains the verification scripts corresponding to the formal results of that chapter.

## Requirements

The scripts are written in Python.

The main external dependency is:

```text
sympy
```

If `sympy` is not installed, install it with:

```bash
python3 -m pip install sympy
```

If your system uses `python` instead of `python3`, use:

```bash
python -m pip install sympy
```

On macOS or Linux, an isolated virtual environment can also be used:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install sympy
```

After this, the verification scripts can be run from the repository root.

## Running the scripts

Run the scripts from the repository root.

Example:

```bash
python verification/chap1_mathematical_foundations/scripts/verify_math_foundations.py
```

If the command fails with:

```text
ModuleNotFoundError: No module named 'sympy'
```

install SymPy first:

```bash
python3 -m pip install sympy
```

Each script prints its own verification certificate or completion message when the corresponding checks pass.

## Scope of verification

The scripts perform symbolic and computational checks of the internal formal structure of the model.

The symbolic checks include algebraic identities, exact simplifications, recursive formulas, closed-form expressions, and exact rational relations.

The computational checks include finite exhaustive tests, numerical examples, boundary cases, guard conditions, and negative tests.

The repository covers verification related to:

- recursive generation rules;
- uniqueness and collision-freeness properties;
- causal order and DAG structure;
- coordinate recursions and causal cones;
- continuum-limit arithmetic;
- spin-like internal spectra;
- stable recursive macroclasses;
- microhistorical multiplicity;
- combinatorial mass;
- sector deficits and emergent-gravity constructions;
- effective horizon and wave-like compatible-measure regimes.

The scripts do not perform empirical validation of the physical interpretation of the model. Physical calibration and comparison with experimental data are treated separately in the paper.

## Reproducibility

This repository preserves the verification code corresponding to the fixed version of the paper.

For reproducible use, run the scripts from a clean Python environment with SymPy installed.

## License

This repository is licensed under the MIT License.
