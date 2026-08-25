## Verification package for Chapter 8

This directory contains the verification scripts associated with **Chapter 8: Gravity as an Emergent Causal-Geometric Deformation**.

### Directory structure

#### `scripts/`

Contains self-contained Python verification scripts.

#### `expected_output/`

Contains the expected deterministic output for each verification script.

#### `manifest.txt`

Maps manuscript source files to verification scripts and expected outputs.

### How to run

From this directory, run:

```bash
python3 scripts/verify_1_microhistorical_density_and_combinatorial_mass.py
python3 scripts/verify_2_full_combinatorial_perturbation_measure.py
python3 scripts/verify_3_sector_causal_geometric_projection.py
python3 scripts/verify_4_gravitational_attraction_as_sector_deficit.py
python3 scripts/verify_5_black_hole_causal_closure.py
python3 scripts/verify_6_gravitational_waves_compatible_stable_measure.py
```