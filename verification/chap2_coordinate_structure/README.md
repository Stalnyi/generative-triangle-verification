## Verification package for Chapter 2

This directory contains the verification scripts associated with **Chapter 2: Coordinate Structure of the Generative Triangle**.

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
python3 scripts/verify_1_coordinate_structure.py
python3 scripts/verify_2_positional_coordinate.py
python3 scripts/verify_3_logarithmic_coordinate.py
python3 scripts/verify_4_causal_cones_tr.py
```