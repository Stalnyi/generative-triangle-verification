## Verification package for Chapter 6

This directory contains the verification scripts associated with **Chapter 6: The Origin of Spin-1/2-Type Structure**.

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
python3 scripts/verify_1_spin_microhistory.py
python3 scripts/verify_2_centered_spectrum_general.py
python3 scripts/verify_3_cumulative_spin_coordinate.py
python3 scripts/verify_4_asymptotic_spin_structure.py
```