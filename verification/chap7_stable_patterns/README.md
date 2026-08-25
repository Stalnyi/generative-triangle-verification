## Verification package for Chapter 7

This directory contains the verification scripts associated with **Chapter 7: Stable Recursive Patterns**.

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
python3 scripts/verify_1_recursive_macro_classes.py
python3 scripts/verify_2_controlled_characteristics_invariants.py
python3 scripts/verify_3_recursive_stability_macroclasses.py
python3 scripts/verify_4_causal_localization_stable_structures.py
python3 scripts/verify_5_stable_internal_structure.py
python3 scripts/verify_6_particle_like_structures.py
python3 scripts/verify_7_microhistorical_multiplicity_combinatorial_characteristics_part_1.py
python3 scripts/verify_7_microhistorical_multiplicity_combinatorial_characteristics_part_2.py
python3 scripts/verify_7_microhistorical_multiplicity_combinatorial_characteristics_part_3.py
```