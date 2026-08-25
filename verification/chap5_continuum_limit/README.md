## Verification package for Chapter 5

This directory contains the verification scripts associated with **Chapter 5: The Continuum Limit, Scale Structure, and Emergent Physical Geometry**.

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
python3 scripts/verify_1_coordinate_basis.py
python3 scripts/verify_2_digit_coordinate_causal_layer.py
python3 scripts/verify_3_normalized_causal_coordinate.py
python3 scripts/verify_4_discrete_causal_interval.py
python3 scripts/verify_5_spectral_structure_causal_layer.py
python3 scripts/verify_6_continuum_coordinate_limit.py
python3 scripts/verify_7_continuum_lorentzian_form.py
python3 scripts/verify_8_scale_structure_causal_cones.py
python3 scripts/verify_9_emergent_stochasticity.py
```