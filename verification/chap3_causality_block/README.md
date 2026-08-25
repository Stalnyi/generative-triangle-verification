## Verification package for Chapter 3

This directory contains the verification scripts associated with **Chapter 3: Causal Structure, DAG Interpretation, and the Block Ontology of Spacetime**.

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
python3 scripts/verify_1_partial_order.py
python3 scripts/verify_2_dag_interpretation.py
python3 scripts/verify_3_block_universe.py
```