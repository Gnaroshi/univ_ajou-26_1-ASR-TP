# hs

Heart sound classification project for Ajou graduate ASR course.

## TODO
- README.md
    - add how to setup environments (e.g. weights downloads)

## Current task
- CirCor outcome classification (Normal vs Abnormal)
- M2D-based baseline

## Setup
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Data

Set:
```bash
export CIRCOR_ROOT=~/private/ajou_grad/dataset/physionet.org
python scripts/prepare_circor.py --circor_root $CIRCOR_ROOT --out_dir artifacts/circor
```

## Team Member
Mark Zuckerberg
meansash
Jensen Huang

