# Anonymous PDE benchmark bundle

This bundle contains the released DeepONet checkpoints, the exact training data
associated with those checkpoints, and a split-blind test set for each PDE.
Every public test set mixes samples from more than one regime in randomized
order. Distribution membership and original-source metadata are intentionally
absent from this bundle.

## Layout

```text
problems/<problem>/
  training_data.npz
  test_data.npz
  checkpoint/model.{data-00000-of-00001,index,meta}
```

`cavity_flow` has separate `checkpoint_u/` and `checkpoint_v/` directories.
The five problem names are `diffusion_reaction`, `burgers`, `advection`,
`poisson`, and `cavity_flow`.

The training files retain the arrays used by the original code:

- Four scalar PDEs: `X_train0`, `X_train1`, `y_train`.
- Cavity flow: `branch_train`, `trunk_train`, `u_train`, `v_train`.

Public test files use a uniform sample-oriented representation:

- Scalar PDEs: `sample_id`, `branch`, `trunk`, `solution`.
- Cavity flow: `sample_id`, `branch`, `trunk`, `velocity_u`, `velocity_v`.

For the four scalar PDEs, `trunk` is a shared coordinate array. For cavity
flow, each sample has its own coordinate array. `sample_id` is an anonymous
identifier and does not encode the source regime.

## Environment and inference

Create the pinned environment from the bundle root:

```bash
conda env create --prefix .conda/repro --file environment.yml
```

Run one problem or all problems:

```bash
conda run --prefix .conda/repro python predict.py --problem poisson
conda run --prefix .conda/repro python predict.py --problem all
```

Predictions are written to `predictions/`. Compute aggregate errors without
revealing test membership using:

```bash
conda run --prefix .conda/repro python evaluate.py
```

The TensorFlow-v1 checkpoint alone does not encode the DeepONet transforms, so
`predict.py` contains the matching network definitions for all five PDEs.

`MANIFEST.json` records the size and SHA-256 digest of every public file. No
private key, split label, shuffle seed, original test filename, or original
sample index is included in the public directory or archive.

