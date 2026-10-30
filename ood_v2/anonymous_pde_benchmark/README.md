# DeepONet Models and Training Data

This bundle contains trained DeepONet checkpoints and their original training
arrays for five PDE problems: `diffusion_reaction`, `burgers`, `advection`,
`poisson`, and `cavity_flow`. It also includes the runtime environment and a
prediction script.

## Files

Each `problems/<problem>/` directory contains `training_data.npz`. The four
scalar problems have a `checkpoint/` directory. `cavity_flow` has separate
`checkpoint_u/` and `checkpoint_v/` directories for its two velocity components.

Every checkpoint consists of three files: `model.data-00000-of-00001`,
`model.index`, and `model.meta`. Keep the three files together. The checkpoint
path passed to the runtime is the shared basename, such as
`problems/poisson/checkpoint/model`.

## Training arrays

| Problem | Arrays | Shapes |
| --- | --- | --- |
| `diffusion_reaction`, `burgers`, `advection` | `X_train0`, `X_train1`, `y_train` | `(1000, 101)`, `(10201, 2)`, `(1000, 10201)` |
| `poisson` | `X_train0`, `X_train1`, `y_train` | `(1000, 101)`, `(5082, 2)`, `(1000, 5082)` |
| `cavity_flow` | `branch_train`, `trunk_train`, `u_train`, `v_train` | `(214221, 101)`, `(214221, 2)`, `(214221, 1)`, `(214221, 1)` |

For the four scalar problems, `X_train0[i]` is a branch input,
`X_train1[j]` is a coordinate shared by all branch inputs, and
`y_train[i, j]` is the corresponding target. For `cavity_flow`, row `k` of
all four arrays describes one branch input and coordinate pair with its `u`
and `v` targets. The stored dtypes and array order are unchanged.

Read a file with NumPy, for example:

```python
import numpy as np

with np.load("problems/poisson/training_data.npz") as data:
    branch = data["X_train0"]
    coordinates = data["X_train1"]
    targets = data["y_train"]
```

## Run a model

From this directory, create the environment and run a prediction:

```bash
conda env create --prefix .conda/repro --file environment.yml
conda run --prefix .conda/repro python predict.py --problem poisson
```

The default input is that problem's `training_data.npz`, and the default run
predicts the first branch row. Use `--count N` for the first `N` rows or
`--count 0` for all rows. Predictions are saved to
`predictions/<problem>.npz`; `--output path.npz` changes that path.

The script also accepts a custom NPZ file through `--input path.npz`. It must
contain `branch` with shape `(N, 101)` and `trunk`. For the four scalar
problems, `trunk` can be shared coordinates `(M, 2)` or per-sample
coordinates `(N, M, 2)`. For `cavity_flow`, use paired coordinates `(N, 2)`
or per-sample coordinates `(N, M, 2)`. For example:

```bash
conda run --prefix .conda/repro python predict.py \
  --problem advection --input inputs.npz --count 0 --output advection_output.npz
```

The output file contains `prediction` with shape `(N, M)` for a scalar
problem, or `velocity_u` and `velocity_v` with shape `(N, M)` for
`cavity_flow`. The default `cavity_flow` training input uses one coordinate
per branch row, so its output has shape `(N, 1)`.

## Model definition

`predict.py` rebuilds the original networks before restoring the TensorFlow
v1 checkpoints. The environment pins Python 3.10, DeepXDE 1.9.1,
TensorFlow CPU 2.12.0, NumPy 1.23.5, and SciPy 1.10.1. DeepXDE uses the
`tensorflow.compat.v1` backend.

All models use branch widths `[101, 100, 100, 100]`, trunk widths
`[2, 100, 100, 100]`, branch ReLU, trunk GELU, and Glorot normal
initialization. The GELU function is
`0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z**3)))`. Models are compiled with
Adam at learning rate `0.001` before checkpoint restoration.

| Problem | Additional network transform |
| --- | --- |
| `diffusion_reaction` | Output: `10*x*(1-x)*t*(z+1)` |
| `advection` | Output: `4*x*t*z + sin(pi*x) + sin(pi*t/2)` |
| `burgers` | Map trunk `(x,t)` to `cos(2*pi*x), sin(2*pi*x), ..., cos(8*pi*x), sin(8*pi*x), t`; L2 regularization coefficient `1e-5` |
| `poisson`, `cavity_flow` | No additional transform |

Here `z` is the raw network output. For the first two rows, `x` and `t`
are the two trunk input columns.
