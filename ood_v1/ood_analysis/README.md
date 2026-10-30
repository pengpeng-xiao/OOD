# ID / OOD analysis of `anonymous_pde_benchmark` test sets

## Result

| Problem | ID | OOD | Separating feature |
|---|---|---|---|
| diffusion_reaction | 100 | 100 | input GRF length scale: train l=0.5, OOD l≈0.2 |
| burgers | 100 | 100 | periodic GRF: train l=1.0, OOD l≈0.6 |
| advection | 100 | 400 | train l=0.5; OOD = four regimes l≈0.05, 0.10, 0.15, 0.20 (100 each) |
| poisson | 100 | 100 | train l=0.5 (KL-truncated at 11 modes), OOD l≈0.2 (not truncated) |
| cavity_flow | 20 | 10 | branch = U·(1−x); train U∈[0,0.4]; ID U=0.01..0.39, OOD U=0.41..0.50 |

Per-sample labels: `results/ood_labels.csv`. Figure: `results/ood_summary.png`.

## Method

1. Pooled ML fit of a GRF model to the training branches (`gp_pooled.py`, `spec.py`).
2. Per-sample profile-likelihood fit of the length scale; test statistic is
   2·log LR of the best-fit vs training kernel (`gp_mle2.py`; Poisson uses the
   leading-11-mode subspace, `gp_poisson2.py`). OOD if chi²₁ p < 1e-4.
3. Cross-checks: model-free Mahalanobis distance in training-PCA space with a
   10-fold CV null (`crosscheck.py`, `crosscheck_burgers.py`), Poisson tail
   energy, and DeepONet relative-L2 error of the released checkpoints
   (`run_models.py`, `errs.py`).
4. `classify.py` writes the labels; `figure.py` draws the summary figure.

`gp_mle.py` and `gp_poisson.py` are earlier attempts with a misspecified nugget,
kept for the record; their outputs are not used.

## Re-running

Run the scripts from inside `analysis/` (`gp_poisson*.py` read `gp_mle2.py` by
relative path). All scripts except `run_models.py` and `figure.py` need only
numpy/scipy/scikit-learn. `run_models.py` needs TensorFlow 2.12 + DeepXDE 1.9.1
(the benchmark's `environment.yml`):

```bash
python run_models.py ../preds      # DeepONet predictions -> preds/
python gp_mle2.py; python gp_poisson2.py   # length-scale features -> feat/
python classify.py                 # -> results/ood_labels.csv
python crosscheck.py; python crosscheck_burgers.py
python figure.py                   # -> results/ood_summary.png (needs matplotlib)
```
