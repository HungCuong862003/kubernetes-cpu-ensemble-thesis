# Results

Experimental outputs organised as **dataset x horizon x artefact-type**.

```
results/
  <dataset>/
    h010/  h030/  h060/  h120/
      predictions/        OOF predictions per base model (.npy)
      intervals/          CQR prediction intervals (.npy)
      foundation_models/  Chronos-Bolt, TimesFM, Granite TTM (.npz)
      meta_learner/       NNLS weights + ensemble predictions
      diagnostics/        Per-container R2, residuals, UQ recalibration
      shap/               SHAP values + summary
      metrics.json        Single source of truth for MAE/RMSE/sMAPE/R2
    cross_horizon/        Friedman, DM-pairwise, CD diagrams
  hpa_simulation/         Cross-dataset HPA Pareto frontier (v2 grid)
  bcf/                    Boundary Condition Framework, cross-dataset
```

Filename convention: `<artefact>_<model>.<ext>`, e.g. `oof_xgb.npy`,
`cqr_ensemble.npy`, `nnls_weights_hetero.json`.

Horizon tokens are zero-padded (`h010`, `h030`, `h060`, `h120`) so
lexicographic sort equals numerical sort.
