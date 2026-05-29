# GNN-PINN Multiscale Materials Coupling Blueprint

This is an English companion summary. The Chinese documents are the authoritative project skeleton for now.

## Core Decision

Direction 3, a dynamic multiscale GNN-PINN coupling framework, does not strictly require Direction 1 as a prerequisite. However, Direction 1 is the lowest-risk entry point for a real code project because it produces a reusable differentiable material closure module before the full macro-meso feedback loop is attempted.

The recommended path is:

```text
PINN + sparse/learnable closure
  -> GNN-conditioned microstructure-aware closure
  -> weak two-way macro PINN <-> meso GNN coupling
  -> full coupled framework and paper experiments
```

## Main Dataset Direction

After comparing several public data directions, the mainline should be metal additive manufacturing using NIST AM-Bench data, especially nickel-based alloy benchmark cases.

Reasons:

- It is a public benchmark ecosystem rather than a single isolated dataset.
- It is aligned with process-structure-property modeling.
- It supports macro field prediction, microstructure modeling, and performance validation.
- It can be supplemented with open simulation tools such as ExaCA, while keeping real public data as the primary evidence.

Fallback directions:

- Lithium-ion battery electrode microstructures and degradation.
- Digital porous media microstructure and transport.
- PFHub phase-field benchmarks.
- Materials Data Facility XCT datasets and crystal-property databases as auxiliary sources.

## Technical Stack

Main stack:

- PyTorch.
- PyTorch Geometric.
- Custom modular PINN kernel.
- Sparse/symbolic equation discovery using PyTorch, PySINDy-style sparse regression, and SymPy export.

Optional branches:

- DeepXDE for baseline comparison.
- NVIDIA Modulus for later large-scale or 3D acceleration.

## Future Repository Shape

The future repository should include:

```text
configs/
  data/
  model/
  experiment/
src/gnnpinn/
  data/
  physics/
  models/
    pinn/
    gnn/
    closure/
    coarse_grain/
    coupled/
  losses/
  train/
  eval/
  viz/
scripts/
  data/
  train/
  eval/
outputs/
  runs/
  figures/
  tables/
  equations/
tests/
```

## Planned Experiments

1. Data audit and AM-Bench minimal subset.
2. Data-driven baseline.
3. Macro PINN baseline.
4. Sparse closure discovery.
5. GNN-conditioned closure.
6. Weak two-way coupling.
7. Full coupled framework and ablations.
8. Fallback transfer experiments on battery, porous media, or PFHub data.

## Key Metrics

- Temperature RMSE/MAE.
- PDE residual.
- Melt pool geometry IoU and width/depth error.
- Microstructure distribution distance.
- Sparse closure active term count and coefficient stability.
- OOD process generalization.
- Improvement over pure PINN, pure GNN, and one-way coupling.

## Key Resources

- NIST AM-Bench: https://www.nist.gov/ambench
- AM-Bench benchmark test data: https://www.nist.gov/ambench/benchmark-test-data
- ExaCA: https://github.com/LLNL/ExaCA
- PFHub benchmarks: https://pages.nist.gov/pfhub/benchmarks/
- Digital Porous Media Portal: https://digitalporousmedia.org/
- NREL battery microstructure library: https://www.nrel.gov/transportation/battery-microstructure-library-data
- Materials Data Facility: https://www.materialsdatafacility.org/

