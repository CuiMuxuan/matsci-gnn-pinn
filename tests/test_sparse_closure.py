from __future__ import annotations

import subprocess
import sys

import pytest

from gnnpinn.models.closure.symbolic_export import (
    expression_to_string,
    export_linear_library_expression,
)


def _torch_available() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


torchmark = pytest.mark.skipif(
    not _torch_available(),
    reason="torch is not importable in the current environment",
)


@torchmark
def test_sparse_library_polynomial_terms():
    import torch

    from gnnpinn.models.closure import SparseLibrary, SparseLibraryConfig

    library = SparseLibrary(
        SparseLibraryConfig(
            feature_names=("T", "grad_T"),
            polynomial_order=2,
            include_bias=True,
        )
    )
    features = torch.tensor([[2.0, 3.0]])
    matrix = library.transform(features)

    assert library.term_names == ["1", "T", "grad_T", "T^2", "T*grad_T", "grad_T^2"]
    assert matrix.tolist()[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])


@torchmark
def test_l1_sparsity_loss():
    import torch

    from gnnpinn.models.closure import l1_sparsity

    coeffs = torch.tensor([1.0, -2.0, 0.5])

    assert float(l1_sparsity(coeffs, weight=0.1)) == pytest.approx(0.35)


def test_symbolic_export_thresholds_terms():
    expr = export_linear_library_expression(
        term_names=["1", "T", "grad_T", "T*grad_T"],
        coefficients=[1.0, 0.05, -2.0, 0.5],
        threshold=0.1,
    )

    assert expression_to_string(expr) == "0.5*T*grad_T - 2.0*grad_T + 1.0"

