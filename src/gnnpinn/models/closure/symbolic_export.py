"""SymPy-based symbolic export for sparse linear libraries."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp


def export_linear_library_expression(
    term_names: Sequence[str],
    coefficients: Sequence[float],
    threshold: float = 0.0,
) -> sp.Expr:
    """Create a SymPy expression from linear library terms and coefficients."""

    if len(term_names) != len(coefficients):
        raise ValueError("term_names and coefficients must have the same length")
    expr = sp.Integer(0)
    for name, coefficient in zip(term_names, coefficients):
        coefficient = float(coefficient)
        if abs(coefficient) <= threshold:
            continue
        expr += coefficient * _parse_term(name)
    return sp.simplify(expr)


def expression_to_string(expr: sp.Expr) -> str:
    return str(sp.simplify(expr))


def _parse_term(name: str) -> sp.Expr:
    if name == "1":
        return sp.Integer(1)
    expr = sp.Integer(1)
    for piece in name.split("*"):
        if "^" in piece:
            symbol, power = piece.split("^", 1)
            expr *= sp.Symbol(symbol) ** int(power)
        else:
            expr *= sp.Symbol(piece)
    return expr

