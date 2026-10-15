#![allow(clippy::needless_range_loop)] // Matrix row/column indexing mirrors the recurrence equations.
//! Exact linear algebra for indicial equations and rational basis changes.
use crate::{Error, Result};
use symbolica::prelude::*;

fn exact_atom_field() -> symbolica::domains::atom::AtomField {
    symbolica::domains::atom::AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: Some(Box::new(|a: AtomView<'_>, out: &mut Atom| {
            *out = a.together().cancel();
            true
        })),
    }
}

pub(crate) fn rref(a: Vec<Vec<Atom>>) -> (Vec<Vec<Atom>>, Vec<usize>) {
    if a.is_empty() || a[0].is_empty() {
        return (a, vec![]);
    }
    let columns = a[0].len();
    let mut matrix = symbolica::tensors::matrix::Matrix::from_nested_vec(a, exact_atom_field())
        .expect("internal rectangular matrix");
    matrix.row_reduce(columns as u32);
    let rows = matrix
        .into_vec()
        .chunks(columns)
        .map(|r| r.to_vec())
        .collect::<Vec<_>>();
    let pivots = rows
        .iter()
        .filter_map(|row| row.iter().position(|v| !v.is_zero()))
        .collect();
    (rows, pivots)
}

pub(crate) fn nullspace(a: Vec<Vec<Atom>>) -> Vec<Vec<Atom>> {
    if a.is_empty() {
        return Vec::new();
    }
    let cols = a[0].len();
    let (a, pivots) = rref(a);
    (0..cols)
        .filter(|j| !pivots.contains(j))
        .map(|j| {
            let mut v = vec![Atom::new(); cols];
            v[j] = Atom::num(1);
            for (i, &pivot) in pivots.iter().enumerate() {
                v[pivot] = -a[i][j].clone();
            }
            v
        })
        .collect()
}

/// Representation choice only: structural exact zeros never use a tolerance.
fn structurally_sparse(matrix: &[Vec<Atom>]) -> bool {
    let entries = matrix.iter().map(Vec::len).sum::<usize>();
    let nonzero = matrix.iter().flatten().filter(|a| !a.is_zero()).count();
    nonzero <= entries / 4
}

pub(crate) fn matmul(a: &[Vec<Atom>], b: &[Vec<Atom>]) -> Vec<Vec<Atom>> {
    if a.is_empty() {
        return Vec::new();
    }
    let columns = b[0].len();
    if columns == 0 {
        return vec![Vec::new(); a.len()];
    }
    let left = symbolica::tensors::matrix::Matrix::from_nested_vec(a.to_vec(), exact_atom_field())
        .expect("internal rectangular matrix");
    let right = symbolica::tensors::matrix::Matrix::from_nested_vec(b.to_vec(), exact_atom_field())
        .expect("internal rectangular matrix");
    let product = if structurally_sparse(a) || structurally_sparse(b) {
        // Reuse Symbolica's exact sparse multiplication when at least one
        // input has at most one quarter structurally nonzero entries.
        (&left.to_sparse() * &right.to_sparse()).to_dense()
    } else {
        &left * &right
    };
    product
        .into_vec()
        .chunks(columns)
        .map(|row| row.iter().map(|a| a.together().cancel()).collect())
        .collect()
}

pub(crate) fn determinant(a: Vec<Vec<Atom>>) -> Atom {
    if a.is_empty() {
        return Atom::num(1);
    }
    symbolica::tensors::matrix::Matrix::from_nested_vec(a, exact_atom_field())
        .expect("internal square matrix")
        .det()
        .expect("internal square matrix")
        .together()
        .cancel()
}

pub(crate) fn inverse(a: &[Vec<Atom>]) -> Result<Vec<Vec<Atom>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) {
        return Err(Error::InvalidInput(
            "inverse needs a nonempty square matrix".into(),
        ));
    }
    let matrix =
        symbolica::tensors::matrix::Matrix::from_nested_vec(a.to_vec(), exact_atom_field())
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
    let inverse = matrix
        .inv()
        .map_err(|e| Error::Numerical(format!("rational matrix inverse: {e}")))?;
    Ok(inverse.into_vec().chunks(n).map(|r| r.to_vec()).collect())
}

/// Rational eigenvalues with algebraic multiplicities, computed exactly.
pub(crate) fn eigenvalues(residue: &[Vec<Atom>]) -> Result<Vec<(Atom, usize)>> {
    let lambda = symbol!("symbolica_amflow::indicial_lambda");
    let x = Atom::var(lambda);
    let mut matrix = residue
        .iter()
        .map(|r| r.iter().map(|a| -a.clone()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = &row[i] + &x;
    }
    let characteristic = determinant(matrix).factor();
    let factors = if let AtomView::Mul(m) = characteristic.as_view() {
        m.iter().map(|a| a.to_owned()).collect::<Vec<_>>()
    } else {
        vec![characteristic]
    };
    let mut result = Vec::new();
    for factor in factors {
        let (factor, multiplicity) = if let AtomView::Pow(v) = factor.as_view() {
            let (base, power) = v.get_base_exp();
            (
                base.to_owned(),
                power
                    .to_string()
                    .parse::<i64>()
                    .map_err(|_| Error::Unsupported("noninteger characteristic factor".into()))?,
            )
        } else {
            (factor, 1)
        };
        let mut a = Atom::new();
        let mut b = Atom::new();
        let mut has_variable = false;
        for (monomial, c) in
            crate::coefficient::exact_coefficient_list(&factor, std::slice::from_ref(&x))?
        {
            if monomial.is_one() {
                b += c;
            } else if monomial == x {
                a += c;
                has_variable = true;
            } else {
                return Err(Error::Unsupported(
                    "indicial polynomial has non-rational eigenvalues".into(),
                ));
            }
        }
        if has_variable {
            if multiplicity <= 0 {
                return Err(Error::Numerical(
                    "invalid characteristic multiplicity".into(),
                ));
            }
            result.push(((-b / a).together().cancel(), multiplicity as usize));
        }
    }
    if result.iter().map(|(_, m)| *m).sum::<usize>() != residue.len() {
        return Err(Error::Numerical("incomplete indicial factorization".into()));
    }
    Ok(result)
}

#[cfg(test)]
#[path = "algebra_matmul_tests.rs"]
mod representation_tests;
