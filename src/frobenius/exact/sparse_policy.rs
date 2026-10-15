//! Opt-in orchestration of Symbolica's existing sparse row reducer.
use super::*;
use symbolica::tensors::sparse::{LuLMode, SparseRowReducer};

impl ExactDomain<'_> {
    fn observe_reducer(&self, reducer: &SparseRowReducer<FloatField<Gaussian>>) -> Result<()> {
        self.context.cancellation.check()?;
        if reducer.u().nvalues() > self.limits.max_scalar_cells {
            return Err(Error::Limit(
                "exact sparse elimination stored cells exceed limit".into(),
            ));
        }
        for value in reducer.u().values() {
            self.check_value(value)?;
        }
        Ok(())
    }

    pub(super) fn sparse_reduced_rows(
        &self,
        rows: Vec<Vec<Gaussian>>,
    ) -> Result<Vec<Vec<Gaussian>>> {
        self.limits.validate()?;
        self.context.cancellation.check()?;
        let columns = rows.first().map_or(0, Vec::len);
        if columns == 0 || rows.iter().any(|row| row.len() != columns) {
            return Err(Error::InvalidInput(
                "exact sparse elimination dimensions".into(),
            ));
        }
        self.preflight_matrix(rows.len().max(columns))?;
        self.check_rows(&rows)?;
        let matrix = ExactMatrix::from_nested_vec(rows, field())
            .map_err(Error::Numerical)?
            .to_sparse();
        let mut reducer = SparseRowReducer::new(columns as u32, field(), LuLMode::None);
        for (_, indices, values) in matrix.row_iter() {
            self.context.cancellation.check()?;
            reducer.add_row(values, indices);
            // Native row insertion is atomic here: observations are at each
            // completed row boundary, not every internal rational operation.
            self.observe_reducer(&reducer)?;
        }
        // Do not use the checked constructor: consistency is established from
        // exact RHS pivots by each caller, including nonzero RHS-only rows.
        reducer.back_substitute();
        self.observe_reducer(&reducer)?;
        let mut reduced = reducer.u().clone();
        reduced.sort_rows_by_pivot(reducer.pivots());
        let rows = reduced
            .to_dense()
            .into_vec()
            .chunks(columns)
            .map(<[Gaussian]>::to_vec)
            .collect::<Vec<Vec<Gaussian>>>();
        self.check_rows(&rows)?;
        Ok(rows)
    }

    pub(super) fn validate_sparse_blocks(
        &self,
        matrix: &[Vec<Gaussian>],
        blocks: &[Vec<usize>],
    ) -> Result<()> {
        let n = matrix.len();
        if n == 0 || matrix.iter().any(|row| row.len() != n) {
            return Err(Error::InvalidInput(
                "exact sparse recurrence dimensions".into(),
            ));
        }
        let mut owners = vec![usize::MAX; n];
        for (block, indices) in blocks.iter().enumerate() {
            if indices.is_empty() {
                return Err(Error::InvalidInput("empty exact recurrence block".into()));
            }
            for &index in indices {
                if index >= n || owners[index] != usize::MAX {
                    return Err(Error::InvalidInput(
                        "invalid exact recurrence block partition".into(),
                    ));
                }
                owners[index] = block;
            }
        }
        if owners.contains(&usize::MAX) {
            return Err(Error::InvalidInput(
                "incomplete exact recurrence block partition".into(),
            ));
        }
        for (i, row) in matrix.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                if owners[j] > owners[i] && !value.is_zero() {
                    return Err(Error::InvalidInput(
                        "invalid exact recurrence block order".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn sparse_inverse(
        &self,
        rows: Vec<Vec<Gaussian>>,
        context: &RunContext,
    ) -> Result<ExactMatrix> {
        self.limits.validate()?;
        context.cancellation.check()?;
        let n = rows.len();
        if n == 0 || rows.iter().any(|row| row.len() != n) {
            return Err(Error::InvalidInput(
                "exact sparse inverse dimensions".into(),
            ));
        }
        // The native inverse augments by an identity. Retain the same scalar
        // allocation admission as the row-observed augmented-matrix path.
        self.preflight_matrix(
            n.checked_mul(2)
                .ok_or_else(|| Error::Limit("exact sparse inverse dimension overflow".into()))?,
        )?;
        self.check_rows(&rows)?;
        let matrix = ExactMatrix::from_nested_vec(rows, field()).map_err(Error::Numerical)?;
        // Reuse the existing owner inverse. Its internal elimination arithmetic
        // is not exposed to this caller: only inputs and outputs are observed.
        let inverse = matrix
            .to_sparse()
            .inv()
            .map_err(|e| Error::Numerical(format!("exact sparse recurrence inverse: {e}")))?
            .to_dense();
        context.cancellation.check()?;
        let inverse_rows = inverse
            .clone()
            .into_vec()
            .chunks(n)
            .map(<[Gaussian]>::to_vec)
            .collect::<Vec<_>>();
        self.check_rows(&inverse_rows)?;
        if &matrix * &inverse != ExactMatrix::identity(n as u32, field()) {
            return Err(Error::Numerical(
                "exact sparse recurrence inverse identity failed".into(),
            ));
        }
        context.cancellation.check()?;
        Ok(inverse)
    }

    pub(super) fn sparse_solve_any(
        &self,
        rows: Vec<Vec<Gaussian>>,
        rhs: Vec<Gaussian>,
        context: &RunContext,
    ) -> Result<Vec<Gaussian>> {
        let n = rows.first().map_or(0, Vec::len);
        if n == 0 || rows.len() != rhs.len() || rows.iter().any(|row| row.len() != n) {
            return Err(Error::InvalidInput(
                "exact sparse resonant dimensions".into(),
            ));
        }
        let matrix =
            ExactMatrix::from_nested_vec(rows.clone(), field()).map_err(Error::Numerical)?;
        let rhs_matrix =
            ExactMatrix::from_nested_vec(rhs.iter().cloned().map(|v| vec![v]).collect(), field())
                .map_err(Error::Numerical)?;
        let augmented = rows
            .into_iter()
            .zip(rhs)
            .map(|(mut row, b)| {
                row.push(b);
                row
            })
            .collect();
        let reduced = self.sparse_reduced_rows(augmented)?;
        if reduced
            .iter()
            .any(|row| row[..n].iter().all(Gaussian::is_zero) && !row[n].is_zero())
        {
            return Err(Error::Numerical(
                "inconsistent exact sparse resonant recurrence".into(),
            ));
        }
        let result = if reduced.is_empty() {
            ExactMatrix::new(n as u32, 1, field())
        } else {
            let native =
                ExactMatrix::from_nested_vec(reduced, field()).map_err(Error::Numerical)?;
            let (a, b) = native
                .split_col(n as u32)
                .map_err(|e| Error::Numerical(e.to_string()))?;
            // Native solve_any chooses a representative of the reduced affine
            // solution set; no new elimination or nullspace algorithm here.
            a.solve_any(&b)
                .map_err(|e| Error::Numerical(format!("exact sparse resonant solve: {e}")))?
        };
        if &matrix * &result != rhs_matrix {
            return Err(Error::Numerical(
                "exact sparse resonant residual failed".into(),
            ));
        }
        context.cancellation.check()?;
        let result = result.into_vec();
        self.check_rows(std::slice::from_ref(&result))?;
        Ok(result)
    }
}
