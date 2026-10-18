//! Bounded retention for the existing numerical Frobenius recurrence.
use super::*;
use std::cell::Cell;

pub(super) struct StreamingAdmission {
    p: Precision,
    order: usize,
    source_bytes: u128,
    column_bytes: Cell<u128>,
    identity_transformation: bool,
}

impl StreamingAdmission {
    pub(super) fn new(
        p: Precision,
        order: usize,
        source_values: u128,
        identity_transformation: bool,
    ) -> Self {
        Self {
            p,
            order,
            source_bytes: source_values.saturating_mul(crate::series_limits::complex_bytes(p)),
            column_bytes: Cell::new(0),
            identity_transformation,
        }
    }

    pub(super) fn estimated_column_bytes(
        p: Precision,
        dimension: usize,
        order: usize,
        logs: usize,
        identity_transformation: bool,
    ) -> u128 {
        let n = dimension as u128;
        // Only a nonidentity transformation needs a replacement coefficient
        // buffer. Retain the same conservative normalization-span padding and
        // workspace estimate on both branches. No order-long solver cache is
        // retained on this path.
        let series = (order as u128 + 257)
            .saturating_mul(n)
            .saturating_mul(logs as u128)
            .saturating_mul(if identity_transformation { 1 } else { 2 });
        let workspace = n
            .saturating_mul(n)
            .saturating_mul(8)
            .saturating_add(n.saturating_mul(n).saturating_mul(257));
        series
            // Contraction evaluates one short leading log chain before moving
            // it into the long column; count that temporary before allocation.
            .saturating_add(n.saturating_mul(logs as u128))
            .saturating_add(workspace)
            .saturating_mul(crate::series_limits::complex_bytes(p))
    }

    pub(super) fn column(&self, dimension: usize, order: usize, logs: usize) -> Result<()> {
        let bytes = Self::estimated_column_bytes(
            self.p,
            dimension,
            order,
            logs,
            self.identity_transformation,
        );
        crate::series_limits::admit_bytes(
            order,
            self.source_bytes.saturating_add(bytes),
            "streamed Frobenius column",
        )?;
        self.column_bytes.set(bytes);
        Ok(())
    }

    pub(super) fn matrix(&self, dimension: usize) -> Result<()> {
        let transient = (dimension as u128)
            .saturating_pow(2)
            .saturating_mul(8)
            .saturating_mul(crate::series_limits::complex_bytes(self.p));
        crate::series_limits::admit_bytes(
            self.order,
            self.source_bytes
                .saturating_add(self.column_bytes.get())
                .saturating_add(transient),
            "streamed Frobenius resonant workspace",
        )
    }
}

impl PreparedFrobenius {
    /// Evaluate one solution while retaining only one long Frobenius column.
    ///
    /// `constants` use the same column order as [`Self::evaluate`], so a short
    /// basis can determine them with existing asymptotic matching. This reuses
    /// the same recurrence and basis transformation, disables order-long solver
    /// retention, and releases each column after accumulating its contribution.
    /// Equal exact exponents without any positive-order resonance share one
    /// combined recurrence; resonant groups retain their original columns.
    /// Every nonzero logarithm row is retained in a combined recurrence.
    ///
    /// The return value is a truncated expansion, not an accuracy certificate.
    /// Matching uncertainty, truncation, cancellation and branch choices still
    /// require independent validation. Storage estimates are admission checks,
    /// not allocator quotas; external process memory limits remain necessary.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_solution(
        &self,
        p: Precision,
        parameters: &ahash::HashMap<Atom, C>,
        order: usize,
        constants: &[C],
        point: &C,
        context: &RunContext,
    ) -> Result<Vec<C>> {
        self.evaluate_solution_with_winding(p, parameters, order, constants, point, 0, context)
    }

    /// As [`Self::evaluate_solution`], on the branch
    /// `log(point) + 2*pi*i*winding` for both powers and logarithms.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_solution_with_winding(
        &self,
        p: Precision,
        parameters: &ahash::HashMap<Atom, C>,
        order: usize,
        constants: &[C],
        point: &C,
        winding: i32,
        context: &RunContext,
    ) -> Result<Vec<C>> {
        let n = self.source.matrix.len();
        if constants.len() != n || constants.iter().any(|value| !p.finite(value)) {
            return Err(Error::InvalidInput(
                "streamed Frobenius constants dimension or finiteness".into(),
            ));
        }
        let mut result = Vec::new();
        self.visit_values(
            p,
            parameters,
            order,
            point,
            winding,
            context,
            Some(constants),
            |indices, values| {
                if result.is_empty() {
                    result.resize(n, p.zero());
                }
                for (target, value) in result.iter_mut().zip(values) {
                    let contribution = if indices.len() == 1 {
                        p.mul(&constants[indices.start], &value)
                    } else {
                        value
                    };
                    *target = p.add(target, &contribution);
                }
                Ok(())
            },
        )?;
        if result.iter().any(|value| !p.finite(value)) {
            return Err(Error::Numerical(
                "nonfinite streamed Frobenius value".into(),
            ));
        }
        Ok(result)
    }

    /// Match a finite-point vector using streamed long columns, retaining only
    /// the small matrix of their evaluated values. This uses the same generic
    /// parameter policy as [`Self::evaluate`] and the existing numerical solve.
    /// As with full-basis matching, a successful solve does not certify the
    /// input boundary precision or expansion truncation.
    #[allow(clippy::too_many_arguments)]
    pub fn match_values(
        &self,
        p: Precision,
        parameters: &ahash::HashMap<Atom, C>,
        order: usize,
        point: &C,
        values: &[C],
        context: &RunContext,
    ) -> Result<Vec<C>> {
        self.match_values_with_winding(p, parameters, order, point, values, 0, context)
    }

    /// As [`Self::match_values`], on the explicitly supplied local winding.
    #[allow(clippy::too_many_arguments)]
    pub fn match_values_with_winding(
        &self,
        p: Precision,
        parameters: &ahash::HashMap<Atom, C>,
        order: usize,
        point: &C,
        values: &[C],
        winding: i32,
        context: &RunContext,
    ) -> Result<Vec<C>> {
        let n = self.source.matrix.len();
        if values.len() != n || values.iter().any(|value| !p.finite(value)) {
            return Err(Error::InvalidInput(
                "streamed Frobenius matching dimension or finiteness".into(),
            ));
        }
        let mut matrix: Vec<Vec<C>> = Vec::new();
        self.visit_values(
            p,
            parameters,
            order,
            point,
            winding,
            context,
            None,
            |indices, values| {
                if indices.len() != 1 {
                    return Err(Error::Numerical("contracted matching column".into()));
                }
                if matrix.is_empty() {
                    matrix = vec![vec![p.zero(); n]; n];
                }
                for (row, value) in matrix.iter_mut().zip(values) {
                    row[indices.start] = value;
                }
                Ok(())
            },
        )?;
        solve(p, matrix, values.to_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_values(
        &self,
        p: Precision,
        parameters: &ahash::HashMap<Atom, C>,
        order: usize,
        point: &C,
        winding: i32,
        context: &RunContext,
        constants: Option<&[C]>,
        mut visitor: impl FnMut(std::ops::Range<usize>, Vec<C>) -> Result<()>,
    ) -> Result<()> {
        let n = self.source.matrix.len();
        let (prepared_rows, source_values) =
            self.numerical_rows(p, parameters, order, context, false, false)?;
        let log = evaluate::logarithm(p, point, winding)?;
        let admission =
            StreamingAdmission::new(p, order, source_values, self.identity_transformation());
        let mut index = 0;
        self.numerical_recurrence_visit(
            p,
            parameters,
            order,
            context,
            &prepared_rows,
            &admission,
            constants,
            |indices, column| {
                context.cancellation.check()?;
                if column
                    .coefficients
                    .iter()
                    .flatten()
                    .flatten()
                    .any(|value| !p.finite(value))
                {
                    return Err(Error::Numerical("nonfinite Frobenius coefficient".into()));
                }
                if indices.start != index || indices.end > n || indices.is_empty() {
                    return Err(Error::Numerical(
                        "invalid streamed Frobenius column range".into(),
                    ));
                }
                let values = evaluate::column_value(p, point, &log, parameters, &column)?;
                if values.iter().any(|value| !p.finite(value)) {
                    return Err(Error::Numerical(
                        "nonfinite streamed Frobenius column value".into(),
                    ));
                }
                index = indices.end;
                visitor(indices, values)?;
                Ok(())
            },
        )?;
        if index != n {
            return Err(Error::Numerical(
                "incomplete streamed Frobenius values".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_admission_counts_log_growth_source_and_resonant_workspace() -> Result<()> {
        let p = Precision::decimal(24_756)?;
        let admission = StreamingAdmission::new(p, 30_000, 1_000, false);
        admission.column(18, 30_000, 1)?;
        admission.matrix(36)?;
        assert!(matches!(
            admission.column(18, 30_000, 2),
            Err(Error::Limit(_))
        ));
        assert!(matches!(admission.matrix(10_000), Err(Error::Limit(_))));
        let identity = StreamingAdmission::new(p, 30_000, 1_000, true);
        identity.column(18, 30_000, 2)?;
        assert!(matches!(
            identity.column(18, 30_000, 3),
            Err(Error::Limit(_))
        ));
        let oversized_source = StreamingAdmission::new(p, 30_000, u128::MAX, true);
        assert!(matches!(
            oversized_source.column(18, 30_000, 1),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            admission.column(18, 200_001, 1),
            Err(Error::Limit(_))
        ));
        Ok(())
    }
}

#[cfg(test)]
#[path = "contracted_solution_tests.rs"]
mod contracted_solution_tests;
