//! Verified rectangular state embeddings for finite-point transport.
//!
//! A supplied map `old_y = Q * selected_y` is checked exactly; this module does
//! not discover an IBP basis or establish that a particular physical boundary
//! lies in the subspace. The latter is checked numerically before projection.
use super::*;
use std::collections::BTreeSet;
use symbolica::tensors::matrix::Matrix;

/// An exact, verified embedding into a system whose variables are selected old
/// components. Fields are private so a compiled map cannot diverge from its ODE.
#[derive(Clone, Debug)]
pub struct StateCompression {
    reduced: DifferentialSystem,
    selected: Vec<usize>,
    conditions: Vec<Atom>,
    prepared_reduced: prepared::PreparedRows,
    prepared_embedding: prepared::PreparedRows,
}

/// One fresh numerical specialization of a verified state compression.
///
/// Projection and reconstruction apply at ordinary finite points, not at a
/// singular physical limit. Keep the original singular-endpoint machinery.
#[derive(Clone, Debug)]
pub struct CompiledStateCompression {
    reduced: CompiledSystem,
    embedding: CompiledSystem,
    selected: Vec<usize>,
    conditions: Vec<Atom>,
    values: ahash::HashMap<Atom, C>,
    variable: Symbol,
    p: Precision,
}

impl DifferentialSystem {
    /// Verify a rectangular embedding and derive the selected-coordinate ODE.
    ///
    /// Requires `Q[selected] = I` and `dQ/dx + Q * A_reduced = A_old * Q`.
    /// `A_reduced` is the selected rows of `A_old * Q`; there is no numerical
    /// rank decision or matrix inversion. This checks differential compatibility,
    /// not the physical origin of a caller-supplied reduction.
    ///
    /// Original matrix/map denominators and all declared conditions survive
    /// cancellation. Supply assumptions erased before constructing these Atoms
    /// explicitly: no API can recover discarded factors.
    pub fn compress_states(
        &self,
        embedding: &[Vec<Atom>],
        selected_old_rows: &[usize],
        declared_nonzero_conditions: &[Atom],
        context: &RunContext,
    ) -> Result<StateCompression> {
        context.cancellation.check()?;
        self.validate()?;
        let old_size = self.matrix.len();
        let size = selected_old_rows.len();
        if size == 0
            || size > old_size
            || embedding.len() != old_size
            || embedding.iter().any(|row| row.len() != size)
            || selected_old_rows.iter().any(|&i| i >= old_size)
            || selected_old_rows
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != size
        {
            return Err(Error::InvalidInput(
                "state compression dimensions or selector".into(),
            ));
        }
        for (i, &selected) in selected_old_rows.iter().enumerate() {
            context.cancellation.check()?;
            for (j, entry) in embedding[selected].iter().enumerate() {
                if !(entry - Atom::num(i64::from(i == j)))
                    .together()
                    .cancel()
                    .is_zero()
                {
                    return Err(Error::InvalidInput(
                        "state compression selected rows are not identity".into(),
                    ));
                }
            }
        }
        let expressions = self
            .matrix
            .iter()
            .chain(embedding)
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let mut symbols = BTreeSet::new();
        for expression in expressions.iter().chain(declared_nonzero_conditions) {
            context.cancellation.check()?;
            crate::family::scalar_symbols(expression.as_view(), &mut symbols)?;
        }
        let mut variables = symbols
            .iter()
            .map(|symbol| {
                let AtomView::Var(v) = symbol.as_view() else {
                    unreachable!("native scalar symbol walker")
                };
                v.get_symbol()
            })
            .collect::<BTreeSet<_>>();
        variables.insert(self.variable);
        let mut conditions = crate::physical_conditions::canonical_conditions(
            declared_nonzero_conditions,
            &variables,
        )?;
        conditions.extend(crate::physical_conditions::rational_denominator_conditions(
            &expressions,
            &variables,
        )?);
        context.cancellation.check()?;
        let product = crate::algebra::matmul(&self.matrix, embedding);
        let reduced_rows = selected_old_rows
            .iter()
            .map(|&i| product[i].clone())
            .collect::<Vec<_>>();
        let lifted = crate::algebra::matmul(embedding, &reduced_rows);
        for i in 0..old_size {
            context.cancellation.check()?;
            for j in 0..size {
                if !(embedding[i][j].derivative(self.variable) + &lifted[i][j] - &product[i][j])
                    .together()
                    .cancel()
                    .is_zero()
                {
                    return Err(Error::InvalidInput(format!(
                        "state compression differential compatibility row {i} column {j}"
                    )));
                }
            }
        }
        conditions.extend(crate::physical_conditions::rational_denominator_conditions(
            &reduced_rows.iter().flatten().cloned().collect::<Vec<_>>(),
            &variables,
        )?);
        let conditions = crate::physical_conditions::canonical_conditions(&conditions, &variables)?;
        let prepared_reduced = prepared::PreparedRows::new(self.variable, &reduced_rows, context)?;
        let prepared_embedding = prepared::PreparedRows::new(self.variable, embedding, context)?;
        context.cancellation.check()?;
        Ok(StateCompression {
            reduced: DifferentialSystem {
                variable: self.variable,
                matrix: reduced_rows,
            },
            selected: selected_old_rows.to_vec(),
            conditions,
            prepared_reduced,
            prepared_embedding,
        })
    }
}

impl StateCompression {
    pub fn reduced_system(&self) -> &DifferentialSystem {
        &self.reduced
    }
    pub fn selected_old_rows(&self) -> &[usize] {
        &self.selected
    }
    /// Canonical polynomial conditions, including original and map denominators.
    pub fn nonzero_conditions(&self) -> &[Atom] {
        &self.conditions
    }

    /// Compile fresh values and precision through the existing rational compiler.
    pub fn compile(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        context: &RunContext,
    ) -> Result<CompiledStateCompression> {
        context.cancellation.check()?;
        if values.contains_key(&Atom::var(self.reduced.variable)) {
            return Err(Error::InvalidInput(
                "state compression ODE variable must remain free".into(),
            ));
        }
        let mut reduced = self.prepared_reduced.compile(p, values, context)?;
        reduced.exclude_polynomials(&self.conditions)?;
        let embedding = self.prepared_embedding.compile(p, values, context)?;
        context.cancellation.check()?;
        Ok(CompiledStateCompression {
            reduced,
            embedding,
            selected: self.selected.clone(),
            conditions: self.conditions.clone(),
            values: values.clone(),
            variable: self.reduced.variable,
            p,
        })
    }
}

impl CompiledStateCompression {
    /// The ordinary reduced system, with every retained source domain excluded.
    pub fn reduced_system(&self) -> &CompiledSystem {
        &self.reduced
    }

    fn check_point(&self, point: &C) -> Result<()> {
        if !self.p.finite(point) {
            return Err(Error::InvalidInput("nonfinite compression point".into()));
        }
        let mut values = self.values.clone();
        values.insert(Atom::var(self.variable), self.p.round(point));
        for condition in &self.conditions {
            if self.p.eval(condition, &values)? == self.p.zero() {
                return Err(Error::InvalidInput(
                    "compression point violates a retained nonzero domain".into(),
                ));
            }
        }
        Ok(())
    }

    /// Select initial values only after checking the complete reconstructed state.
    ///
    /// Uses the existing component-wise `Precision::close` criterion:
    /// `norm(full[i] - reconstructed[i]) <= 10^-checked_digits * max(1, norm(full[i]))`.
    /// Thus `checked_digits` is a mixed-scale tolerance exponent, not significant
    /// digits for a small nonzero component, and passing never certifies an exact
    /// zero. Reconstruct the projected values when recording individual residuals.
    /// This is a consistency check, not a forward-error or interval certificate
    /// for the physical boundary; independent relative checks remain separate.
    pub fn project_boundary(
        &self,
        full: &BoundaryData,
        checked_digits: u32,
    ) -> Result<BoundaryData> {
        let available = (u64::from(self.p.bits) * 1000 / 3322) as u32;
        if full.values.len() != self.embedding.matrix.len()
            || full.values.iter().any(|v| !self.p.finite(v))
            || checked_digits == 0
            || checked_digits >= available
        {
            return Err(Error::InvalidInput(
                "compression boundary dimensions, precision or values".into(),
            ));
        }
        let selected = self
            .selected
            .iter()
            .map(|&i| self.p.round(&full.values[i]))
            .collect::<Vec<_>>();
        let reconstructed = self.reconstruct(&full.point, &selected)?;
        for (i, (original, restored)) in full.values.iter().zip(&reconstructed).enumerate() {
            if !self.p.close(original, restored, checked_digits) {
                return Err(Error::Accuracy(format!(
                    "initial boundary is outside the supplied state subspace at component {i}"
                )));
            }
        }
        Ok(BoundaryData {
            point: self.p.round(&full.point),
            values: selected,
        })
    }

    /// Reconstruct all old components at a regular finite matching point.
    pub fn reconstruct(&self, point: &C, reduced: &[C]) -> Result<Vec<C>> {
        if reduced.len() != self.selected.len() || reduced.iter().any(|v| !self.p.finite(v)) {
            return Err(Error::InvalidInput(
                "reduced compression state dimensions or values".into(),
            ));
        }
        self.check_point(point)?;
        let coefficients = self
            .embedding
            .matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| Ok(entry.series(self.p, point, 0)?[0].clone()))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let field = FloatField::from_rep(self.p.zero());
        let matrix = Matrix::from_nested_vec(coefficients, field.clone())
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let state = Matrix::from_nested_vec(
            reduced.iter().map(|v| vec![self.p.round(v)]).collect(),
            field,
        )
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let output = (&matrix * &state).into_vec();
        if output.iter().any(|v| !self.p.finite(v)) {
            return Err(Error::Numerical("nonfinite reconstructed state".into()));
        }
        Ok(output.into_iter().map(|v| self.p.round(&v)).collect())
    }
}
