#![allow(clippy::needless_range_loop)] // Matrix row/column indexing mirrors the recurrence equations.
//! Generalized power/logarithm series at regular singular points.
use crate::algebra;
use crate::family::substitute;
use crate::numeric::solve;
use crate::ode::{polynomial_coefficients, quotient_series};
mod endpoint_support;
mod evaluate;
pub(crate) mod exact;
#[cfg(test)]
mod projection_tests;
mod recurrence;
mod streaming;
pub use exact::{ExactEliminationMode, ExactFrobeniusBasis, ExactFrobeniusLimits};

use crate::{
    ComplexFloat as C, DifferentialSystem, Error, Precision, Progress, Result, RunContext,
};
use std::collections::BTreeMap;
use symbolica::prelude::*;

#[derive(Clone, Debug)]
pub struct FrobeniusColumn {
    /// Symbolic exponent, retained to distinguish dimensional-regulator regions.
    pub exponent: Atom,
    /// coefficients[power][log_power][component]
    pub coefficients: Vec<Vec<Vec<C>>>,
}

#[derive(Clone, Debug)]
pub struct FrobeniusBasis {
    pub columns: Vec<FrobeniusColumn>,
    pub precision: Precision,
}

/// Parameter-independent, exact preparation of a regular-singular system.
///
/// Normalization, indicial factorization and generalized eigenspaces are computed
/// once with Symbolica's exact algebra. Numerical evaluations can then use any
/// precision, admissible generic parameter sample and truncation order without specializing the
/// symbolic exponents or sharing mutable numerical workspaces.
#[derive(Clone, Debug)]
pub struct PreparedFrobenius {
    source: DifferentialSystem,
    normal: DifferentialSystem,
    transformation: Vec<Vec<Atom>>,
    blocks: Vec<Vec<usize>>,
    eigenspaces: Vec<PreparedEigenspace>,
    nonzero_conditions: Vec<Atom>,
}

#[derive(Clone, Debug)]
struct PreparedEigenspace {
    exponent: Atom,
    /// columns[generalized eigenvector][log power][component]
    leading: Vec<Vec<Vec<Atom>>>,
}

fn phase(context: &RunContext, name: impl Into<String>) -> Result<()> {
    context.emit(Progress::Stage { name: name.into() })?;
    // A progress callback can request cancellation before the next exact owner call.
    context.cancellation.check()
}

#[cfg(test)]
mod high_order_memory_tests {
    use super::*;

    #[test]
    fn exact_log_width_admits_semisimple_high_precision_but_bounds_resonances() -> Result<()> {
        let x = symbol!("frobenius_memory::x");
        let prepare = |matrix| {
            DifferentialSystem {
                variable: x,
                matrix,
            }
            .prepare_frobenius(&RunContext::default())
        };
        let p = Precision::decimal(24_000)?;
        let simple = prepare(vec![vec![Atom::new(); 4]; 4])?;
        assert!(
            simple.estimated_numerical_peak_bytes(p, 27_000, false)
                < crate::series_limits::MAX_ESTIMATED_BYTES
        );
        assert!(
            simple.estimated_numerical_peak_bytes(p, 27_000, true)
                > crate::series_limits::MAX_ESTIMATED_BYTES
        );
        let mut resonant = vec![vec![Atom::new(); 4]; 4];
        for (i, row) in resonant.iter_mut().enumerate() {
            row[i] = Atom::num(i as i64) / Atom::var(x);
        }
        let resonant = prepare(resonant)?;
        assert!(
            resonant.estimated_numerical_peak_bytes(p, 27_000, false)
                > crate::series_limits::MAX_ESTIMATED_BYTES
        );
        let mut jordan = vec![vec![Atom::new(); 4]; 4];
        for (i, row) in jordan.iter_mut().enumerate().take(3) {
            row[i + 1] = Atom::one() / Atom::var(x);
        }
        let jordan = prepare(jordan)?;
        assert!(
            jordan.estimated_numerical_peak_bytes(p, 27_000, false)
                > crate::series_limits::MAX_ESTIMATED_BYTES
        );
        Ok(())
    }

    #[test]
    fn sparse_high_degree_source_is_admitted_before_dense_float_conversion() -> Result<()> {
        let x = symbol!("frobenius_source_memory::x");
        let prepared = DifferentialSystem {
            variable: x,
            matrix: vec![vec![Atom::var(x).pow(60000); 4]; 4],
        }
        .prepare_frobenius(&RunContext::default())?;
        let p = Precision::decimal(24_000)?;
        // The solution/solver estimate fits, but the dense connection would
        // exceed the budget. Refuse before allocating those float vectors.
        assert!(
            prepared.estimated_numerical_peak_bytes(p, 27_000, false)
                < crate::series_limits::MAX_ESTIMATED_BYTES
        );
        assert!(matches!(
            prepared.evaluate(p, &Default::default(), 27_000, &RunContext::default()),
            Err(Error::Limit(_))
        ));
        Ok(())
    }
}

/// Reject parameter specialization that changes the symbolic resonance pattern.
pub(crate) fn ensure_generic_exponents(
    exponents: &[&Atom],
    p: Precision,
    parameters: &ahash::HashMap<Atom, C>,
) -> Result<()> {
    for (i, a) in exponents.iter().enumerate() {
        for b in &exponents[..i] {
            let difference = (*a - *b).together().cancel();
            if difference.to_string().parse::<i64>().is_ok() {
                continue;
            }
            let value = p.eval(&difference, parameters)?;
            // Raw MPFR and Astro integer conversions have different rounding
            // semantics. Bound the binary exponent before using their shared
            // exact rational rounding owner: tiny dyadics need no denominator,
            // and large values cannot produce an i64 candidate.
            let integer = match value.re.as_raw().get_exp() {
                None => value.re.is_zero().then_some(0),
                Some(exponent) if exponent < 0 => Some(0),
                Some(exponent) if exponent > 64 => None,
                Some(_) => value
                    .re
                    .to_rational()
                    .round_to_nearest_integer()
                    .to_string()
                    .parse::<i64>()
                    .ok(),
            };
            if let Some(integer) = integer
                && p.close(&value, &p.i(integer), p.bits / 5)
            {
                return Err(Error::Unsupported("parameter specialization introduces an additional indicial resonance; choose a generic sample or prepare the exactly specialized system".into()));
            }
        }
    }
    Ok(())
}

pub(crate) fn valuation(a: &Atom, x: Symbol) -> Result<i64> {
    if a.is_zero() {
        return Ok(i64::MAX / 4);
    }
    let s = a
        .series(x, 0, 1)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    s.get_trailing_exponent()
        .to_string()
        .parse()
        .map_err(|_| Error::Unsupported("noninteger matrix valuation".into()))
}

impl DifferentialSystem {
    /// Remove higher poles when a diagonal integer-power shearing suffices.
    /// Returns shifts s such that original_y[i] = x^s[i] * normalized_y[i].
    pub fn diagonal_fuchsian_form(&self) -> Result<(Self, Vec<i64>)> {
        self.validate()?;
        let n = self.matrix.len();
        let orders = self
            .matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|a| {
                        if a.is_zero() {
                            Ok(None)
                        } else {
                            valuation(a, self.variable).map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut shifts = crate::integer_shearing::integer_potentials(&orders, 1, 10000, None)?;
        let minimum = *shifts.iter().min().unwrap();
        for s in &mut shifts {
            *s -= minimum;
        }
        if shifts.iter().any(|&s| s > 10000) {
            return Err(Error::Limit(
                "Fuchsian shearing exceeds 10000 powers".into(),
            ));
        }
        let x = Atom::var(self.variable);
        let transformation = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            x.clone().pow(shifts[i])
                        } else {
                            Atom::new()
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        Ok((self.change_basis(&transformation)?, shifts))
    }
    /// Constant similarity built from the kernel filtration of the leading
    /// nilpotent coefficient. This exposes hidden triangular higher poles
    /// before integer-power shearing.
    pub fn nilpotent_leading_rotation(&self) -> Result<Vec<Vec<Atom>>> {
        self.validate()?;
        let n = self.matrix.len();
        let order = self
            .matrix
            .iter()
            .flatten()
            .map(|a| valuation(a, self.variable))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min()
            .unwrap();
        if order >= -1 {
            return Err(Error::InvalidInput("rotation needs a higher pole".into()));
        }
        let leading = self
            .matrix
            .iter()
            .map(|r| {
                r.iter()
                    .map(|a| {
                        Ok(a.series(self.variable, 0, order + 1)
                            .map_err(|e| Error::Unsupported(e.to_string()))?
                            .coefficient(Rational::from(order))
                            .unwrap_or_default())
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut power = leading.clone();
        let mut vectors = Vec::new();
        for _ in 0..n {
            for v in algebra::nullspace(power.clone()) {
                let mut trial = vectors.clone();
                trial.push(v.clone());
                if algebra::rref(trial).1.len() > vectors.len() {
                    vectors.push(v);
                }
            }
            if vectors.len() == n {
                break;
            }
            power = algebra::matmul(&power, &leading);
        }
        if vectors.len() != n {
            return Err(Error::Unsupported(
                "nonnilpotent irregular singularity".into(),
            ));
        }
        Ok((0..n)
            .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
            .collect())
    }

    pub fn invert_variable(&self, new_variable: Symbol) -> Self {
        let z = Atom::var(new_variable);
        let rules = BTreeMap::from([(Atom::var(self.variable), z.clone().pow(-1))]);
        Self {
            variable: new_variable,
            matrix: self
                .matrix
                .iter()
                .map(|row| {
                    row.iter()
                        // Preserve original denominator domains until preparation
                        // collects them; normalization can cancel afterward.
                        .map(|a| -substitute(a, &rules) / z.clone().pow(2))
                        .collect()
                })
                .collect(),
        }
    }
    /// Exact residue for a system already in regular-singular form.
    pub fn residue(&self) -> Result<Vec<Vec<Atom>>> {
        self.residue_with_context(&RunContext::default())
    }

    fn residue_with_context(&self, context: &RunContext) -> Result<Vec<Vec<Atom>>> {
        self.validate()?;
        self.matrix
            .iter()
            .map(|row| {
                context.cancellation.check()?;
                row.iter()
                    .map(|a| {
                        if valuation(a, self.variable)? < -1 {
                            return Err(Error::Unsupported(
                                "matrix needs a nontrivial Fuchsian basis transformation".into(),
                            ));
                        }
                        let expansion = a
                            .series(self.variable, 0, 1)
                            .map_err(|e| Error::Unsupported(e.to_string()))?;
                        Ok(expansion
                            .coefficient(Rational::from(-1))
                            .unwrap_or_default())
                    })
                    .collect()
            })
            .collect()
    }

    /// Perform the exact work shared by numerical epsilon samples and precision
    /// refinements. Failed or cancelled preparations return no partial object.
    pub fn prepare_frobenius(&self, context: &RunContext) -> Result<PreparedFrobenius> {
        phase(context, "normalizing the exact Frobenius system")?;
        self.validate()?;
        let expressions = self.matrix.iter().flatten().cloned().collect::<Vec<_>>();
        let variables = expressions
            .iter()
            .flat_map(|a| a.get_all_symbols(true))
            .collect();
        // Collect before normalization so cancellable factors retain their domain.
        let nonzero_conditions =
            crate::physical_conditions::rational_denominator_conditions(&expressions, &variables)?;
        let (normal, transformation) = self.fuchsian_form(32)?;
        phase(context, "extracting the exact Frobenius residue")?;
        let residue = normal.residue_with_context(context)?;
        let n = residue.len();
        let blocks = normal.blocks()?;
        let mut roots = BTreeMap::new();
        for (index, block) in blocks.iter().enumerate() {
            phase(
                context,
                format!(
                    "factoring exact Frobenius indicial block {}/{}",
                    index + 1,
                    blocks.len()
                ),
            )?;
            let diagonal = block
                .iter()
                .map(|&i| block.iter().map(|&j| residue[i][j].clone()).collect())
                .collect::<Vec<_>>();
            for (root, multiplicity) in algebra::eigenvalues(&diagonal)? {
                *roots.entry(root).or_insert(0) += multiplicity;
            }
        }
        let mut eigenspaces = Vec::new();
        for (index, (lambda, multiplicity)) in roots.iter().enumerate() {
            phase(
                context,
                format!(
                    "constructing exact Frobenius eigenspace {}/{}",
                    index + 1,
                    roots.len()
                ),
            )?;
            let mut nilpotent = residue.clone();
            for (i, row) in nilpotent.iter_mut().enumerate() {
                context.cancellation.check()?;
                row[i] = (&row[i] - lambda).together().cancel();
            }
            let mut power = nilpotent.clone();
            let mut generalized = Vec::new();
            // Once the kernel reaches the algebraic multiplicity it is the full
            // generalized eigenspace. Large repeated eigenvalues need not have
            // equally long Jordan chains.
            for depth in 1..=*multiplicity {
                context.cancellation.check()?;
                generalized = algebra::nullspace(power.clone());
                if generalized.len() == *multiplicity {
                    break;
                }
                if depth < *multiplicity {
                    context.cancellation.check()?;
                    power = algebra::matmul(&power, &nilpotent);
                }
            }
            if generalized.len() != *multiplicity {
                return Err(Error::Numerical(
                    "generalized eigenspace dimension mismatch".into(),
                ));
            }
            let mut leading = Vec::new();
            for vector in generalized {
                let mut column = Vec::new();
                let mut v = vector;
                for k in 0..*multiplicity {
                    context.cancellation.check()?;
                    column.push(v.clone());
                    v = (0..n)
                        .map(|i| {
                            context.cancellation.check()?;
                            Ok(
                                ((0..n).fold(Atom::new(), |s, j| s + &nilpotent[i][j] * &v[j])
                                    / Atom::num((k + 1) as i64))
                                .together()
                                .cancel(),
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    if v.iter().all(|entry| entry.is_zero()) {
                        break;
                    }
                }
                leading.push(column);
            }
            eigenspaces.push(PreparedEigenspace {
                exponent: lambda.clone(),
                leading,
            });
        }
        context.cancellation.check()?;
        Ok(PreparedFrobenius {
            source: self.clone(),
            normal,
            transformation,
            blocks,
            eigenspaces,
            nonzero_conditions,
        })
    }

    /// Construct all power/logarithm solutions. Rational indicial roots are kept
    /// exact; integer resonances use a coupled logarithmic recurrence.
    /// For repeated samples, use [`Self::prepare_frobenius`] once instead.
    pub fn frobenius(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
    ) -> Result<FrobeniusBasis> {
        let context = RunContext::default();
        self.prepare_frobenius(&context)?
            .evaluate(p, values, order, &context)
    }
}

impl PreparedFrobenius {
    fn identity_transformation(&self) -> bool {
        self.transformation.iter().enumerate().all(|(i, row)| {
            row.iter().enumerate().all(|(j, entry)| {
                if i == j {
                    entry.is_one()
                } else {
                    entry.is_zero()
                }
            })
        })
    }

    fn estimated_numerical_peak_bytes(&self, p: Precision, order: usize, strict: bool) -> u128 {
        let n = self.source.matrix.len() as u128;
        // New logarithms can appear at positive integer differences of
        // indicial exponents. Generic numerical specialization is checked
        // before recurrence evaluation; strict endpoint mode keeps the full
        // dimension bound because it does not apply that genericity check.
        let resonant = strict
            || self.eigenspaces.iter().enumerate().any(|(i, a)| {
                self.eigenspaces[..i].iter().any(|b| {
                    (&a.exponent - &b.exponent)
                        .together()
                        .cancel()
                        .to_string()
                        .parse::<Integer>()
                        .is_ok_and(|difference| !difference.is_zero())
                })
            });
        let logs = if resonant {
            n
        } else {
            self.eigenspaces
                .iter()
                .flat_map(|space| &space.leading)
                .map(|column| column.len() as u128)
                .max()
                .unwrap_or(1)
        };
        let columns = self
            .eigenspaces
            .iter()
            .map(|space| space.leading.len() as u128)
            .sum::<u128>();
        // Basis normalization can extend a column by at most 256 powers.
        let length = (order as u128).saturating_add(257);
        let coefficients = length
            .saturating_mul(n)
            .saturating_mul(columns)
            .saturating_mul(logs);
        let matrix = n.saturating_mul(n);
        let block_factors = self
            .blocks
            .iter()
            .map(|block| (block.len() as u128).saturating_pow(2))
            .sum::<u128>();
        let retained_solvers = if self.eigenspaces.iter().any(|space| space.leading.len() > 1) {
            (order as u128 + 1).saturating_mul(matrix.saturating_add(block_factors))
        } else {
            matrix.saturating_add(block_factors)
        };
        let resonant_workspace = if resonant {
            n.saturating_pow(4).saturating_mul(8)
        } else {
            matrix.saturating_mul(2)
        };
        let recurrence_peak = coefficients
            .saturating_add(retained_solvers)
            .saturating_add(resonant_workspace);
        // Transformation replaces one column at a time and retains its short
        // exact coefficient list alongside the untransformed columns.
        let transformation_peak = coefficients
            .saturating_add(length.saturating_mul(n).saturating_mul(logs))
            .saturating_add(matrix.saturating_mul(257));
        recurrence_peak
            .max(transformation_peak)
            .saturating_mul(crate::series_limits::complex_bytes(p))
    }

    /// Original exact connection, including expressions before rational cancellation.
    pub fn system(&self) -> &DifferentialSystem {
        &self.source
    }

    /// Fuchsian connection used by the recurrence.
    pub fn normalized_system(&self) -> &DifferentialSystem {
        &self.normal
    }

    /// Exact transformation with original_y = T * normalized_y.
    pub fn basis_transformation(&self) -> &[Vec<Atom>] {
        &self.transformation
    }

    /// Original rational denominator restrictions. A singular expansion center
    /// is allowed; a parameter sample must not annihilate a whole restriction.
    pub fn nonzero_conditions(&self) -> &[Atom] {
        &self.nonzero_conditions
    }

    /// Distinct exact indicial exponents before the final basis transformation.
    pub fn exponents(&self) -> impl ExactSizeIterator<Item = &Atom> {
        self.eigenspaces.iter().map(|space| &space.exponent)
    }

    /// Evaluate with a fresh numerical workspace. The exact preparation can be
    /// shared concurrently across epsilon samples, precisions and series orders.
    pub fn evaluate(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
    ) -> Result<FrobeniusBasis> {
        self.evaluate_with_policy(p, values, order, context, false)
    }

    /// Endpoint admission must not erase a small divergent logarithmic sector.
    /// Keep every nonzero numerical row and require resonant equations to close
    /// without a magnitude-based zero threshold. Independent profiles remain
    /// necessary: this strict policy is not an exact arithmetic certificate.
    pub(crate) fn evaluate_endpoint(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
    ) -> Result<FrobeniusBasis> {
        self.evaluate_with_policy(p, values, order, context, true)
    }

    fn evaluate_with_policy(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
        strict: bool,
    ) -> Result<FrobeniusBasis> {
        let (prepared_rows, _) = self.numerical_rows(p, values, order, context, strict, true)?;
        self.numerical_recurrence(p, values, order, context, strict, &prepared_rows)
    }

    fn numerical_rows(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
        strict: bool,
        full_basis: bool,
    ) -> Result<(crate::ode::prepared::PreparedRows, u128)> {
        phase(context, "evaluating the prepared Frobenius recurrence")?;
        let solution_peak = if full_basis {
            self.estimated_numerical_peak_bytes(p, order, strict)
        } else {
            streaming::StreamingAdmission::estimated_column_bytes(
                p,
                self.source.matrix.len(),
                order,
                self.eigenspaces
                    .iter()
                    .flat_map(|space| &space.leading)
                    .map(Vec::len)
                    .max()
                    .unwrap_or(1),
                self.identity_transformation(),
            )
        };
        crate::series_limits::admit_bytes(order, solution_peak, "Frobenius expansion")?;
        // Reuse the finite-row owner for a dense source-storage estimate before
        // any high-precision coefficient conversion. Sparse high polynomial
        // powers can otherwise dwarf the retained solution series.
        let x = Atom::var(self.normal.variable);
        let rows = self
            .normal
            .matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| (entry * &x).together().cancel())
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let prepared_rows =
            crate::ode::prepared::PreparedRows::new(self.normal.variable, &rows, context)?;
        let source_values = prepared_rows
            .polynomial_storage_values(context)?
            .saturating_mul(4);
        let mut condition_values = 0_u128;
        for condition in &self.nonzero_conditions {
            context.cancellation.check()?;
            let length = crate::ode::polynomial_coefficient_terms(condition, self.source.variable)?
                .iter()
                .map(|(degree, _)| *degree as u128 + 1)
                .max()
                .unwrap_or(1);
            condition_values = condition_values.max(length);
        }
        crate::series_limits::admit_bytes(
            order,
            solution_peak.saturating_add(
                source_values
                    .saturating_add(condition_values)
                    .saturating_mul(crate::series_limits::complex_bytes(p)),
            ),
            "Frobenius expansion with finite source rows",
        )?;
        for condition in &self.nonzero_conditions {
            context.cancellation.check()?;
            let coefficients = polynomial_coefficients(condition, self.source.variable, p, values)?;
            if coefficients
                .iter()
                .all(|coefficient| coefficient == &p.zero())
            {
                return Err(Error::Numerical(
                    "parameter sample annihilates an original Frobenius denominator condition"
                        .into(),
                ));
            }
        }
        if !strict {
            ensure_generic_exponents(&self.exponents().collect::<Vec<_>>(), p, values)?;
        }
        Ok((
            prepared_rows,
            source_values.saturating_add(condition_values),
        ))
    }
}

/// RREF with free parameters set to zero, used only at exact detected resonances.
fn solve_any(
    p: Precision,
    mut a: Vec<Vec<C>>,
    mut b: Vec<C>,
    context: &RunContext,
    strict: bool,
) -> Result<Vec<C>> {
    let n = b.len();
    let mut row = 0;
    let mut pivots = Vec::new();
    let tol = p.tolerance(p.bits / 4);
    for col in 0..n {
        context.cancellation.check()?;
        let Some(pivot) =
            (row..n).max_by(|&i, &j| p.norm(&a[i][col]).partial_cmp(&p.norm(&a[j][col])).unwrap())
        else {
            break;
        };
        if if strict {
            a[pivot][col] == p.zero()
        } else {
            p.norm(&a[pivot][col]) < tol
        } {
            continue;
        }
        a.swap(row, pivot);
        b.swap(row, pivot);
        let v = a[row][col].clone();
        for j in col..n {
            a[row][j] = p.div(&a[row][j], &v);
        }
        b[row] = p.div(&b[row], &v);
        for i in 0..n {
            if i != row && a[i][col] != p.zero() {
                let f = a[i][col].clone();
                for j in col..n {
                    a[i][j] = p.sub(&a[i][j], &p.mul(&f, &a[row][j]));
                }
                b[i] = p.sub(&b[i], &p.mul(&f, &b[row]));
            }
        }
        pivots.push(col);
        row += 1;
        if row == n {
            break;
        }
    }
    if b.iter().skip(row).any(|v| {
        if strict {
            v != &p.zero()
        } else {
            p.norm(v) > tol
        }
    }) {
        return Err(Error::Numerical(
            "inconsistent resonant Frobenius recurrence".into(),
        ));
    }
    let mut result = vec![p.zero(); n];
    for (i, &j) in pivots.iter().enumerate() {
        result[j] = b[i].clone();
    }
    Ok(result)
}

impl FrobeniusBasis {
    pub(crate) fn validate(&self) -> Result<()> {
        let n = self.columns.len();
        if n == 0
            || self.columns.iter().any(|c| {
                c.coefficients.is_empty()
                    || c.coefficients
                        .iter()
                        .any(|logs| logs.is_empty() || logs.iter().any(|row| row.len() != n))
            })
        {
            return Err(Error::InvalidInput(
                "Frobenius coefficient dimensions".into(),
            ));
        }
        if self
            .columns
            .iter()
            .flat_map(|c| &c.coefficients)
            .flatten()
            .flatten()
            .any(|value| !self.precision.finite(value))
        {
            return Err(Error::Numerical("nonfinite Frobenius coefficient".into()));
        }
        Ok(())
    }
    pub fn evaluate(&self, z: &C, values: &ahash::HashMap<Atom, C>) -> Result<Vec<Vec<C>>> {
        self.evaluate_with_winding(z, values, 0)
    }

    /// Evaluate on the logarithm branch `log(z) + 2*pi*i*winding`. The same
    /// logarithm determines fractional powers and all explicit logarithms.
    /// Winding is local about the expansion center, not a contour planner.
    pub fn evaluate_with_winding(
        &self,
        z: &C,
        values: &ahash::HashMap<Atom, C>,
        winding: i32,
    ) -> Result<Vec<Vec<C>>> {
        self.validate()?;
        let p = self.precision;
        let log = evaluate::logarithm(p, z, winding)?;
        let n = self.columns.len();
        let mut matrix = vec![vec![p.zero(); n]; n];
        for (j, column) in self.columns.iter().enumerate() {
            for (i, value) in evaluate::column_value(p, z, &log, values, column)?
                .into_iter()
                .enumerate()
            {
                matrix[i][j] = value;
            }
        }
        Ok(matrix)
    }
    pub fn match_values(
        &self,
        z: &C,
        values: &[C],
        parameters: &ahash::HashMap<Atom, C>,
    ) -> Result<Vec<C>> {
        solve(
            self.precision,
            self.evaluate(z, parameters)?,
            values.to_vec(),
        )
    }
    /// Select the constant Taylor coefficient in dimensional regularization.
    /// Nonconstant epsilon-dependent powers are discarded before epsilon fitting.
    pub fn physical_limit(&self, constants: &[C], epsilon: Symbol) -> Result<Vec<C>> {
        self.validate()?;
        let n = self.columns.len();
        (0..n)
            .map(|i| {
                let weights = (0..n)
                    .map(|j| Atom::num(i64::from(i == j)))
                    .collect::<Vec<_>>();
                project_limit(
                    self,
                    constants,
                    &weights,
                    epsilon,
                    symbol!("symbolica_amflow::endpoint_projection"),
                    &ahash::HashMap::default(),
                )
            })
            .collect()
    }
}

pub(crate) fn project_limit(
    basis: &FrobeniusBasis,
    constants: &[C],
    weights: &[Atom],
    epsilon: Symbol,
    eta: Symbol,
    parameters: &ahash::HashMap<Atom, C>,
) -> Result<C> {
    basis.validate()?;
    let p = basis.precision;
    if constants.len() != basis.columns.len() || weights.len() != basis.columns.len() {
        return Err(Error::InvalidInput("endpoint projection dimensions".into()));
    }
    let mut terms: BTreeMap<(Rational, usize), C> = BTreeMap::new();
    // Several endpoint columns share an exponent. A target's rational
    // weights have the same valuation and series in each such column;
    // expand and evaluate each required prefix only once per projection.
    let mut valuations = vec![None; weights.len()];
    let mut expansions = BTreeMap::<(usize, i64), Vec<(i64, C)>>::new();
    for (column, constant) in basis.columns.iter().zip(constants) {
        if *constant == p.zero() || !column.exponent.derivative(epsilon).is_zero() {
            continue;
        }
        let lambda = if let AtomView::Num(n) = column.exponent.as_view() {
            if let symbolica::coefficient::Coefficient::Complex(c) = n.get_coeff_view().to_owned() {
                if !c.im.is_zero() {
                    return Err(Error::Unsupported(
                        "complex physical endpoint exponent".into(),
                    ));
                }
                c.re
            } else {
                return Err(Error::Unsupported("nonrational endpoint exponent".into()));
            }
        } else {
            return Err(Error::Unsupported(
                "parameter-dependent physical endpoint exponent".into(),
            ));
        };
        for (i, weight) in weights.iter().enumerate() {
            if weight.is_zero() {
                continue;
            }
            let valuation = if let Some(value) = valuations[i] {
                value
            } else {
                let value = crate::frobenius::valuation(weight, eta)?;
                valuations[i] = Some(value);
                value
            };
            if &lambda + &Rational::from(valuation + column.coefficients.len() as i64)
                <= Rational::zero()
            {
                return Err(Error::Accuracy(
                    "endpoint series does not reach every term required by the target reduction"
                        .into(),
                ));
            }
            let through = (-lambda.clone())
                .floor()
                .to_string()
                .parse::<i64>()
                .map_err(|_| Error::Limit("endpoint exponent exceeds index range".into()))?;
            if through < valuation {
                continue;
            }
            if through.saturating_sub(valuation) > 10000 {
                return Err(Error::Limit(
                    "endpoint target expansion exceeds 10000 terms".into(),
                ));
            }
            let key = (i, through);
            if let std::collections::btree_map::Entry::Vacant(entry) = expansions.entry(key) {
                let expansion = weight
                    .series(eta, 0, through + 1)
                    .map_err(|e| Error::Unsupported(e.to_string()))?;
                let coefficients = expansion
                    .terms()
                    .map(|(power, coefficient)| {
                        let power = power.to_string().parse::<i64>().map_err(|_| {
                            Error::Unsupported("noninteger reduction exponent".into())
                        })?;
                        Ok((power, p.eval(coefficient, parameters)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                entry.insert(coefficients);
            }
            for (power, coefficient) in &expansions[&key] {
                for (k, logs) in column.coefficients.iter().enumerate() {
                    let exponent = &lambda + &Rational::from(*power + k as i64);
                    if exponent > Rational::zero() {
                        break;
                    }
                    for (l, row) in logs.iter().enumerate() {
                        let term = p.mul(constant, &p.mul(coefficient, &row[i]));
                        let entry = terms
                            .entry((exponent.clone(), l))
                            .or_insert_with(|| p.zero());
                        *entry = p.add(entry, &term);
                    }
                }
            }
        }
    }
    for ((power, log), value) in &terms {
        if (*power < Rational::zero() || *log > 0) && p.norm(value) > p.tolerance(p.bits / 5) {
            return Err(Error::Numerical(
                "uncancelled physical endpoint divergence".into(),
            ));
        }
    }
    Ok(terms
        .remove(&(Rational::zero(), 0))
        .unwrap_or_else(|| p.zero()))
}
