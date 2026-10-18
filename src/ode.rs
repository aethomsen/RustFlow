//! Rational linear differential systems and arbitrary-precision Taylor transport.
use crate::StepSizeStrategy;
use crate::local_coordinates::{LocalCoordinate, TaylorCoordinate};
use crate::{ComplexFloat as C, Error, FlowOptions, Precision, Progress, Result, RunContext};
use std::sync::Arc;
#[path = "ode/conditioning.rs"]
pub(crate) mod conditioning;
#[path = "ode/residual.rs"]
pub(crate) mod residual;
use conditioning::ConditioningChart;
#[path = "ode/integer_residual.rs"]
pub(crate) mod integer_residual;
#[path = "ode/pade.rs"]
pub mod pade;
#[path = "ode/source.rs"]
pub(crate) mod source;
use residual::RationalResidualChart;
#[path = "ode/compression.rs"]
mod compression;
#[path = "ode/fundamental.rs"]
pub(crate) mod fundamental;
#[path = "ode/prepared.rs"]
pub(crate) mod prepared;
pub use compression::{CompiledStateCompression, StateCompression};
#[cfg(test)]
#[path = "ode/compression_tests.rs"]
mod compression_tests;
use symbolica::domains::float::FloatField;
use symbolica::poly::univariate::UnivariatePolynomial;
use symbolica::prelude::*;

#[derive(Clone, Debug)]
pub struct DifferentialSystem {
    pub variable: Symbol,
    pub matrix: Vec<Vec<Atom>>,
}

#[derive(Clone, Debug)]
pub struct BoundaryData {
    pub point: C,
    pub values: Vec<C>,
}

#[derive(Clone, Debug, Default)]
pub struct FlowDiagnostics {
    /// Whole-chart fundamental proofs used for supplied boundary errors.
    pub fundamental_boundary_charts: usize,
    /// Why an optional fundamental proof retained scalar source-error handling.
    pub fundamental_boundary_fallback: Option<String>,
    /// Scalar accuracy rejection which triggered an automatic stronger proof.
    pub fundamental_boundary_retry: Option<String>,
    pub pade_trials: usize,
    pub pade_steps: usize,
    pub pade_fallbacks: usize,
    pub last_pade_fallback: Option<String>,
    /// Committed continuation segments.
    pub steps: usize,
    /// Trials that failed an acceptance check.
    pub rejected_steps: usize,
    /// Physical step proposals checked; the authoritative proposal budget.
    /// A Padé failure may additionally check Taylor at the same proposal.
    /// On success this equals steps + rejected_steps + superseded_successes.
    pub predicate_evaluations: usize,
    /// Successful trials replaced by a larger successful trial in the same chart.
    pub superseded_successes: usize,
    pub working_bits: u32,
    pub expansion_order: usize,
    /// Minimum checked endpoint arithmetic-conditioning tolerance along this
    /// run (mixed decimal scale). This is not a full forward-error certificate.
    pub conditioning_digits: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct FlowResult {
    pub point: C,
    pub values: Vec<C>,
    pub diagnostics: FlowDiagnostics,
}

#[derive(Clone, Debug)]
pub(crate) struct NumericRational {
    pub numerator: Vec<C>,
    pub denominator: Vec<C>,
}

/// Native coefficient extraction shared by fixed-precision and exact source compilation.
pub(crate) fn polynomial_coefficient_terms(
    a: &Atom,
    variable: Symbol,
) -> Result<Vec<(usize, Atom)>> {
    let x = Atom::var(variable);
    let mut terms = Vec::new();
    for (monomial, coefficient) in
        crate::coefficient::exact_coefficient_list(a, std::slice::from_ref(&x))?
    {
        if coefficient.contains_symbol(variable) {
            return Err(Error::InvalidInput("non-polynomial coefficient".into()));
        }
        let degree = if monomial.is_one() {
            0
        } else if monomial == x {
            1
        } else if let AtomView::Pow(power) = monomial.as_view() {
            let (base, exponent) = power.get_base_exp();
            if base != x.as_view() {
                return Err(Error::InvalidInput("non-polynomial coefficient".into()));
            }
            exponent
                .to_string()
                .parse::<usize>()
                .map_err(|_| Error::InvalidInput("noninteger polynomial power".into()))?
        } else {
            return Err(Error::InvalidInput(format!(
                "non-polynomial monomial {monomial}"
            )));
        };
        if degree > 100_000 {
            return Err(Error::Limit("polynomial degree exceeds 100000".into()));
        }
        terms.push((degree, coefficient));
    }
    Ok(terms)
}

pub(crate) fn polynomial_coefficients(
    a: &Atom,
    variable: Symbol,
    p: Precision,
    values: &ahash::HashMap<Atom, C>,
) -> Result<Vec<C>> {
    let mut result = vec![p.zero()];
    for (degree, coefficient) in polynomial_coefficient_terms(a, variable)? {
        result.resize(result.len().max(degree + 1), p.zero());
        result[degree] = p.add(&result[degree], &p.eval(&coefficient, values)?);
    }
    while result.len() > 1 && result.last() == Some(&p.zero()) {
        result.pop();
    }
    Ok(result)
}

pub(crate) fn shift_polynomial(
    p: Precision,
    coefficients: &[C],
    center: &C,
    order: usize,
) -> Vec<C> {
    let mut out = vec![p.zero(); order + 1];
    for a in coefficients.iter().rev() {
        for k in (1..=order).rev() {
            out[k] = p.add(&p.mul(&out[k], center), &out[k - 1]);
        }
        out[0] = p.add(&p.mul(&out[0], center), a);
    }
    out
}

pub(crate) fn quotient_series(
    p: Precision,
    numerator: &[C],
    denominator: &[C],
    order: usize,
) -> Result<Vec<C>> {
    if denominator.is_empty() || denominator[0] == p.zero() {
        return Err(Error::Numerical("expansion center is a pole".into()));
    }
    let mut out = vec![p.zero(); order + 1];
    for n in 0..=order {
        let mut v = numerator.get(n).cloned().unwrap_or_else(|| p.zero());
        for j in 1..=n.min(denominator.len() - 1) {
            v = p.sub(&v, &p.mul(&denominator[j], &out[n - j]));
        }
        out[n] = p.div(&v, &denominator[0]);
    }
    Ok(out)
}

fn polynomial_roots(p: Precision, coefficients: &[C], variable: Symbol) -> Result<Vec<C>> {
    let degree = coefficients.len() - 1;
    if degree == 1 {
        return Ok(vec![p.neg(&p.div(&coefficients[0], &coefficients[1]))]);
    }
    // Fujiwara's bound, rounded upward to a power of two. With x = scale*z,
    // every root lies in the unit disk and the monic coefficients stay bounded.
    // Symbolica's Aberth solver uses an absolute polynomial residual; passing
    // raw integer coefficients can make its stopping tolerance unattainable.
    let leading = &coefficients[degree];
    let mut exponent = None;
    for (i, coefficient) in coefficients[..degree].iter().enumerate() {
        let magnitude = p.norm(&p.div(coefficient, leading));
        if !magnitude.is_finite() {
            return Err(Error::Numerical("nonfinite pole polynomial".into()));
        }
        if let Some(e) = magnitude.as_raw().get_exp() {
            let n = (degree - i) as i64;
            let bound = (i64::from(e) + n - 1).div_euclid(n) + 1;
            exponent = Some(exponent.map_or(bound, |old: i64| old.max(bound)));
        }
    }
    let scale = p.powi(&p.i(2), exponent.unwrap_or(0));
    if !p.finite(&scale) || scale == p.zero() {
        return Err(Error::Numerical(
            "pole scaling exceeds numerical range".into(),
        ));
    }
    let mut scaled = coefficients
        .iter()
        .enumerate()
        .map(|(i, c)| p.div(&p.div(c, leading), &p.powi(&scale, (degree - i) as i64)))
        .collect::<Vec<_>>();
    scaled[degree] = p.i(1);
    let poly = UnivariatePolynomial::from_coefficients(
        &FloatField::from_rep(p.zero()),
        scaled.clone(),
        Arc::new(PolyVariable::Symbol(variable)),
    );
    let roots = poly.roots(1000, &p.tolerance(p.bits / 4)).map_err(|_| {
        Error::Numerical(format!(
            "degree-{degree} scaled pole root finding did not converge"
        ))
    })?;
    if roots.len() != degree || roots.iter().any(|root| !p.finite(root)) {
        return Err(Error::Numerical("incomplete pole root set".into()));
    }
    // Verify Newton corrections as well as the polynomial reconstructed from
    // all roots. A small residual alone is weak near clustered roots.
    let tolerance = p.tolerance(p.bits / 5);
    let mut reconstructed = vec![p.i(1)];
    for (i, root) in roots.iter().enumerate() {
        let mut value = p.i(1);
        let mut derivative = p.zero();
        for coefficient in scaled[..degree].iter().rev() {
            derivative = p.add(&p.mul(&derivative, root), &value);
            value = p.add(&p.mul(&value, root), coefficient);
        }
        if derivative == p.zero()
            || p.norm(&p.div(&value, &derivative)) > tolerance
            || roots[..i]
                .iter()
                .any(|other| p.norm(&p.sub(root, other)) <= tolerance)
        {
            return Err(Error::Numerical(
                "unresolved or ill-conditioned pole roots".into(),
            ));
        }
        let mut next = vec![p.zero(); reconstructed.len() + 1];
        for (j, coefficient) in reconstructed.iter().enumerate() {
            next[j] = p.sub(&next[j], &p.mul(root, coefficient));
            next[j + 1] = p.add(&next[j + 1], coefficient);
        }
        reconstructed = next;
    }
    if reconstructed
        .iter()
        .zip(&scaled)
        .any(|(a, b)| p.norm(&p.sub(a, b)) > tolerance)
    {
        return Err(Error::Numerical(
            "pole roots fail polynomial reconstruction".into(),
        ));
    }
    Ok(roots.iter().map(|root| p.mul(root, &scale)).collect())
}

fn retain_distinct_pole(p: Precision, poles: &mut Vec<C>, root: C) {
    if !poles.iter().any(|v| {
        let a = p.norm(v);
        let b = p.norm(&root);
        let scale = if a < b { a } else { b };
        p.norm(&p.sub(v, &root)) <= p.tolerance(p.bits / 5) * scale
    }) {
        poles.push(root);
    }
}

impl NumericRational {
    pub(crate) fn series(&self, p: Precision, center: &C, order: usize) -> Result<Vec<C>> {
        quotient_series(
            p,
            &shift_polynomial(p, &self.numerator, center, order),
            &shift_polynomial(p, &self.denominator, center, order),
            order,
        )
    }
}

type ExactPolynomial = MultivariatePolynomial<IntegerRing, u16>;
type DenominatorFactors = Arc<Vec<(ExactPolynomial, usize)>>;

/// Factor each exact denominator once, retaining content and multiplicities.
/// The pole scan and the directed disk enclosure share this certificate.
fn denominator_factors(
    denominator: &ExactPolynomial,
    cache: &mut ahash::HashMap<ExactPolynomial, DenominatorFactors>,
) -> Result<DenominatorFactors> {
    if let Some(factors) = cache.get(denominator) {
        return Ok(factors.clone());
    }
    let factors = denominator.factor();
    let mut product = denominator.one();
    for (factor, multiplicity) in &factors {
        if *multiplicity == 0
            || factor.variables().iter().enumerate().any(|(i, _)| {
                usize::from(factor.degree(i))
                    .checked_mul(*multiplicity)
                    .is_none_or(|d| d > usize::from(u16::MAX))
            })
        {
            return Err(Error::Limit("denominator factor degree overflow".into()));
        }
        product = multiply_polynomials(&product, &factor.pow(*multiplicity))?;
    }
    if product != *denominator {
        return Err(Error::Numerical(
            "exact denominator factor reconstruction failed".into(),
        ));
    }
    let factors = Arc::new(factors);
    cache.insert(denominator.clone(), factors.clone());
    Ok(factors)
}

fn multiply_polynomials(a: &ExactPolynomial, b: &ExactPolynomial) -> Result<ExactPolynomial> {
    for (i, variable) in a.variables().iter().enumerate() {
        if let Some(j) = b.variables().iter().position(|v| v == variable)
            && a.degree(i).checked_add(b.degree(j)).is_none()
        {
            return Err(Error::Limit(
                "cleared differential polynomial degree overflow".into(),
            ));
        }
    }
    Ok(a * b)
}

/// A row of D(x) y'(x) = A(x) y(x), with an exact common denominator
/// cleared before numerical specialization. Structural zero entries are absent.
#[derive(Clone, Debug)]
pub(crate) struct PolynomialRow {
    pub(crate) denominator: Vec<C>,
    pub(crate) entries: Vec<(usize, Vec<C>)>,
}

#[derive(Clone, Debug)]
struct ExactRows {
    variable: Symbol,
    rows: Vec<Vec<Atom>>,
    values: ahash::HashMap<Atom, C>,
}

#[derive(Clone, Debug)]
pub struct CompiledSystem {
    source: Arc<ExactRows>,
    pub(crate) p: Precision,
    pub(crate) matrix: Vec<Vec<NumericRational>>,
    pub poles: Vec<C>,
    pub(crate) pole_polynomials: Vec<Atom>,
    pub(crate) polynomial_rows: Vec<PolynomialRow>,
    pub(crate) exact_source_rows: Vec<source::ExactPolynomialRow>,
}

impl DifferentialSystem {
    /// Change basis using old_y = transformation * new_y, preserving the
    /// derivative term: B = T^-1 A T - T^-1 dT/dx.
    pub fn change_basis(&self, transformation: &[Vec<Atom>]) -> Result<Self> {
        self.validate()?;
        if transformation.len() != self.matrix.len()
            || transformation.iter().any(|r| r.len() != self.matrix.len())
        {
            return Err(Error::InvalidInput(
                "basis transformation dimensions".into(),
            ));
        }
        let inverse = crate::algebra::inverse(transformation)?;
        let mut product = crate::algebra::matmul(&self.matrix, transformation);
        for (i, row) in product.iter_mut().enumerate() {
            for (j, a) in row.iter_mut().enumerate() {
                *a = (&*a - transformation[i][j].derivative(self.variable))
                    .together()
                    .cancel();
            }
        }
        Ok(Self {
            variable: self.variable,
            matrix: crate::algebra::matmul(&inverse, &product),
        })
    }
    pub fn validate(&self) -> Result<()> {
        validate_ode_variable(self.variable)?;
        let n = self.matrix.len();
        if n == 0 || self.matrix.iter().any(|r| r.len() != n) {
            return Err(Error::InvalidInput(
                "nonempty square differential matrix required".into(),
            ));
        }
        Ok(())
    }
    pub fn compile(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
    ) -> Result<CompiledSystem> {
        self.validate()?;
        compile_rows(self.variable, &self.matrix, p, values)
    }
    /// Strongly connected blocks in dependency order, independent of input ordering.
    pub fn blocks(&self) -> Result<Vec<Vec<usize>>> {
        self.validate()?;
        use linnet::half_edge::{
            HedgeGraph, algorithms::DirectionBasis, builder::HedgeGraphBuilder,
        };
        let mut builder = HedgeGraphBuilder::<(), usize>::new();
        let nodes = (0..self.matrix.len())
            .map(|i| builder.add_node(i))
            .collect::<Vec<_>>();
        for (i, row) in self.matrix.iter().enumerate() {
            for (j, coefficient) in row.iter().enumerate() {
                if i != j && !coefficient.is_zero() {
                    builder.add_edge(nodes[i], nodes[j], (), true);
                }
            }
        }
        let graph: HedgeGraph<(), usize> = builder.build();
        Ok(graph
            .strongly_connected_components(DirectionBasis::Underlying)
            .into_iter()
            .map(|component| component.into_iter().map(|node| node.0).collect())
            .collect())
    }
}

fn validate_ode_variable(variable: Symbol) -> Result<()> {
    if variable == crate::family::imaginary_parameter() {
        return Err(Error::InvalidInput(
            "the reserved imaginary-unit symbol cannot be an ODE variable".into(),
        ));
    }
    Ok(())
}

/// Compile possibly rectangular rational rows. Column indices remain sparse in
/// the cleared polynomial representation, allowing shared epsilon-order rows.
pub(crate) fn compile_rows(
    variable: Symbol,
    rows: &[Vec<Atom>],
    p: Precision,
    values: &ahash::HashMap<Atom, C>,
) -> Result<CompiledSystem> {
    let context = RunContext::default();
    prepared::PreparedRows::new(variable, rows, &context)?.compile(p, values, &context)
}

impl CompiledSystem {
    /// Retain declared nonzero polynomial domains, even when the differential
    /// matrix is regular there. Roots restrict contour planning and Taylor
    /// radii; the exact polynomials survive local-coordinate recompilation.
    ///
    /// Supply the exact polynomial factors before canceling their assumptions
    /// from a reduction or change of basis. Other symbols must already have
    /// values in this compiled system. An identically zero or numerically
    /// unresolved guard is rejected. This preserves the supplied domain; it
    /// does not recover discarded factors or choose a physical branch.
    pub fn exclude_polynomials(&mut self, guards: &[Atom]) -> Result<()> {
        if guards.is_empty() {
            return Ok(());
        }
        let guard_rows = guards
            .iter()
            .map(|g| {
                if g.is_zero() {
                    return Err(Error::InvalidInput(
                        "continuation chart violates a declared nonzero domain".into(),
                    ));
                }
                Ok(vec![Atom::num(1) / g])
            })
            .collect::<Result<Vec<_>>>()?;
        let exclusions = compile_rows(
            self.source.variable,
            &guard_rows,
            self.p,
            &self.source.values,
        )?;
        for root in exclusions.poles {
            retain_distinct_pole(self.p, &mut self.poles, root);
        }
        self.pole_polynomials.extend(exclusions.pole_polynomials);
        Ok(())
    }

    /// Recompile the retained exact source in the local coordinate. Original
    /// source domains and the map denominator survive Jacobian cancellation.
    pub(crate) fn in_coordinate(&self, coordinate: &TaylorCoordinate) -> Result<Self> {
        let source = &self.source;
        let (rows, mut guards) = coordinate.pullback_rows(source.variable, &source.rows)?;
        let replacements = std::collections::BTreeMap::from([(
            Atom::var(source.variable),
            coordinate.expression(source.variable)?,
        )]);
        guards.extend(self.pole_polynomials.iter().map(|g| {
            crate::family::substitute(g, &replacements)
                .together()
                .cancel()
        }));
        let mut compiled = compile_rows(source.variable, &rows, self.p, &source.values)?;
        compiled.exclude_polynomials(&guards)?;
        Ok(compiled)
    }
    /// A real radius strictly inside the local convergence disk around an
    /// endpoint, excluding only a pole exactly at that endpoint.
    pub fn endpoint_radius(&self, endpoint: &C) -> Float {
        let p = self.p;
        self.poles
            .iter()
            .map(|pole| p.norm(&p.sub(pole, endpoint)))
            .filter(|r| *r > p.real(0))
            .fold(p.real(1), |a, b| if a < b { a } else { b })
            / 8
    }
    pub fn dimension(&self) -> usize {
        self.matrix.len()
    }
    pub(crate) fn taylor(&self, center: &C, values: &[C], order: usize) -> Result<Vec<Vec<C>>> {
        self.taylor_channels(center, values, order, 1)
    }

    pub(crate) fn taylor_workspace_bytes(&self, order: usize, channels: usize) -> u128 {
        let degree = self
            .polynomial_rows
            .iter()
            .flat_map(|row| {
                std::iter::once(row.denominator.len()).chain(
                    row.entries
                        .iter()
                        .map(|(_, coefficients)| coefficients.len()),
                )
            })
            .max()
            .unwrap_or(1)
            .saturating_sub(1);
        // y/y', directed residual polynomials and their derivatives, temporary
        // native polynomial products, and conditioning balls coexist during
        // chart construction. Keep the full source polynomial degree: sparse
        // high powers must not bypass either the defect or memory checks.
        (order as u128 + degree as u128 + 2)
            .saturating_mul(self.dimension() as u128)
            .saturating_mul(channels as u128)
            .saturating_mul(12)
            .saturating_mul(crate::series_limits::complex_bytes(self.p))
    }

    /// Shared finite-polynomial recurrence for ordinary and epsilon-series
    /// transport. A column `shift * n + component` couples channel `e` to
    /// `e - shift`; ordinary transport has one channel and zero shifts.
    /// Translate each row once and retain the sparse coefficient hierarchy,
    /// without constructing a dense augmented differential system.
    pub(crate) fn taylor_channels(
        &self,
        center: &C,
        values: &[C],
        order: usize,
        channels: usize,
    ) -> Result<Vec<Vec<C>>> {
        let p = self.p;
        let n = self.dimension();
        let size = n
            .checked_mul(channels)
            .ok_or_else(|| Error::Limit("series channel dimension overflow".into()))?;
        if channels == 0 || values.len() != size {
            return Err(Error::InvalidInput("boundary channel dimensions".into()));
        }
        crate::series_limits::admit_bytes(
            order,
            self.taylor_workspace_bytes(order, channels),
            "rational Taylor chart",
        )?;
        let zero = p.zero();
        // Translate only the finite polynomial degrees needed at this order;
        // no rational matrix entry is expanded into an order-long power series.
        let rows = self
            .polynomial_rows
            .iter()
            .map(|row| {
                let mut denominator = shift_polynomial(
                    p,
                    &row.denominator,
                    center,
                    (row.denominator.len() - 1).min(order.saturating_sub(1)),
                );
                if denominator[0] == zero {
                    return Err(Error::Numerical("expansion center is a pole".into()));
                }
                let divisor = denominator[0].clone();
                for c in &mut denominator {
                    *c = p.div(c, &divisor);
                }
                let entries = row
                    .entries
                    .iter()
                    .map(|(j, coefficients)| {
                        let mut shifted = shift_polynomial(
                            p,
                            coefficients,
                            center,
                            (coefficients.len() - 1).min(order.saturating_sub(1)),
                        );
                        for c in &mut shifted {
                            *c = p.div(c, &divisor);
                        }
                        (*j, shifted)
                    })
                    .collect::<Vec<_>>();
                Ok(PolynomialRow {
                    denominator,
                    entries,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut y = vec![vec![zero.clone(); size]; order + 1];
        // Store y' coefficients as well as y. Then the denominator convolution
        // does not repeatedly multiply y_k by its integer derivative weight.
        let mut derivative = vec![vec![zero.clone(); size]; order];
        y[0] = values.to_vec();
        // With D_0 normalized to one, match each channel of D y' = A y.
        for k in 0..order {
            for e in 0..channels {
                for (i, row) in rows.iter().enumerate() {
                    let index = e * n + i;
                    let mut sum = zero.clone();
                    for (column, coefficients) in &row.entries {
                        let shift = column / n;
                        if shift > e {
                            continue;
                        }
                        let source = (e - shift) * n + column % n;
                        for (l, coefficient) in coefficients.iter().enumerate().take(k + 1) {
                            if *coefficient != zero {
                                sum = p.add(&sum, &p.mul(coefficient, &y[k - l][source]));
                            }
                        }
                    }
                    for (l, coefficient) in row.denominator.iter().enumerate().skip(1).take(k) {
                        if *coefficient != zero {
                            sum = p.sub(&sum, &p.mul(coefficient, &derivative[k - l][index]));
                        }
                    }
                    derivative[k][index] = sum;
                    y[k + 1][index] = p.scale(&derivative[k][index], 1, (k + 1) as i64);
                }
            }
        }
        Ok(y)
    }
    /// Plan a polygonal contour around disjoint pole disks. `side` is +1
    /// for a counterclockwise detour relative to the segment, or -1 for the
    /// opposite side. The choice fixes the continuation homotopy.
    pub fn plan_path(&self, start: &C, end: &C, side: i64) -> Result<Vec<C>> {
        plan_path(self.p, &self.poles, start, end, side)
    }

    pub fn transport(
        &self,
        boundary: &BoundaryData,
        waypoints: &[C],
        options: &FlowOptions,
        context: &RunContext,
    ) -> Result<FlowResult> {
        transport_series(self, boundary, waypoints, options, context, None)
    }
}

/// A retained local Taylor or rational segment, validated from `center` to
/// `end`. Seed coefficients are indexed by Taylor power, then component.
#[derive(Clone, Debug)]
pub struct TaylorSegment {
    pub center: C,
    pub end: C,
    /// Stored Taylor coefficients used to build the chart. If `pade` is
    /// present, use `evaluate_local` for the accepted/saved function.
    pub coefficients: Vec<Vec<C>>,
    pub pade: Option<Arc<pade::RationalCandidate>>,
    pub working_bits: u32,
    pub coordinate: TaylorCoordinate,
    pub conditioning_digits: u32,
}

impl TaylorSegment {
    pub fn evaluate_local(&self, point: &C) -> Result<(Vec<C>, u32)> {
        let p = Precision {
            bits: self.working_bits,
        };
        if let Some(candidate) = &self.pade {
            let values = candidate.evaluate(p, point)?.0;
            let seed_values = evaluate_taylor(p, &self.coefficients, point).0;
            let seed = ConditioningChart::new(p, &self.coefficients)?.check(
                p,
                point,
                &seed_values,
                self.conditioning_digits,
            )?;
            let checked = candidate
                .check_conditioning(p, point, &values, self.conditioning_digits)?
                .min(seed);
            Ok((values, checked))
        } else {
            let values = evaluate_taylor(p, &self.coefficients, point).0;
            let checked = ConditioningChart::new(p, &self.coefficients)?.check(
                p,
                point,
                &values,
                self.conditioning_digits,
            )?;
            Ok((values, checked))
        }
    }
}

pub(crate) type MappedTaylorChart<Chart> = (Vec<Vec<C>>, Chart, Vec<C>);

pub(crate) trait SeriesSystem {
    type State;
    type Chart;
    fn initial_state(&self, boundary: &BoundaryData) -> Result<Self::State>;
    fn precision(&self) -> Precision;
    fn dimension(&self) -> usize;
    fn poles(&self) -> &[C];
    /// Owners with a bounded high-order chart implementation provide its
    /// conservative workspace estimate. Other owners retain the former cap.
    fn extended_order_workspace_bytes(&self, _order: usize) -> Result<u128> {
        Err(Error::Unsupported(
            "series orders above 10000 require an allocation estimate from this chart owner".into(),
        ))
    }
    fn local_chart(
        &self,
        center: &C,
        values: &[C],
        order: usize,
        state: &Self::State,
    ) -> Result<(Vec<Vec<C>>, Self::Chart)>;
    fn mapped_chart(
        &self,
        coordinate: &TaylorCoordinate,
        center: &C,
        values: &[C],
        order: usize,
        state: &Self::State,
    ) -> Result<MappedTaylorChart<Self::Chart>> {
        let _ = (coordinate, center, values, order, state);
        Err(Error::Unsupported(
            "this series system has no mapped local-chart compiler".into(),
        ))
    }
    fn local_chart_with_options(
        &self,
        center: &C,
        values: &[C],
        order: usize,
        state: &Self::State,
        _options: &FlowOptions,
        context: &RunContext,
    ) -> Result<(Vec<Vec<C>>, Self::Chart)> {
        context.cancellation.check()?;
        let chart = self.local_chart(center, values, order, state)?;
        context.cancellation.check()?;
        Ok(chart)
    }
    #[allow(clippy::too_many_arguments)] // Shared source options and cancellation context.
    fn mapped_chart_with_options(
        &self,
        coordinate: &TaylorCoordinate,
        center: &C,
        values: &[C],
        order: usize,
        state: &Self::State,
        _options: &FlowOptions,
        context: &RunContext,
    ) -> Result<MappedTaylorChart<Self::Chart>> {
        context.cancellation.check()?;
        let chart = self.mapped_chart(coordinate, center, values, order, state)?;
        context.cancellation.check()?;
        Ok(chart)
    }
    /// The step is in the chart's polynomial coordinate. Unavailable bounds
    /// reject the trial; they cannot be treated as a successful certificate.
    fn whole_segment_residual(&self, chart: &Self::Chart, step: &C) -> Result<Option<Vec<Float>>>;
    fn rhs(&self, point: &C, values: &[C], chart: &Self::Chart) -> Result<Vec<C>>;
    /// A source owner may distinguish arithmetic enclosure width from a real
    /// differential defect. Existing mandatory evidence remains the default.
    fn whole_segment_residual_with_budget(
        &self,
        chart: &Self::Chart,
        step: &C,
        _values: &[C],
        _tolerance: &Float,
    ) -> Result<Option<Vec<Float>>> {
        self.whole_segment_residual(chart, step)
    }

    fn pade_candidate(
        &self,
        _coordinate: &TaylorCoordinate,
        _center: &C,
        _coefficients: &[Vec<C>],
        _options: &crate::PadeOptions,
        _context: &RunContext,
    ) -> Result<Option<pade::RationalCandidate>> {
        Ok(None)
    }

    fn accepted_state(
        &self,
        chart: &Self::Chart,
        point: &C,
        tolerance: &Float,
    ) -> Result<Option<Self::State>>;
}
impl SeriesSystem for CompiledSystem {
    type State = ();
    type Chart = RationalResidualChart;
    fn initial_state(&self, _: &BoundaryData) -> Result<()> {
        Ok(())
    }
    fn accepted_state(&self, _: &RationalResidualChart, _: &C, _: &Float) -> Result<Option<()>> {
        Ok(Some(()))
    }
    fn precision(&self) -> Precision {
        self.p
    }
    fn dimension(&self) -> usize {
        self.dimension()
    }
    fn poles(&self) -> &[C] {
        &self.poles
    }
    fn extended_order_workspace_bytes(&self, order: usize) -> Result<u128> {
        Ok(self.taylor_workspace_bytes(order, 1))
    }
    fn local_chart(
        &self,
        center: &C,
        values: &[C],
        order: usize,
        state: &(),
    ) -> Result<(Vec<Vec<C>>, Self::Chart)> {
        self.local_chart_with_options(
            center,
            values,
            order,
            state,
            &FlowOptions::default(),
            &RunContext::default(),
        )
    }
    fn local_chart_with_options(
        &self,
        center: &C,
        values: &[C],
        order: usize,
        _: &(),
        options: &FlowOptions,
        context: &RunContext,
    ) -> Result<(Vec<Vec<C>>, Self::Chart)> {
        context.cancellation.check()?;
        let coefficients = self.taylor(center, values, order)?;
        let chart = RationalResidualChart::new_with_options(
            self.p,
            &self.exact_source_rows,
            center,
            &coefficients,
            1,
            options,
            context,
        )?;
        Ok((coefficients, chart))
    }
    fn mapped_chart(
        &self,
        coordinate: &TaylorCoordinate,
        center: &C,
        values: &[C],
        order: usize,
        state: &(),
    ) -> Result<MappedTaylorChart<Self::Chart>> {
        self.mapped_chart_with_options(
            coordinate,
            center,
            values,
            order,
            state,
            &FlowOptions::default(),
            &RunContext::default(),
        )
    }
    #[allow(clippy::too_many_arguments)] // Shared source options and cancellation context.
    fn mapped_chart_with_options(
        &self,
        coordinate: &TaylorCoordinate,
        _: &C,
        values: &[C],
        order: usize,
        _: &(),
        options: &FlowOptions,
        context: &RunContext,
    ) -> Result<MappedTaylorChart<Self::Chart>> {
        context.cancellation.check()?;
        let mapped = self.in_coordinate(coordinate)?;
        let coefficients = mapped.taylor(&self.p.zero(), values, order)?;
        let chart = RationalResidualChart::new_with_options(
            self.p,
            &mapped.exact_source_rows,
            &self.p.zero(),
            &coefficients,
            1,
            options,
            context,
        )?;
        Ok((coefficients, chart, mapped.poles))
    }
    fn pade_candidate(
        &self,
        coordinate: &TaylorCoordinate,
        center: &C,
        coefficients: &[Vec<C>],
        options: &crate::PadeOptions,
        context: &RunContext,
    ) -> Result<Option<pade::RationalCandidate>> {
        let mapped;
        let (source, center) = if matches!(coordinate, TaylorCoordinate::Identity) {
            (self, center.clone())
        } else {
            mapped = self.in_coordinate(coordinate)?;
            (&mapped, self.p.zero())
        };
        pade::RationalCandidate::build(
            self.p,
            &source.exact_source_rows,
            &center,
            coefficients,
            1,
            options,
            context,
        )
        .map(Some)
    }
    fn whole_segment_residual(&self, chart: &Self::Chart, step: &C) -> Result<Option<Vec<Float>>> {
        chart.defect_bounds(self.p, step).map(Some)
    }
    fn whole_segment_residual_with_budget(
        &self,
        chart: &RationalResidualChart,
        step: &C,
        values: &[C],
        tolerance: &Float,
    ) -> Result<Option<Vec<Float>>> {
        chart
            .defect_bounds_with_budget(self.p, step, values, tolerance)
            .map(Some)
    }
    fn rhs(&self, point: &C, values: &[C], _: &RationalResidualChart) -> Result<Vec<C>> {
        self.matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(values)
                    .try_fold(self.p.zero(), |sum, (entry, value)| {
                        Ok(self.p.add(
                            &sum,
                            &self.p.mul(&entry.series(self.p, point, 0)?[0], value),
                        ))
                    })
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)] // Shared physical-coordinate defect check.
fn residual_is_small<S: SeriesSystem>(
    system: &S,
    chart: &S::Chart,
    coordinate: &TaylorCoordinate,
    center: &C,
    point: &C,
    step: &C,
    coefficients: &[Vec<C>],
    candidate: Option<&pade::RationalCandidate>,
    values: &[C],
    tolerance: &Float,
    location: &str,
    rejection: &mut String,
) -> Result<bool> {
    let p = system.precision();
    let local = coordinate.local_point(p, center, point)?;
    let jacobian = coordinate.jacobian_at(p, center, point)?;
    let rhs = match system.rhs(point, values, chart) {
        Ok(rhs) => rhs,
        Err(Error::Accuracy(message)) => {
            *rejection = format!("{location} right-hand side: {message}");
            return Ok(false);
        }
        Err(error) => return Err(error),
    };
    let rational_derivatives = candidate
        .map(|a| a.evaluate(p, &local).map(|a| a.1))
        .transpose()?;
    for (i, rhs) in rhs.iter().enumerate() {
        let mut derivative = p.zero();
        for k in (1..coefficients.len()).rev() {
            derivative = p.add(
                &p.mul(&derivative, &local),
                &p.scale(&coefficients[k][i], k as i64, 1),
            );
        }
        if let Some(derivatives) = &rational_derivatives {
            derivative = derivatives[i].clone();
        }
        let derivative = p.div(&derivative, &jacobian);
        let defect = p.mul(step, &p.sub(&derivative, rhs));
        let magnitude = p.norm(&values[i]);
        let scale = if magnitude > p.real(1) {
            magnitude
        } else {
            p.real(1)
        };
        if !p.finite(&defect) {
            *rejection = format!("{location} nonfinite differential defect in component {i}");
            return Ok(false);
        }
        let error = p.norm(&defect);
        let budget = tolerance.clone() * scale;
        if error > budget {
            *rejection = format!(
                "{location} differential defect in component {i}: error={error}, budget={budget}"
            );
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn transport_series(
    system: &impl SeriesSystem,
    boundary: &BoundaryData,
    waypoints: &[C],
    options: &FlowOptions,
    context: &RunContext,
    saved: Option<&mut Vec<TaylorSegment>>,
) -> Result<FlowResult> {
    Ok(transport_series_with_state(system, boundary, waypoints, options, context, saved)?.0)
}

pub(crate) fn transport_series_with_state<S: SeriesSystem>(
    system: &S,
    boundary: &BoundaryData,
    waypoints: &[C],
    options: &FlowOptions,
    context: &RunContext,
    saved: Option<&mut Vec<TaylorSegment>>,
) -> Result<(FlowResult, S::State)> {
    transport_series_observed(
        system,
        boundary,
        waypoints,
        options,
        context,
        saved,
        |_, _| {},
    )
}

type CheckedStep<State> = Option<(C, Vec<C>, State, u32, bool)>;

/// Read-only checks within one fixed Taylor chart. Only the returned candidate
/// chosen by the controller may become committed state or a saved segment.
struct StepTrial<'a, S: SeriesSystem> {
    system: &'a S,
    chart: &'a S::Chart,
    coordinate: &'a TaylorCoordinate,
    center: &'a C,
    delta: &'a C,
    target: &'a C,
    coefficients: &'a [Vec<C>],
    tolerance: &'a Float,
    conditioning: &'a ConditioningChart,
    conditioning_digits: u32,
    pade: Option<&'a pade::RationalCandidate>,
}
impl<S: SeriesSystem> StepTrial<'_, S> {
    fn evaluate(
        &self,
        step: &C,
        rejection: &mut String,
        diagnostics: &mut FlowDiagnostics,
    ) -> Result<CheckedStep<S::State>> {
        if let Some(candidate) = self.pade {
            diagnostics.pade_trials += 1;
            match self.evaluate_candidate(step, rejection, Some(candidate)) {
                Ok(Some(accepted)) => return Ok(Some(accepted)),
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(error) => *rejection = error.to_string(),
                Ok(None) => {}
            }
            diagnostics.pade_fallbacks += 1;
            diagnostics.last_pade_fallback = Some(rejection.clone());
        }
        self.evaluate_candidate(step, rejection, None)
    }
    fn evaluate_candidate(
        &self,
        step: &C,
        rejection: &mut String,
        candidate: Option<&pade::RationalCandidate>,
    ) -> Result<CheckedStep<S::State>> {
        let p = self.system.precision();
        // A supplied endpoint may retain more bits than this compiled system.
        // Use the declared physical point for values, defects and branch state.
        let next = if step == self.delta {
            self.target.clone()
        } else {
            p.add(self.center, step)
        };
        if next == *self.center {
            return Err(Error::InsufficientPrecision {
                minimum_bits: p.bits.saturating_mul(2),
                context: "physical continuation step rounds back to its center".into(),
            });
        }
        let displacement = p.sub(&next, self.center);
        let local = self.coordinate.local_point(p, self.center, &next)?;
        let (values, tail) = if let Some(candidate) = candidate {
            (
                candidate.evaluate(p, &local)?.0,
                vec![p.real(0); self.system.dimension()],
            )
        } else {
            evaluate_taylor(p, self.coefficients, &local)
        };
        for (i, (value, tail)) in values.iter().zip(&tail).enumerate() {
            let magnitude = p.norm(value);
            let scale = if magnitude > p.real(1) {
                magnitude
            } else {
                p.real(1)
            };
            let budget = self.tolerance.clone() * scale;
            let good = p.finite(value) && *tail <= budget;
            if !good {
                *rejection = format!(
                    "Taylor tail in component {i}: error={tail}, budget={budget}, finite_value={}",
                    p.finite(value)
                );
                return Ok(None);
            }
        }
        if !residual_is_small(
            self.system,
            self.chart,
            self.coordinate,
            self.center,
            &next,
            &displacement,
            self.coefficients,
            candidate,
            &values,
            self.tolerance,
            "endpoint",
            rejection,
        )? {
            return Ok(None);
        }
        // The final retained terms can vanish for sparse systems. Keep the
        // independent midpoint defect as well as the endpoint defect above.
        let midpoint = p.coordinate_midpoint(self.center, &next)?;
        let half = p.sub(&midpoint, self.center);
        let local_middle = self.coordinate.local_point(p, self.center, &midpoint)?;
        let middle = if let Some(candidate) = candidate {
            candidate.evaluate(p, &local_middle)?.0
        } else {
            evaluate_taylor(p, self.coefficients, &local_middle).0
        };
        if !residual_is_small(
            self.system,
            self.chart,
            self.coordinate,
            self.center,
            &midpoint,
            &half,
            self.coefficients,
            candidate,
            &middle,
            self.tolerance,
            "midpoint",
            rejection,
        )? {
            return Ok(None);
        }
        let enclosure = if let Some(candidate) = candidate {
            candidate
                .defect_bounds(p, &local, &values, self.tolerance)
                .map(Some)
        } else {
            self.system.whole_segment_residual_with_budget(
                self.chart,
                &local,
                &values,
                self.tolerance,
            )
        };
        let majorant = match enclosure {
            Ok(Some(bounds)) => Some(bounds),
            Ok(None) => {
                *rejection = "whole-segment differential defect bound is unavailable".into();
                return Ok(None);
            }
            Err(Error::Accuracy(message)) => {
                *rejection = format!("whole-segment differential defect: {message}");
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        if let Some(bounds) = majorant {
            if bounds.len() != values.len() {
                return Err(Error::InvalidInput("residual bound dimensions".into()));
            }
            for (i, (bound, value)) in bounds.iter().zip(&values).enumerate() {
                let magnitude = p.norm(value);
                let scale = if magnitude > p.real(1) {
                    magnitude
                } else {
                    p.real(1)
                };
                let budget = self.tolerance.clone() * scale;
                if !bound.is_finite() || *bound < p.real(0) || *bound > budget {
                    *rejection = format!(
                        "whole-segment differential defect in component {i}: bound={bound}, budget={budget}"
                    );
                    return Ok(None);
                }
            }
        }
        let checked = if let Some(candidate) = candidate {
            // Preserve the existing arithmetic safeguard for seed coefficients;
            // compare that diagnostic with the Taylor value, not the rational value.
            let seed_values = evaluate_taylor(p, self.coefficients, &local).0;
            let seed =
                self.conditioning
                    .check(p, &local, &seed_values, self.conditioning_digits)?;
            seed.min(candidate.check_conditioning(p, &local, &values, self.conditioning_digits)?)
        } else {
            self.conditioning
                .check(p, &local, &values, self.conditioning_digits)?
        };
        let state = self
            .system
            .accepted_state(self.chart, &next, self.tolerance)?;
        if state.is_none() {
            *rejection = "local branch or state admission failed".into();
        }
        Ok(state.map(|state| (next, values, state, checked, candidate.is_some())))
    }
}

/// Metadata follows the exact same acceptance gate as values and state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn transport_series_observed<S: SeriesSystem>(
    system: &S,
    boundary: &BoundaryData,
    waypoints: &[C],
    options: &FlowOptions,
    context: &RunContext,
    mut saved: Option<&mut Vec<TaylorSegment>>,
    mut observe: impl FnMut(&S::Chart, &S::State),
) -> Result<(FlowResult, S::State)> {
    options.validate()?;
    context.cancellation.check()?;
    let p = system.precision();
    let extended_workspace = if options.series_order > crate::series_limits::LEGACY_MAX_ORDER {
        Some(system.extended_order_workspace_bytes(options.series_order)?)
    } else {
        None
    };
    let mut retained_bytes = saved.as_ref().map_or(0, |segments| {
        segments
            .iter()
            .map(|segment| {
                (segment.coefficients.iter().map(Vec::len).sum::<usize>() as u128).saturating_mul(
                    crate::series_limits::complex_bytes(Precision {
                        bits: segment.working_bits,
                    }),
                )
            })
            .fold(0_u128, u128::saturating_add)
    });
    if let Some(workspace) = extended_workspace {
        crate::series_limits::admit_bytes(
            options.series_order,
            workspace.saturating_add(retained_bytes),
            "Taylor transport with retained segments",
        )?;
    }
    if boundary.values.len() != system.dimension()
        || !p.finite(&boundary.point)
        || boundary.values.iter().any(|v| !p.finite(v))
        || waypoints.iter().any(|v| !p.finite(v))
    {
        return Err(Error::InvalidInput("invalid boundary or path".into()));
    }
    let mut state = system.initial_state(boundary)?;
    let mut center = boundary.point.clone();
    let mut values = boundary.values.clone();
    let mut diagnostics = FlowDiagnostics {
        working_bits: p.bits,
        expansion_order: options.series_order,
        ..Default::default()
    };
    for target in waypoints {
        let mut previous_step: Option<C> = None;
        while center != *target {
            if let Some(workspace) = extended_workspace {
                crate::series_limits::admit_bytes(
                    options.series_order,
                    workspace.saturating_add(retained_bytes),
                    "Taylor transport with retained segments",
                )?;
            }
            context.emit(Progress::Step {
                index: diagnostics.steps,
            })?;
            if diagnostics.predicate_evaluations >= options.max_steps {
                return Err(Error::Limit(
                    "Taylor continuation step budget exhausted".into(),
                ));
            }
            let delta = p.sub(target, &center);
            if !p.finite(&delta) || delta == p.zero() {
                return Err(Error::Accuracy(
                    "distinct continuation coordinates have an unresolved displacement".into(),
                ));
            }
            let distance = p.norm(&delta);
            let coordinate = match options.local_coordinate {
                LocalCoordinate::Identity => TaylorCoordinate::Identity,
                LocalCoordinate::BalancedMobius => {
                    TaylorCoordinate::balanced(p, &center, target, system.poles())?
                }
            };
            let mut step = delta.clone();
            context.cancellation.check()?;
            let (coefficients, chart) = if matches!(coordinate, TaylorCoordinate::Identity) {
                let radius = system
                    .poles()
                    .iter()
                    .map(|s| p.norm(&p.sub(&center, s)))
                    .min_by(|a, b| a.partial_cmp(b).unwrap());
                if let Some(radius) = radius {
                    if radius == p.real(0) {
                        return Err(Error::Numerical(
                            "path reached a differential-equation pole".into(),
                        ));
                    }
                    let safe = radius / 3;
                    if distance > safe {
                        step = p.mul(
                            &delta,
                            &p.div(&C::new(safe, p.real(0)), &C::new(distance, p.real(0))),
                        );
                    }
                } else if let Some(previous) = &previous_step {
                    // Entire systems have no pole-based scale. Reuse the last
                    // accepted step instead of repeatedly rejecting the full
                    // remaining interval. Every proposal still passes all checks.
                    let proposal = p.norm(&p.scale(previous, 2, 1));
                    if distance > proposal {
                        step = p.mul(
                            &delta,
                            &p.div(&C::new(proposal, p.real(0)), &C::new(distance, p.real(0))),
                        );
                    }
                }
                system.local_chart_with_options(
                    &center,
                    &values,
                    options.series_order,
                    &state,
                    options,
                    context,
                )?
            } else {
                let (coefficients, chart, poles) = system.mapped_chart_with_options(
                    &coordinate,
                    &center,
                    &values,
                    options.series_order,
                    &state,
                    options,
                    context,
                )?;
                if let Some(radius) = poles
                    .iter()
                    .map(|pole| p.norm(pole))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                {
                    if radius == p.real(0) {
                        return Err(Error::Numerical(
                            "mapped chart starts on an excluded domain".into(),
                        ));
                    }
                    // Positive real local steps preserve the declared straight
                    // physical leg. Actual transformed poles bound this proposal.
                    let proposed =
                        coordinate.physical_point(p, &center, &C::new(radius / 3, p.real(0)))?;
                    let candidate = p.sub(&proposed, &center);
                    if p.norm(&candidate) < distance {
                        step = candidate;
                    }
                }
                (coefficients, chart)
            };
            // Regulator fitting and endpoint matching consume guard digits too.
            // Keep the same truncation target under either proposal policy.
            let truncation_digits = options
                .digits
                .saturating_add(8)
                .max((options.digits + options.guard_digits).saturating_sub(10));
            let available_digits = (u64::from(p.bits) * 1000 / 3322) as u32;
            let tolerance = p.tolerance(truncation_digits.min(available_digits.saturating_sub(3)));
            let conditioning = ConditioningChart::new(p, &coefficients)?;
            let pade = if let Some(pade_options) = &options.pade {
                match system.pade_candidate(
                    &coordinate,
                    &center,
                    &coefficients,
                    pade_options,
                    context,
                ) {
                    Ok(Some(candidate)) => Some(Arc::new(candidate)),
                    Err(Error::Cancelled) => return Err(Error::Cancelled),
                    result => {
                        diagnostics.pade_fallbacks += 1;
                        diagnostics.last_pade_fallback = Some(match result {
                            Err(error) => error.to_string(),
                            _ => "source owner supports Taylor candidates only".into(),
                        });
                        None
                    }
                }
            } else {
                None
            };
            let trial = StepTrial {
                system,
                chart: &chart,
                coordinate: &coordinate,
                center: &center,
                delta: &delta,
                target,
                coefficients: &coefficients,
                tolerance: &tolerance,
                conditioning: &conditioning,
                conditioning_digits: options.digits,
                pade: pade.as_deref(),
            };
            let bracketed = options.step_size_strategy == StepSizeStrategy::Bracketed;
            let mut accepted = None;
            let mut failed_upper = None;
            let mut rejection = String::new();
            let mut first_rejection = None;
            for _ in 0..32 {
                context.cancellation.check()?;
                if diagnostics.predicate_evaluations >= options.max_steps {
                    return Err(Error::Limit(
                        "Taylor continuation predicate budget exhausted".into(),
                    ));
                }
                diagnostics.predicate_evaluations += 1;
                if let Some(candidate) = trial.evaluate(&step, &mut rejection, &mut diagnostics)? {
                    accepted = Some(candidate);
                    break;
                }
                first_rejection.get_or_insert_with(|| rejection.clone());
                if bracketed {
                    failed_upper = Some(step.clone());
                }
                step = p.scale(&step, 1, 2);
                diagnostics.rejected_steps += 1;
                if diagnostics.predicate_evaluations >= options.max_steps {
                    return Err(Error::Limit(format!(
                        "Taylor continuation step budget exhausted ({} accepted, {} rejected, {} predicates) at {center} toward {target}",
                        diagnostics.steps,
                        diagnostics.rejected_steps,
                        diagnostics.predicate_evaluations,
                    )));
                }
            }
            let mut accepted = accepted.ok_or_else(|| Error::Accuracy(format!(
                "Taylor tail or local chart did not meet tolerance after 32 local rejections ({} accepted, {} rejected, {} predicates overall) at {center} toward {target}; first rejection: {}; last rejection: {rejection}",
                diagnostics.steps, diagnostics.rejected_steps, diagnostics.predicate_evaluations,
                first_rejection.as_deref().unwrap_or("unavailable"),
            )))?;
            if let Some(mut upper) = failed_upper {
                // This is only a bounded proposal search, not a monotonicity
                // assumption. Every selected trial independently passes all checks.
                for _ in 0..2 {
                    context.cancellation.check()?;
                    if diagnostics.predicate_evaluations >= options.max_steps {
                        break;
                    }
                    let proposal = p.scale(&p.add(&step, &upper), 1, 2);
                    if proposal == step || proposal == upper {
                        break;
                    }
                    diagnostics.predicate_evaluations += 1;
                    if let Some(candidate) =
                        trial.evaluate(&proposal, &mut rejection, &mut diagnostics)?
                    {
                        accepted = candidate;
                        step = proposal;
                        diagnostics.superseded_successes += 1;
                    } else {
                        upper = proposal;
                        diagnostics.rejected_steps += 1;
                    }
                }
            }
            context.cancellation.check()?;
            let (next, next_values, next_state, conditioning_digits, used_pade) = accepted;
            if used_pade {
                diagnostics.pade_steps += 1;
            }
            diagnostics.conditioning_digits = Some(
                diagnostics
                    .conditioning_digits
                    .map_or(conditioning_digits, |old| old.min(conditioning_digits)),
            );
            values = next_values;
            state = next_state;
            observe(&chart, &state);
            if let Some(segments) = saved.as_deref_mut() {
                retained_bytes = retained_bytes.saturating_add(
                    (coefficients.iter().map(Vec::len).sum::<usize>() as u128)
                        .saturating_mul(crate::series_limits::complex_bytes(p)),
                );
                segments.push(TaylorSegment {
                    center: center.clone(),
                    end: next.clone(),
                    coefficients,
                    pade: if used_pade { pade } else { None },
                    working_bits: p.bits,
                    coordinate: coordinate.clone(),
                    conditioning_digits: options.digits,
                });
            }
            previous_step = Some(step);
            center = next;
            diagnostics.steps += 1;
        }
    }
    Ok((
        FlowResult {
            point: center,
            values,
            diagnostics,
        },
        state,
    ))
}

pub(crate) fn plan_path(
    p: Precision,
    poles: &[C],
    start: &C,
    end: &C,
    side: i64,
) -> Result<Vec<C>> {
    if ![-1, 1].contains(&side) || !p.finite(start) || !p.finite(end) {
        return Err(Error::InvalidInput(
            "invalid contour endpoints or side".into(),
        ));
    }
    let mut disks = Vec::new();
    for (i, pole) in poles.iter().enumerate() {
        let mut radius = p.norm(&p.sub(start, pole));
        let end_distance = p.norm(&p.sub(end, pole));
        if end_distance < radius {
            radius = end_distance;
        }
        if radius == p.real(0) {
            return Err(Error::InvalidInput("contour endpoint is a pole".into()));
        }
        for (j, other) in poles.iter().enumerate() {
            if i != j && pole != other {
                let distance = p.norm(&p.sub(pole, other));
                if distance < radius {
                    radius = distance;
                }
            }
        }
        disks.push((pole, radius / 4));
    }
    let mut segments = vec![(start.clone(), end.clone())];
    let mut output = Vec::new();
    let mut iterations = 0;
    while let Some((a, b)) = segments.pop() {
        iterations += 1;
        if iterations > 4096 {
            return Err(Error::Limit(
                "contour planning exceeds 4096 subdivisions".into(),
            ));
        }
        let delta = p.sub(&b, &a);
        let length = p.norm(&delta);
        if length == p.real(0) {
            output.push(b);
            continue;
        }
        let obstruction = disks
            .iter()
            .filter_map(|(pole, radius)| {
                let t = p.div(&p.sub(pole, &a), &delta).re;
                if t <= p.real(0) || t >= p.real(1) {
                    return None;
                }
                let projection = p.add(&a, &p.mul(&delta, &C::new(t.clone(), p.real(0))));
                (p.norm(&p.sub(pole, &projection)) < *radius).then_some((t, *pole, radius))
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let Some((_, pole, radius)) = obstruction {
            let normal = p.div(
                &p.mul(&delta, &p.complex(0, side)),
                &C::new(length, p.real(0)),
            );
            let waypoint = p.add(
                pole,
                &p.mul(&normal, &C::new(radius.clone() * 2, p.real(0))),
            );
            segments.push((waypoint.clone(), b));
            segments.push((a, waypoint));
        } else {
            output.push(b);
        }
    }
    Ok(output)
}

pub(crate) fn evaluate_taylor(
    p: Precision,
    coefficients: &[Vec<C>],
    h: &C,
) -> (Vec<C>, Vec<Float>) {
    let n = coefficients[0].len();
    let mut out = vec![p.zero(); n];
    let mut tail = vec![p.real(0); n];
    let start = coefficients.len().saturating_sub(6);
    for k in (0..coefficients.len()).rev() {
        for i in 0..n {
            out[i] = p.add(&p.mul(&out[i], h), &coefficients[k][i]);
        }
    }
    let mut power = p.powi(h, start as i64);
    for row in coefficients.iter().skip(start) {
        for (i, c) in row.iter().enumerate() {
            tail[i] += p.norm(&p.mul(c, &power));
        }
        power = p.mul(&power, h);
    }
    (out, tail)
}

#[cfg(test)]
mod recurrence_tests {
    use super::*;

    #[test]
    fn denominator_factor_cache_retains_native_content_and_multiplicity() {
        let polynomial: RationalPolynomial<IntegerRing, u16> =
            parse!("-42*(factor_cache_x-1)^7*(factor_cache_x+2)^3")
                .try_to_rational_polynomial(&Q, &Z, None)
                .unwrap();
        let mut cache = ahash::HashMap::default();
        let factors = denominator_factors(&polynomial.numerator, &mut cache).unwrap();
        let again = denominator_factors(&polynomial.numerator, &mut cache).unwrap();
        assert!(Arc::ptr_eq(&factors, &again));
        assert_eq!(cache.len(), 1);
        assert!(factors.iter().any(|(_, multiplicity)| *multiplicity == 7));
        assert!(factors.iter().any(|(_, multiplicity)| *multiplicity == 3));
        assert_eq!(
            factors
                .iter()
                .fold(polynomial.numerator.one(), |out, (factor, multiplicity)| {
                    out * &factor.pow(*multiplicity)
                }),
            polynomial.numerator
        );
    }

    #[test]
    fn large_exact_row_clearing_preserves_the_connection() -> Result<()> {
        // Each source coefficient fits in double's exponent range, but clearing
        // the two denominators creates coefficients of order 10^400. Collection
        // must remain exact regardless of this intermediate normalization.
        let x = symbol!("large_cleared_row::x");
        let epsilon = Atom::var(symbol!("large_cleared_row::eps"));
        let large = Atom::num(Integer::from(10).pow(200));
        let source = DifferentialSystem {
            variable: x,
            matrix: vec![
                vec![
                    Atom::Zero,
                    &large * (&epsilon + Atom::num(1)) / (&large + Atom::var(x)),
                    &large / (&large + Atom::num(1) + Atom::var(x)),
                ],
                vec![Atom::Zero; 3],
                vec![Atom::Zero; 3],
            ],
        };
        for working in [40, 80] {
            let p = Precision::decimal(working)?;
            let system = source.compile(
                p,
                &ahash::HashMap::from_iter([(epsilon.clone(), p.scale(&p.i(1), 1, 2))]),
            )?;
            assert_eq!(system.polynomial_rows[0].entries.len(), 2);
            let values = vec![p.zero(), p.i(1), p.i(1)];
            for center in [p.zero(), p.complex(0, -79_245)] {
                let actual = system.taylor(&center, &values, 8)?;
                let expected = rational_series_taylor(&system, &center, &values, 8)?;
                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                    assert!(p.close(a, b, working - 10), "{a} != {b}");
                }
            }
            let result = system.transport(
                &BoundaryData {
                    point: p.zero(),
                    values,
                },
                &[p.i(1)],
                &FlowOptions {
                    digits: 20,
                    guard_digits: working - 20,
                    series_order: 8,
                    ..Default::default()
                },
                &RunContext::default(),
            )?;
            // Integrating each positive rational term from 0 to 1 bounds the
            // difference from 5/2 by 3/10^200, far below either check tolerance.
            assert!(p.close(&result.values[0], &p.scale(&p.i(5), 1, 2), working - 10));
        }
        Ok(())
    }

    // Independent reference recurrence: first expand each rational matrix entry,
    // then match coefficients of y' = M y without clearing denominators.
    fn rational_series_taylor(
        system: &CompiledSystem,
        center: &C,
        values: &[C],
        order: usize,
    ) -> Result<Vec<Vec<C>>> {
        let p = system.p;
        let matrix = system
            .matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.series(p, center, order.saturating_sub(1)))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut y = vec![vec![p.zero(); values.len()]; order + 1];
        y[0] = values.to_vec();
        for k in 0..order {
            for (i, row) in matrix.iter().enumerate() {
                let mut sum = p.zero();
                for (j, entry) in row.iter().enumerate() {
                    for (l, a) in entry.iter().enumerate().take(k + 1) {
                        sum = p.add(&sum, &p.mul(a, &y[k - l][j]));
                    }
                }
                y[k + 1][i] = p.scale(&sum, 1, (k + 1) as i64);
            }
        }
        Ok(y)
    }

    #[test]
    fn cleared_recurrence_matches_rational_series_at_complex_centers() {
        let p = Precision::decimal(80).unwrap();
        let x = Atom::var(symbol!("recurrence_x"));
        let epsilon = Atom::var(symbol!("recurrence_eps"));
        let first = &x - Atom::num(1);
        let second = &x + Atom::num(2);
        let system = DifferentialSystem {
            variable: symbol!("recurrence_x"),
            matrix: vec![
                vec![
                    (x.clone().pow(3) + Atom::num(2)) / first.clone().pow(2),
                    &epsilon / &second,
                    Atom::new(),
                ],
                vec![Atom::num(1) / (&first * &second), Atom::new(), x.pow(20)],
                vec![Atom::new(); 3],
            ],
        }
        .compile(
            p,
            &ahash::HashMap::from_iter([(epsilon, p.rational(&Rational::from((1, 37))))]),
        )
        .unwrap();
        assert_eq!(system.polynomial_rows[0].denominator.len(), 4);
        assert_eq!(system.polynomial_rows[0].entries.len(), 2);
        assert!(system.polynomial_rows[2].entries.is_empty());
        let values = vec![p.i(2), p.complex(3, 1), p.i(7)];
        for center in [p.zero(), p.parse("0.375", "0.25").unwrap()] {
            for order in [0, 1, 12, 32] {
                let finite = system.taylor(&center, &values, order).unwrap();
                let reference = rational_series_taylor(&system, &center, &values, order).unwrap();
                for (a, b) in finite.iter().flatten().zip(reference.iter().flatten()) {
                    assert!(p.close(a, b, 65), "order {order}: {a} != {b}");
                }
                assert!(finite.iter().skip(1).all(|row| row[2] == p.zero()));
            }
        }
        assert!(matches!(
            system.taylor(&p.i(1), &values, 12),
            Err(Error::Numerical(_))
        ));
    }

    #[test]
    fn sparse_channels_match_independent_augmented_rational_recurrence() {
        let p = Precision::decimal(80).unwrap();
        let x = symbol!("channel_recurrence_x");
        // Noncommuting, nonzero A_0, A_1 and A_2 prevent a canonical-only
        // implementation or reversed epsilon indexing from passing this check.
        let matrices = [
            vec![
                vec![parse!("1/(channel_recurrence_x-1)"), Atom::one()],
                vec![Atom::new(), parse!("channel_recurrence_x")],
            ],
            vec![
                vec![Atom::new(), parse!("1/(channel_recurrence_x+2)^2")],
                vec![parse!("channel_recurrence_x^3"), Atom::new()],
            ],
            vec![
                vec![
                    parse!("(channel_recurrence_x+1)/(channel_recurrence_x-1)"),
                    Atom::new(),
                ],
                vec![Atom::one(), Atom::num(2)],
            ],
        ];
        let rows = (0..2)
            .map(|i| matrices.iter().flat_map(|m| m[i].clone()).collect())
            .collect::<Vec<Vec<Atom>>>();
        let compiled = compile_rows(x, &rows, p, &Default::default()).unwrap();
        let mut lifted = vec![vec![Atom::new(); 6]; 6];
        for e in 0..3 {
            for (shift, matrix) in matrices.iter().enumerate().take(e + 1) {
                for i in 0..2 {
                    for j in 0..2 {
                        lifted[e * 2 + i][(e - shift) * 2 + j] = matrix[i][j].clone();
                    }
                }
            }
        }
        let reference = DifferentialSystem {
            variable: x,
            matrix: lifted,
        }
        .compile(p, &Default::default())
        .unwrap();
        let values = vec![p.i(2), p.complex(3, 1), p.i(-1), p.i(7), p.zero(), p.i(11)];
        for center in [p.zero(), p.parse("0.375", "0.25").unwrap()] {
            for order in [0, 1, 6, 17] {
                let sparse = compiled
                    .taylor_channels(&center, &values, order, 3)
                    .unwrap();
                let dense = rational_series_taylor(&reference, &center, &values, order).unwrap();
                for (a, b) in sparse.iter().flatten().zip(dense.iter().flatten()) {
                    assert!(p.close(a, b, 65), "order {order}: {a} != {b}");
                }
            }
        }
        assert!(matches!(
            compiled.taylor_channels(&p.i(1), &values, 6, 3),
            Err(Error::Numerical(_))
        ));
        assert!(matches!(
            compiled.taylor_channels(&p.zero(), &values, 6, 2),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn polynomial_clearing_reports_degree_overflow() {
        let a: RationalPolynomial<IntegerRing, u16> = parse!("overflow_x^40000")
            .try_to_rational_polynomial(&Q, &Z, None)
            .unwrap();
        assert!(matches!(
            multiply_polynomials(&a.numerator, &a.numerator),
            Err(Error::Limit(_))
        ));
    }
}

#[cfg(test)]
#[path = "ode/source_tests.rs"]
mod source_tests;

#[cfg(test)]
#[path = "ode/endpoint_tests.rs"]
mod endpoint_tests;

#[cfg(test)]
#[path = "ode/exclusion_tests.rs"]
mod exclusion_tests;

#[cfg(test)]
#[path = "ode/prepared_tests.rs"]
mod prepared_tests;

#[cfg(test)]
#[path = "ode/high_order_tests.rs"]
mod high_order_tests;
