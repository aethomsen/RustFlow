//! Direct epsilon-coefficient transport of supplied differential equations.
//!
//! This independently implemented series recurrence shares the AMF continuation
//! engine. For A = sum epsilon^j A_j and Y = epsilon^leading sum epsilon^k Y_k,
//! it solves Y_k' = sum_{j=0}^k A_j Y_{k-j}, without sampled-epsilon fitting or
//! construction of a dense augmented matrix. Matrix coefficients must be
//! regular in epsilon at zero. A common Laurent prefactor of the solution is
//! allowed and is carried explicitly.
use crate::ode::{
    CompiledSystem, SeriesSystem, TaylorSegment, compile_rows, shift_polynomial, transport_series,
};
use crate::{
    BoundaryData, ComplexFloat as C, DifferentialSystem, Error, FlowDiagnostics, FlowOptions,
    Precision, Result, RunContext,
};
use symbolica::prelude::*;

/// Exact coefficient matrices A_0, ..., A_order. Omitted higher orders do not
/// affect solution coefficients through `order` when the boundary starts at
/// the declared common leading power.
#[derive(Clone, Debug)]
pub struct EpsilonSystem {
    pub variable: Symbol,
    pub matrices: Vec<Vec<Vec<Atom>>>,
}

/// Boundary coefficients indexed first by epsilon offset, then by component.
/// Components may have different leading powers by supplying exact zeros.
#[derive(Clone, Debug)]
pub struct EpsilonBoundary {
    pub point: C,
    pub leading: i32,
    pub coefficients: Vec<Vec<C>>,
}

#[derive(Clone, Debug)]
pub struct EpsilonSolution {
    pub point: C,
    pub leading: i32,
    pub coefficients: Vec<Vec<C>>,
    pub diagnostics: FlowDiagnostics,
    /// Present only when requested; epsilon offsets precede component indices
    /// in each segment's flattened coefficient vectors.
    pub segments: Vec<TaylorSegment>,
    /// Set by independent precision/order recomputation, not working precision.
    /// A value d uses the mixed coefficient scale 10^(-d) * max(1, |c|).
    /// It does not promise d significant digits for coefficients smaller than one.
    pub verified_digits: Option<u32>,
    pub comparison_errors: Vec<Vec<Float>>,
    /// Independently compared accepted endpoints, suitable for a physical
    /// boundary bank after mapping coordinates and accounting for input errors.
    pub checkpoints: Vec<VerifiedCheckpoint>,
}

#[derive(Clone, Debug)]
pub struct VerifiedCheckpoint {
    pub segment: usize,
    pub point: C,
    pub coefficients: Vec<Vec<C>>,
    pub comparison_errors: Vec<Vec<Float>>,
}

#[derive(Clone, Debug)]
pub struct CompiledEpsilonSystem {
    rows: CompiledSystem,
    count: usize,
    // Established on the exact input, never inferred from rounded coefficients.
    strictly_positive_epsilon: bool,
}

impl EpsilonSystem {
    pub fn validate(&self) -> Result<()> {
        let n = self.matrices.first().map_or(0, Vec::len);
        if n == 0
            || self
                .matrices
                .iter()
                .any(|m| m.len() != n || m.iter().any(|r| r.len() != n))
        {
            return Err(Error::InvalidInput(
                "epsilon matrices must be nonempty and square with equal dimensions".into(),
            ));
        }
        if self.matrices.len() > 1024 {
            return Err(Error::Limit("at most 1024 epsilon orders supported".into()));
        }
        Ok(())
    }

    /// Expand a rational epsilon dependence exactly. Negative epsilon powers
    /// in the matrix require a basis change and are rejected explicitly.
    pub fn from_differential_system(
        system: &DifferentialSystem,
        epsilon: Symbol,
        order: usize,
    ) -> Result<Self> {
        system.validate()?;
        if epsilon == system.variable {
            return Err(Error::InvalidInput(
                "epsilon and path parameter must differ".into(),
            ));
        }
        if order >= 1024 {
            return Err(Error::Limit("at most 1024 epsilon orders supported".into()));
        }
        let n = system.matrix.len();
        let mut matrices = vec![vec![vec![Atom::new(); n]; n]; order + 1];
        for (i, row) in system.matrix.iter().enumerate() {
            for (j, a) in row.iter().enumerate() {
                // In particular, do not accept floating coefficients or
                // fractional epsilon powers as an ordinary Taylor series.
                let a = crate::family::encode_complex(a);
                let _: RationalPolynomial<IntegerRing, u16> =
                    a.try_to_rational_polynomial(&Q, &Z, None).map_err(|e| {
                        Error::Unsupported(format!("rational epsilon system required: {e}"))
                    })?;
                let expansion = a
                    .series(epsilon, 0, order as i64 + 1)
                    .map_err(|e| Error::Unsupported(e.to_string()))?;
                if !expansion.is_zero() && expansion.get_trailing_exponent() < 0 {
                    return Err(Error::Unsupported(
                        "matrix has an epsilon pole; supply an epsilon-regular basis".into(),
                    ));
                }
                for (k, matrix) in matrices.iter_mut().enumerate() {
                    matrix[i][j] = expansion
                        .coefficient(Rational::from(k as i64))
                        .unwrap_or_default()
                        .together()
                        .cancel();
                }
            }
        }
        Ok(Self {
            variable: system.variable,
            matrices,
        })
    }

    pub fn compile(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
    ) -> Result<CompiledEpsilonSystem> {
        self.validate()?;
        // One denominator LCM per physical row across all epsilon orders.
        // Reuse these finite polynomials at every coefficient order; there is
        // no repeated (n*orders)^2 matrix allocation or pole finding.
        let n = self.matrices[0].len();
        let rows = (0..n)
            .map(|i| {
                self.matrices
                    .iter()
                    .flat_map(|matrix| matrix[i].iter().cloned())
                    .collect()
            })
            .collect::<Vec<_>>();
        Ok(CompiledEpsilonSystem {
            rows: compile_rows(self.variable, &rows, p, values)?,
            count: self.matrices.len(),
            strictly_positive_epsilon: self.matrices[0].iter().flatten().all(Atom::is_zero),
        })
    }
}

impl CompiledEpsilonSystem {
    pub(crate) fn fundamental_boundary_errors(
        &self,
        segments: &[TaylorSegment],
        input_errors: &[Float],
        weights: &[Float],
        order: usize,
        context: &RunContext,
    ) -> Result<crate::ode::fundamental::FundamentalErrors> {
        context.cancellation.check()?;
        if self.count != 1 {
            return Err(Error::Unsupported(
                "fundamental boundary proof currently supports ordinary rational systems only; epsilon hierarchies retain scalar bounds".into(),
            ));
        }
        crate::ode::fundamental::propagate(
            &self.rows,
            segments,
            input_errors,
            weights,
            order,
            context,
        )
    }

    pub fn poles(&self) -> &[C] {
        &self.rows.poles
    }

    pub(crate) fn singularity_polynomials(&self) -> &[Atom] {
        &self.rows.pole_polynomials
    }

    pub fn plan_path(&self, start: &C, end: &C, side: i64) -> Result<Vec<C>> {
        crate::ode::plan_path(self.rows.p, self.poles(), start, end, side)
    }

    pub fn transport(
        &self,
        boundary: &EpsilonBoundary,
        waypoints: &[C],
        options: &FlowOptions,
        context: &RunContext,
        save_segments: bool,
    ) -> Result<EpsilonSolution> {
        let n = self.rows.dimension();
        if boundary.coefficients.len() != self.count
            || boundary.coefficients.iter().any(|r| r.len() != n)
        {
            return Err(Error::InvalidInput("epsilon boundary dimensions".into()));
        }
        boundary
            .leading
            .checked_add(self.count as i32 - 1)
            .ok_or_else(|| Error::InvalidInput("epsilon power range overflows".into()))?;
        let flat = BoundaryData {
            point: boundary.point.clone(),
            values: boundary.coefficients.iter().flatten().cloned().collect(),
        };
        let mut segments = Vec::new();
        let result = transport_series(
            self,
            &flat,
            waypoints,
            options,
            context,
            save_segments.then_some(&mut segments),
        )?;
        Ok(EpsilonSolution {
            point: result.point,
            leading: boundary.leading,
            coefficients: result.values.chunks(n).map(<[C]>::to_vec).collect(),
            diagnostics: result.diagnostics,
            segments,
            verified_digits: None,
            comparison_errors: Vec::new(),
            checkpoints: Vec::new(),
        })
    }

    /// A norm-based estimate of amplification of boundary uncertainty along a
    /// regular segment. An exact zero epsilon0 block permits a finite Dyson
    /// bound; otherwise the estimate uses the exponential Gronwall bound.
    /// Numerators are bounded above on each disk and common
    /// denominators below by the triangle inequality; subdivide if that lower
    /// bound is inconclusive. MPFR rounding and initial error estimates prevent
    /// interpreting this as an interval-arithmetic proof.
    pub fn error_amplification(&self, start: &C, end: &C) -> Result<Float> {
        self.error_amplification_weighted(start, end, &vec![self.rows.p.real(1); self.dimension()])
    }

    /// Estimate amplification in the fixed weighted infinity norm
    /// `max_i |error_i| / weights[i]`. Weights follow the solution's flattened
    /// order: epsilon offset first, then physical component. They must be
    /// positive and finite. The bound uses the row sums of `W^-1 A W` for the
    /// triangular epsilon hierarchy, without constructing its dense matrix.
    /// The same rounding and initial-evidence qualifications as
    /// [`Self::error_amplification`] apply.
    pub fn error_amplification_weighted(
        &self,
        start: &C,
        end: &C,
        weights: &[Float],
    ) -> Result<Float> {
        amplification_from_integral(
            self.rows.p,
            &self.error_norm_integral_weighted(start, end, weights)?,
            self.epsilon_product_limit(),
        )
    }

    /// Every product of this many connection matrices vanishes exactly.
    /// Physical blocks may be noncommuting; each factor raises epsilon order.
    pub(crate) fn epsilon_product_limit(&self) -> Option<usize> {
        self.strictly_positive_epsilon.then_some(self.count)
    }

    pub(crate) fn error_norm_integral_weighted(
        &self,
        start: &C,
        end: &C,
        weights: &[Float],
    ) -> Result<Float> {
        let p = self.rows.p;
        if weights.len() != self.dimension()
            || weights.iter().any(|w| !w.is_finite() || *w <= p.real(0))
            || !start.re.is_finite()
            || !start.im.is_finite()
            || !end.re.is_finite()
            || !end.im.is_finite()
        {
            return Err(Error::InvalidInput(
                "error amplification requires finite endpoints and one positive finite weight per epsilon coefficient".into(),
            ));
        }
        let n = self.rows.dimension();
        integrate_matrix_norm(p, start, end, |start, radius| {
            let mut norm = p.real(0);
            for (i, row) in self.rows.polynomial_rows.iter().enumerate() {
                let denominator =
                    shift_polynomial(p, &row.denominator, start, row.denominator.len() - 1);
                let mut lower = p.norm(&denominator[0]);
                let mut power = radius.clone();
                for a in denominator.iter().skip(1) {
                    lower -= p.norm(a) * &power;
                    power *= radius;
                }
                if !lower.is_finite() {
                    return Err(Error::Accuracy(
                        "error-bound denominator exceeds numerical range".into(),
                    ));
                }
                if lower <= p.real(0) {
                    return Ok(None);
                }
                let mut row_norms = vec![p.real(0); self.count];
                for (column, coefficients) in &row.entries {
                    let shifted = shift_polynomial(p, coefficients, start, coefficients.len() - 1);
                    let mut upper = p.real(0);
                    let mut power = p.real(1);
                    for a in shifted {
                        upper += p.norm(&a) * &power;
                        power *= radius;
                    }
                    let bound = upper / &lower;
                    let shift = column / n;
                    let j = column % n;
                    for (order, row_norm) in row_norms.iter_mut().enumerate().skip(shift) {
                        *row_norm += bound.clone() * &weights[(order - shift) * n + j]
                            / &weights[order * n + i];
                    }
                }
                for row_norm in row_norms {
                    if !row_norm.is_finite() {
                        return Err(Error::Accuracy(
                            "weighted error-bound matrix norm exceeds numerical range".into(),
                        ));
                    }
                    if row_norm > norm {
                        norm = row_norm;
                    }
                }
            }
            Ok(Some(norm))
        })
    }
}

/// Common disk-subdivision estimate of the integrated weighted matrix norm.
/// This uses finite-precision norms, not directed interval arithmetic.
pub(crate) fn integrate_matrix_norm(
    p: Precision,
    start: &C,
    end: &C,
    mut matrix_norm: impl FnMut(&C, &Float) -> Result<Option<Float>>,
) -> Result<Float> {
    let mut intervals = vec![(start.clone(), end.clone(), 0usize)];
    let mut exponent = p.real(0);
    while let Some((start, end, depth)) = intervals.pop() {
        let radius = p.norm(&p.sub(&end, &start));
        if !radius.is_finite() {
            return Err(Error::Accuracy(
                "error-bound segment length exceeds numerical range".into(),
            ));
        }
        if radius == p.real(0) {
            continue;
        }
        if let Some(norm) = matrix_norm(&start, &radius)? {
            if !norm.is_finite() || norm < p.real(0) {
                return Err(Error::Accuracy(
                    "error-bound matrix norm exceeds numerical range".into(),
                ));
            }
            exponent += radius * norm;
        } else {
            if depth >= 32 {
                return Err(Error::Accuracy(
                    "cannot bound cached boundary error along this path".into(),
                ));
            }
            let middle = p.scale(&p.add(&start, &end), 1, 2);
            if middle == start || middle == end {
                return Err(Error::Accuracy(
                    "error-bound subdivision lost to rounding".into(),
                ));
            }
            intervals.push((middle.clone(), end, depth + 1));
            intervals.push((start, middle, depth + 1));
        }
    }
    Ok(exponent)
}

/// Bound the time-ordered exponential using its integrated norm. If every
/// product of N connection matrices vanishes, the Dyson series stops at N-1
/// even for noncommuting physical blocks. A fixed positive diagonal scaling
/// preserves this structure. The caller must accumulate the norm over the
/// whole path, rather than multiply truncated scalar polynomials per segment.
pub(crate) fn amplification_from_integral(
    p: Precision,
    integral: &Float,
    product_limit: Option<usize>,
) -> Result<Float> {
    if !integral.is_finite() || *integral < p.real(0) || product_limit == Some(0) {
        return Err(Error::Accuracy(
            "invalid integrated error-growth bound".into(),
        ));
    }
    let amplification = if let Some(count) = product_limit {
        let mut term = p.real(1);
        let mut sum = term.clone();
        for order in 1..count {
            term *= integral;
            term /= p.real(order as i64);
            sum += &term;
        }
        sum
    } else {
        p.exp(&C::new(integral.clone(), p.real(0))).re
    };
    if !amplification.is_finite() {
        return Err(Error::Accuracy(
            "boundary-error amplification exceeds numerical range".into(),
        ));
    }
    Ok(amplification)
}

/// A full polynomial at a disk center. At exact zero the coefficients are
/// already in the requested coordinate; preserve fixed-precision rounding
/// without running a quadratic identity shift. Nonfinite inputs keep the
/// existing shift/error behavior.
fn disk_coefficients<'a>(
    p: Precision,
    coefficients: &'a [C],
    center: &C,
) -> std::borrow::Cow<'a, [C]> {
    if !coefficients.is_empty() && *center == p.zero() && coefficients.iter().all(|a| p.finite(a)) {
        if coefficients
            .iter()
            .all(|a| a.re.as_raw().prec() == p.bits && a.im.as_raw().prec() == p.bits)
        {
            std::borrow::Cow::Borrowed(coefficients)
        } else {
            // The old zero shift rounds every finite coefficient to p.bits.
            std::borrow::Cow::Owned(coefficients.iter().map(|a| p.round(a)).collect())
        }
    } else {
        std::borrow::Cow::Owned(shift_polynomial(
            p,
            coefficients,
            center,
            coefficients.len() - 1,
        ))
    }
}

pub(crate) fn rational_disk_bound(
    p: Precision,
    value: &crate::ode::NumericRational,
    center: &C,
    radius: &Float,
) -> Result<Option<Float>> {
    let denominator = disk_coefficients(p, &value.denominator, center);
    let mut lower = p.norm(&denominator[0]);
    let mut power = radius.clone();
    for a in denominator.iter().skip(1) {
        lower -= p.norm(a) * &power;
        power *= radius;
    }
    if !lower.is_finite() {
        return Err(Error::Accuracy(
            "error-bound denominator exceeds numerical range".into(),
        ));
    }
    if lower <= p.real(0) {
        return Ok(None);
    }
    let numerator = disk_coefficients(p, &value.numerator, center);
    let mut upper = p.real(0);
    let mut power = p.real(1);
    for a in numerator.iter() {
        upper += p.norm(a) * &power;
        power *= radius;
    }
    let bound = upper / lower;
    if !bound.is_finite() {
        return Err(Error::Accuracy(
            "error-bound numerator exceeds numerical range".into(),
        ));
    }
    Ok(Some(bound))
}

impl SeriesSystem for CompiledEpsilonSystem {
    type State = ();
    type Chart = crate::ode::residual::RationalResidualChart;
    fn initial_state(&self, _: &BoundaryData) -> Result<()> {
        Ok(())
    }
    fn accepted_state(&self, _: &Self::Chart, _: &C, _: &Float) -> Result<Option<()>> {
        Ok(Some(()))
    }
    fn precision(&self) -> Precision {
        self.rows.p
    }
    fn dimension(&self) -> usize {
        self.rows.dimension() * self.count
    }
    fn poles(&self) -> &[C] {
        self.poles()
    }
    fn extended_order_workspace_bytes(&self, order: usize) -> Result<u128> {
        Ok(self.rows.taylor_workspace_bytes(order, self.count))
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
        let coefficients = self
            .rows
            .taylor_channels(center, values, order, self.count)?;
        let chart = crate::ode::residual::RationalResidualChart::new_with_options(
            self.rows.p,
            &self.rows.exact_source_rows,
            center,
            &coefficients,
            self.count,
            options,
            context,
        )?;
        Ok((coefficients, chart))
    }
    fn mapped_chart(
        &self,
        coordinate: &crate::local_coordinates::TaylorCoordinate,
        center: &C,
        values: &[C],
        order: usize,
        state: &(),
    ) -> Result<crate::ode::MappedTaylorChart<Self::Chart>> {
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
        coordinate: &crate::local_coordinates::TaylorCoordinate,
        _: &C,
        values: &[C],
        order: usize,
        _: &(),
        options: &FlowOptions,
        context: &RunContext,
    ) -> Result<crate::ode::MappedTaylorChart<Self::Chart>> {
        context.cancellation.check()?;
        let mapped = self.rows.in_coordinate(coordinate)?;
        let coefficients =
            mapped.taylor_channels(&self.rows.p.zero(), values, order, self.count)?;
        let chart = crate::ode::residual::RationalResidualChart::new_with_options(
            self.rows.p,
            &mapped.exact_source_rows,
            &self.rows.p.zero(),
            &coefficients,
            self.count,
            options,
            context,
        )?;
        Ok((coefficients, chart, mapped.poles))
    }
    fn pade_candidate(
        &self,
        coordinate: &crate::local_coordinates::TaylorCoordinate,
        center: &C,
        coefficients: &[Vec<C>],
        options: &crate::PadeOptions,
        context: &RunContext,
    ) -> Result<Option<crate::ode::pade::RationalCandidate>> {
        let mapped;
        let (source, center) = if matches!(
            coordinate,
            crate::local_coordinates::TaylorCoordinate::Identity
        ) {
            (&self.rows, center.clone())
        } else {
            mapped = self.rows.in_coordinate(coordinate)?;
            (&mapped, self.rows.p.zero())
        };
        crate::ode::pade::RationalCandidate::build(
            self.rows.p,
            &source.exact_source_rows,
            &center,
            coefficients,
            self.count,
            options,
            context,
        )
        .map(Some)
    }
    fn whole_segment_residual(&self, chart: &Self::Chart, step: &C) -> Result<Option<Vec<Float>>> {
        chart.defect_bounds(self.rows.p, step).map(Some)
    }

    fn whole_segment_residual_with_budget(
        &self,
        chart: &Self::Chart,
        step: &C,
        values: &[C],
        tolerance: &Float,
    ) -> Result<Option<Vec<Float>>> {
        chart
            .defect_bounds_with_budget(self.rows.p, step, values, tolerance)
            .map(Some)
    }
    fn rhs(&self, point: &C, values: &[C], _: &Self::Chart) -> Result<Vec<C>> {
        let p = self.rows.p;
        let n = self.rows.dimension();
        let mut rhs = vec![p.zero(); self.dimension()];
        // Evaluate each nonzero rational coefficient only once, sharing it
        // across epsilon orders when checking the differential-equation defect.
        for (i, row) in self.rows.polynomial_rows.iter().enumerate() {
            let denominator = shift_polynomial(p, &row.denominator, point, 0).remove(0);
            if denominator == p.zero() {
                return Err(Error::Numerical("path reached a pole".into()));
            }
            for (column, polynomial) in &row.entries {
                let a = p.div(&shift_polynomial(p, polynomial, point, 0)[0], &denominator);
                let shift = column / n;
                for e in shift..self.count {
                    rhs[e * n + i] = p.add(
                        &rhs[e * n + i],
                        &p.mul(&a, &values[(e - shift) * n + column % n]),
                    );
                }
            }
        }
        Ok(rhs)
    }
}

impl EpsilonSolution {
    /// Evaluate only when the retained path identifies one value at the point.
    /// Self-intersections carrying different monodromies are rejected.
    pub fn evaluate_path(&self, point: &C) -> Result<Vec<Vec<C>>> {
        Ok(self.evaluate_path_conditioned(point)?.0)
    }
    fn evaluate_path_conditioned(&self, point: &C) -> Result<(Vec<Vec<C>>, u32)> {
        let p = Precision {
            bits: self.diagnostics.working_bits,
        };
        let mut value: Option<Vec<Vec<C>>> = None;
        let mut conditioning_digits = u32::MAX;
        for i in 0..self.segments.len() {
            match self.evaluate_segment_conditioned(i, point) {
                Err(error @ Error::InsufficientPrecision { .. }) => return Err(error),
                Err(_) => continue,
                Ok((candidate, checked)) => {
                    conditioning_digits = conditioning_digits.min(checked);
                    if let Some(previous) = &value {
                        if !candidate
                            .iter()
                            .flatten()
                            .zip(previous.iter().flatten())
                            .all(|(a, b)| p.close(a, b, 30))
                        {
                            return Err(Error::InvalidInput(
                            "saved path has multiple branch values at this point; select a segment"
                                .into(),
                        ));
                        }
                    } else {
                        value = Some(candidate);
                    }
                }
            }
        }
        value
            .map(|v| (v, conditioning_digits))
            .ok_or_else(|| Error::InvalidInput("point is outside the saved path".into()))
    }
    /// Evaluate a saved segment only on its traversed straight interval. This
    /// avoids silently switching analytic branches at overlapping disks. At a
    /// self-intersection the caller chooses the desired segment explicitly.
    pub fn evaluate_segment(&self, index: usize, point: &C) -> Result<Vec<Vec<C>>> {
        Ok(self.evaluate_segment_conditioned(index, point)?.0)
    }
    fn evaluate_segment_conditioned(&self, index: usize, point: &C) -> Result<(Vec<Vec<C>>, u32)> {
        let segment = self
            .segments
            .get(index)
            .ok_or_else(|| Error::InvalidInput("unknown saved segment".into()))?;
        let p = Precision {
            bits: segment.working_bits,
        };
        let n = self.coefficients.first().map_or(0, Vec::len);
        if n == 0
            || segment.center == segment.end
            || segment.coefficients.is_empty()
            || segment
                .coefficients
                .iter()
                .any(|r| r.len() != n * self.coefficients.len())
        {
            return Err(Error::InvalidInput("malformed saved segment".into()));
        }
        if !p.finite(point) {
            return Err(Error::InvalidInput("nonfinite evaluation point".into()));
        }
        let span = p.sub(&segment.end, &segment.center);
        if !p.finite(&span) || span == p.zero() {
            return Err(Error::InvalidInput(
                "saved segment has a nonfinite or zero numerical span".into(),
            ));
        }
        let t = p.div(&p.sub(point, &segment.center), &span);
        if !p.finite(&t) {
            return Err(Error::InvalidInput(
                "saved segment interval coordinate is nonfinite".into(),
            ));
        }
        let tolerance = p.tolerance((p.bits / 4).min(30));
        if t.im > tolerance
            || t.im < -tolerance.clone()
            || t.re < -tolerance.clone()
            || t.re > p.real(1) + tolerance
        {
            return Err(Error::InvalidInput(
                "point is outside the saved segment interval".into(),
            ));
        }
        let local = segment.coordinate.local_point(p, &segment.center, point)?;
        let (flat, checked) = segment.evaluate_local(&local)?;
        Ok((flat.chunks(n).map(<[C]>::to_vec).collect(), checked))
    }
}

/// Recompute the complete transport with increasing precision and series order.
/// The provider must supply fresh boundary coefficients at the requested
/// precision; any uncertainty in those inputs also limits the result. Exact
/// waypoints avoid rounding a path through binary64.
pub fn transport_epsilon(
    system: &EpsilonSystem,
    boundary_provider: impl Fn(Precision) -> Result<EpsilonBoundary>,
    waypoints: &[Atom],
    options: &FlowOptions,
    context: &RunContext,
    save_segments: bool,
) -> Result<EpsilonSolution> {
    refine_epsilon_transport(options, context, save_segments, |p, refined| {
        let compiled = system.compile(p, &Default::default())?;
        let points = waypoints
            .iter()
            .map(|a| p.eval(a, &Default::default()))
            .collect::<Result<Vec<_>>>()?;
        let boundary = boundary_provider(p)?;
        let coordinate_digits = crate::ode::conditioning::exact_waypoint_conditioning(
            p,
            &boundary.point,
            waypoints,
            &points,
            options.digits,
        )?;
        let mut result = compiled.transport(&boundary, &points, refined, context, save_segments)?;
        result.diagnostics.conditioning_digits = Some(
            result
                .diagnostics
                .conditioning_digits
                .map_or(coordinate_digits, |old| old.min(coordinate_digits)),
        );
        Ok(result)
    })
}

pub(crate) fn refine_epsilon_transport(
    options: &FlowOptions,
    context: &RunContext,
    save_segments: bool,
    mut run: impl FnMut(Precision, &FlowOptions) -> Result<EpsilonSolution>,
) -> Result<EpsilonSolution> {
    Ok(refine_epsilon_transport_with_metadata(
        options,
        context,
        save_segments,
        |p, o| Ok((run(p, o)?, ())),
        |_, _, _, _, _| Ok(true),
    )?
    .0)
}

/// One refinement engine, with additional independently checked metadata.
pub(crate) fn refine_epsilon_transport_with_metadata<M>(
    options: &FlowOptions,
    context: &RunContext,
    save_segments: bool,
    mut run: impl FnMut(Precision, &FlowOptions) -> Result<(EpsilonSolution, M)>,
    mut compatible: impl FnMut(&EpsilonSolution, &M, &EpsilonSolution, &M, usize) -> Result<bool>,
) -> Result<(EpsilonSolution, M)> {
    options.validate()?;
    let mut previous: Option<(EpsilonSolution, M)> = None;
    let mut working_digits = options
        .digits
        .checked_add(options.guard_digits)
        .ok_or_else(|| Error::Limit("working precision overflow".into()))?;
    let mut last_precision_failure = None;
    for attempt in 0..=options.max_precision_attempts {
        context.cancellation.check()?;
        let mut refined = options.clone();
        refined.guard_digits = working_digits - options.digits;
        refined.series_order = refined
            .series_order
            .checked_add(32 * attempt)
            .ok_or_else(|| Error::Limit("series order overflow".into()))?;
        refined.refine_rational_order(attempt);
        refined.validate()?;
        let p = Precision::decimal(refined.digits + refined.guard_digits)?;
        let (mut result, metadata) = match run(p, &refined) {
            Ok(result) => result,
            Err(Error::InsufficientPrecision {
                minimum_bits,
                context,
            }) => {
                working_digits = Precision::refinement_digits(working_digits, minimum_bits)?;
                previous = None;
                last_precision_failure = Some((minimum_bits, context));
                continue;
            }
            Err(error) => return Err(error),
        };
        last_precision_failure = None;
        working_digits = working_digits
            .checked_add(20)
            .ok_or_else(|| Error::Limit("working precision overflow".into()))?;
        if let Some((old, old_metadata)) = &previous {
            if old.leading != result.leading {
                return Err(Error::InvalidInput(
                    "boundary provider changed the leading epsilon power".into(),
                ));
            }
            result.diagnostics.conditioning_digits = match (
                old.diagnostics.conditioning_digits,
                result.diagnostics.conditioning_digits,
            ) {
                (Some(a), Some(b)) => Some(a.min(b)),
                _ => None,
            };
            result.comparison_errors = result
                .coefficients
                .iter()
                .zip(&old.coefficients)
                .map(|(new, old)| {
                    new.iter()
                        .zip(old)
                        .map(|(a, b)| p.norm(&p.sub(a, b)))
                        .collect()
                })
                .collect();
            if result
                .coefficients
                .iter()
                .flatten()
                .zip(old.coefficients.iter().flatten())
                .all(|(a, b)| p.close(a, b, options.digits))
            {
                result.verified_digits = Some(options.digits);
                if save_segments {
                    for (index, segment) in result.segments.iter().enumerate() {
                        if let Ok((reference, old_checked)) =
                            old.evaluate_path_conditioned(&segment.end)
                            && compatible(old, old_metadata, &result, &metadata, index)?
                        {
                            let (values, new_checked) =
                                match result.evaluate_segment_conditioned(index, &segment.end) {
                                    Ok(v) => v,
                                    Err(Error::InsufficientPrecision { .. }) => continue,
                                    Err(error) => return Err(error),
                                };
                            let checked = old_checked.min(new_checked);
                            result.diagnostics.conditioning_digits = Some(
                                result
                                    .diagnostics
                                    .conditioning_digits
                                    .map_or(checked, |old| old.min(checked)),
                            );
                            if values
                                .iter()
                                .flatten()
                                .zip(reference.iter().flatten())
                                .all(|(a, b)| p.close(a, b, options.digits))
                            {
                                let errors = values
                                    .iter()
                                    .zip(reference)
                                    .map(|(a, b)| {
                                        a.iter()
                                            .zip(b)
                                            .map(|(a, b)| p.norm(&p.sub(a, &b)))
                                            .collect()
                                    })
                                    .collect();
                                result.checkpoints.push(VerifiedCheckpoint {
                                    segment: index,
                                    point: segment.end.clone(),
                                    coefficients: values,
                                    comparison_errors: errors,
                                });
                            }
                        }
                    }
                }
                return Ok((result, metadata));
            }
        }
        previous = Some((result, metadata));
    }
    if let Some((minimum_bits, context)) = last_precision_failure {
        return Err(Error::InsufficientPrecision {
            minimum_bits,
            context: format!(
                "precision retry budget exhausted without two successful profiles; last diagnostic: {context}"
            ),
        });
    }
    Err(Error::Accuracy(
        "epsilon transport did not stabilize under precision/order refinement".into(),
    ))
}

#[cfg(test)]
#[path = "diffexp/uncertainty_tests.rs"]
mod uncertainty_tests;
