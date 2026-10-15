//! One Frobenius recurrence shared by numerical and exact coefficient owners.
use super::*;

pub(crate) struct RecurrenceColumn<S> {
    pub(crate) exponent: Atom,
    pub(crate) coefficients: Vec<Vec<Vec<S>>>,
}

/// Finite polynomial rows for x D_i(x) y'_i = sum_j P_ij(x) y_j.
/// The constant coefficient of each D_i is normalized to one.
pub(super) struct RecurrenceRow<S> {
    denominator: Vec<S>,
    entries: Vec<(usize, Vec<S>)>,
}

pub(super) trait CoefficientDomain {
    type Scalar: Clone + PartialEq;
    type Solver;
    fn zero(&self) -> Self::Scalar;
    fn integer(&self, value: i64) -> Self::Scalar;
    fn add(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    fn sub(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    fn mul(&self, a: &Self::Scalar, b: &Self::Scalar) -> Self::Scalar;
    fn neg(&self, a: &Self::Scalar) -> Self::Scalar;
    fn scale(&self, a: &Self::Scalar, n: i64, d: i64) -> Self::Scalar;
    fn evaluate(&self, expression: &Atom) -> Result<Self::Scalar>;
    fn series(
        &self,
        expression: &Atom,
        variable: Symbol,
        order: usize,
    ) -> Result<Vec<Self::Scalar>>;
    fn near_resonance(&self, difference: &Atom) -> Result<bool>;
    fn insignificant(&self, value: &Self::Scalar) -> bool;
    fn solver(
        &self,
        matrix: Vec<Vec<Self::Scalar>>,
        blocks: &[Vec<usize>],
        context: &RunContext,
    ) -> Result<Self::Solver>;
    fn solve(&self, solver: &Self::Solver, rhs: &[Self::Scalar]) -> Result<Vec<Self::Scalar>>;
    fn solve_singular(
        &self,
        matrix: Vec<Vec<Self::Scalar>>,
        rhs: Vec<Self::Scalar>,
        context: &RunContext,
    ) -> Result<Vec<Self::Scalar>>;
    fn polynomial_rows(
        &self,
        _system: &DifferentialSystem,
        _context: &RunContext,
    ) -> Result<Option<Vec<RecurrenceRow<Self::Scalar>>>> {
        Ok(None)
    }
    fn preflight(&self, _dimension: usize, _order: usize) -> Result<()> {
        Ok(())
    }
    fn preflight_matrix(&self, _dimension: usize) -> Result<()> {
        Ok(())
    }
    fn preflight_column(&self, _dimension: usize, _order: usize, _logs: usize) -> Result<()> {
        Ok(())
    }
    fn preflight_series(&self, _expression: &Atom, _variable: Symbol, _order: usize) -> Result<()> {
        Ok(())
    }
    fn check_rows(&self, _rows: &[Vec<Self::Scalar>]) -> Result<()> {
        Ok(())
    }
    fn check_value(&self, _value: &Self::Scalar) -> Result<()> {
        Ok(())
    }
}

struct NumericalDomain<'a> {
    p: Precision,
    values: &'a ahash::HashMap<Atom, C>,
    strict: bool,
    prepared_rows: &'a crate::ode::prepared::PreparedRows,
    streaming: Option<&'a super::streaming::StreamingAdmission>,
}
impl CoefficientDomain for NumericalDomain<'_> {
    type Scalar = C;
    type Solver = crate::numeric::BlockSolve;
    fn zero(&self) -> C {
        self.p.zero()
    }
    fn preflight_column(&self, dimension: usize, order: usize, logs: usize) -> Result<()> {
        if let Some(admission) = self.streaming {
            admission.column(dimension, order, logs)?;
        }
        Ok(())
    }
    fn preflight_matrix(&self, dimension: usize) -> Result<()> {
        if let Some(admission) = self.streaming {
            admission.matrix(dimension)?;
        }
        Ok(())
    }
    fn integer(&self, value: i64) -> C {
        self.p.i(value)
    }
    fn add(&self, a: &C, b: &C) -> C {
        self.p.add(a, b)
    }
    fn sub(&self, a: &C, b: &C) -> C {
        self.p.sub(a, b)
    }
    fn mul(&self, a: &C, b: &C) -> C {
        self.p.mul(a, b)
    }
    fn neg(&self, a: &C) -> C {
        self.p.neg(a)
    }
    fn scale(&self, a: &C, n: i64, d: i64) -> C {
        self.p.scale(a, n, d)
    }
    fn evaluate(&self, expression: &Atom) -> Result<C> {
        self.p.eval(expression, self.values)
    }
    fn series(&self, expression: &Atom, variable: Symbol, order: usize) -> Result<Vec<C>> {
        let rat: RationalPolynomial<IntegerRing, u16> = crate::family::encode_complex(expression)
            .try_to_rational_polynomial(&Q, &Z, None)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        let num = polynomial_coefficients(
            &rat.numerator.to_expression(),
            variable,
            self.p,
            self.values,
        )?;
        let den = polynomial_coefficients(
            &rat.denominator.to_expression(),
            variable,
            self.p,
            self.values,
        )?;
        quotient_series(self.p, &num, &den, order)
    }
    fn near_resonance(&self, difference: &Atom) -> Result<bool> {
        Ok(!self.strict
            && self.p.norm(&self.evaluate(difference)?) < self.p.tolerance(self.p.bits / 4))
    }
    fn insignificant(&self, value: &C) -> bool {
        if self.strict {
            value == &self.zero()
        } else {
            self.p.norm(value) < self.p.tolerance(self.p.bits / 4)
        }
    }
    fn solver(
        &self,
        matrix: Vec<Vec<C>>,
        blocks: &[Vec<usize>],
        _context: &RunContext,
    ) -> Result<Self::Solver> {
        crate::numeric::BlockSolve::new(self.p, matrix, blocks)
    }
    fn solve(&self, solver: &Self::Solver, rhs: &[C]) -> Result<Vec<C>> {
        Ok(solver.solve(rhs))
    }
    fn solve_singular(
        &self,
        matrix: Vec<Vec<C>>,
        rhs: Vec<C>,
        context: &RunContext,
    ) -> Result<Vec<C>> {
        solve_any(self.p, matrix, rhs, context, self.strict)
    }
    fn polynomial_rows(
        &self,
        _system: &DifferentialSystem,
        context: &RunContext,
    ) -> Result<Option<Vec<RecurrenceRow<C>>>> {
        // Reuse the ordinary Taylor solver's exact row-denominator owner.
        // Expanding x M(x) into a long geometric series would make the
        // Frobenius convolution quadratic in the requested expansion order.
        let rows = self
            .prepared_rows
            .compile_polynomial_rows(self.p, self.values, context)?
            .into_iter()
            .map(|row| {
                let divisor = &row.denominator[0];
                if divisor == &self.p.zero() {
                    return Err(Error::Numerical(
                        "Frobenius row denominator vanishes at expansion center".into(),
                    ));
                }
                Ok(RecurrenceRow {
                    denominator: row
                        .denominator
                        .iter()
                        .map(|v| self.p.div(v, divisor))
                        .collect(),
                    entries: row
                        .entries
                        .into_iter()
                        .map(|(column, values)| {
                            (
                                column,
                                values.iter().map(|v| self.p.div(v, divisor)).collect(),
                            )
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(rows))
    }
}

impl PreparedFrobenius {
    pub(super) fn numerical_recurrence(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
        strict: bool,
        prepared_rows: &crate::ode::prepared::PreparedRows,
    ) -> Result<FrobeniusBasis> {
        let domain = NumericalDomain {
            p,
            values,
            strict,
            prepared_rows,
            streaming: None,
        };
        let normal = self.recurrence_normal(&domain, order, context)?;
        let columns = self.transform_recurrence(&domain, normal, order, context)?;
        Ok(FrobeniusBasis {
            columns: columns
                .into_iter()
                .map(|c| FrobeniusColumn {
                    exponent: c.exponent,
                    coefficients: c.coefficients,
                })
                .collect(),
            precision: p,
        })
    }

    pub(super) fn numerical_recurrence_visit(
        &self,
        p: Precision,
        values: &ahash::HashMap<Atom, C>,
        order: usize,
        context: &RunContext,
        prepared_rows: &crate::ode::prepared::PreparedRows,
        admission: &super::streaming::StreamingAdmission,
        mut visitor: impl FnMut(FrobeniusColumn) -> Result<()>,
    ) -> Result<()> {
        let domain = NumericalDomain {
            p,
            values,
            strict: false,
            prepared_rows,
            streaming: Some(admission),
        };
        self.recurrence_normal_visit(&domain, order, context, false, |normal| {
            let column = if self.identity_transformation() {
                // The native column already has the physical coefficients and
                // exponent. Move it without allocating a duplicate long series.
                normal
            } else {
                let mut transformed =
                    self.transform_recurrence(&domain, vec![normal], order, context)?;
                transformed.pop().expect("one transformed Frobenius column")
            };
            visitor(FrobeniusColumn {
                exponent: column.exponent,
                coefficients: column.coefficients,
            })
        })
    }
}

impl PreparedFrobenius {
    pub(super) fn recurrence_normal<D: CoefficientDomain>(
        &self,
        domain: &D,
        order: usize,
        context: &RunContext,
    ) -> Result<Vec<RecurrenceColumn<D::Scalar>>> {
        let mut columns = Vec::new();
        self.recurrence_normal_visit(domain, order, context, true, |column| {
            columns.push(column);
            Ok(())
        })?;
        Ok(columns)
    }

    fn recurrence_normal_visit<D: CoefficientDomain>(
        &self,
        domain: &D,
        order: usize,
        context: &RunContext,
        retain_solvers: bool,
        mut visitor: impl FnMut(RecurrenceColumn<D::Scalar>) -> Result<()>,
    ) -> Result<()> {
        let n = self.normal.matrix.len();
        domain.preflight(n, order)?;
        let polynomial_rows = domain.polynomial_rows(&self.normal, context)?;
        let mut a = vec![
            vec![vec![domain.zero(); n]; n];
            if polynomial_rows.is_some() {
                1
            } else {
                order + 1
            }
        ];
        if let Some(rows) = &polynomial_rows {
            for (i, row) in rows.iter().enumerate() {
                for (column, coefficients) in &row.entries {
                    a[0][i][*column] = coefficients[0].clone();
                }
            }
        } else {
            for (i, row) in self.normal.matrix.iter().enumerate() {
                context.cancellation.check()?;
                for (j, expr) in row.iter().enumerate() {
                    domain.preflight_series(expr, self.source.variable, order)?;
                    let expr = (expr * Atom::var(self.source.variable)).together().cancel();
                    let series = domain.series(&expr, self.source.variable, order)?;
                    for k in 0..=order {
                        a[k][i][j] = series[k].clone();
                    }
                }
            }
        }
        let zero = domain.zero();
        let sparse = a
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .enumerate()
                    .flat_map(|(i, row)| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(j, value)| (value != &zero).then_some((i, j, value)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for space in &self.eigenspaces {
            context.cancellation.check()?;
            let lambda = &space.exponent;
            let exponent = domain.evaluate(lambda)?;
            let mut solvers = (0..=order)
                .map(|_| None)
                .collect::<Vec<Option<D::Solver>>>();
            for vector in &space.leading {
                context.cancellation.check()?;
                let mut leading = vector
                    .iter()
                    .map(|log| {
                        log.iter()
                            .map(|x| domain.evaluate(x))
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                while leading.len() > 1 && leading.last().unwrap().iter().all(|v| v == &zero) {
                    leading.pop();
                }
                let mut logs = leading.len();
                domain.preflight_column(n, order, logs)?;
                let mut coefficients = vec![leading];
                for k in 1..=order {
                    context.cancellation.check()?;
                    let mut m = a[0]
                        .iter()
                        .map(|r| r.iter().map(|v| domain.neg(v)).collect::<Vec<_>>())
                        .collect::<Vec<_>>();
                    for (i, row) in m.iter_mut().enumerate() {
                        row[i] =
                            domain.add(&row[i], &domain.add(&exponent, &domain.integer(k as i64)));
                    }
                    let mut resonance = false;
                    for other in self.eigenspaces.iter().map(|space| &space.exponent) {
                        let difference = (lambda + Atom::num(k as i64) - other).together().cancel();
                        if difference.is_zero() || domain.near_resonance(&difference)? {
                            resonance = true;
                            break;
                        }
                    }
                    let mut rhs = vec![vec![domain.zero(); n]; logs];
                    if let Some(rows) = &polynomial_rows {
                        for (i, row) in rows.iter().enumerate() {
                            for (column, polynomial) in &row.entries {
                                for (j, value) in polynomial.iter().enumerate().skip(1).take(k) {
                                    if value == &zero {
                                        continue;
                                    }
                                    for (l, previous) in coefficients[k - j].iter().enumerate() {
                                        if previous[*column] != zero {
                                            rhs[l][i] = domain.add(
                                                &rhs[l][i],
                                                &domain.mul(value, &previous[*column]),
                                            );
                                        }
                                    }
                                }
                            }
                            for (j, value) in row.denominator.iter().enumerate().skip(1).take(k) {
                                if value == &zero {
                                    continue;
                                }
                                let previous = &coefficients[k - j];
                                let weight = domain.add(&exponent, &domain.integer((k - j) as i64));
                                for (l, previous_row) in previous.iter().enumerate() {
                                    let mut derivative = domain.mul(&weight, &previous_row[i]);
                                    if let Some(next_log) = previous.get(l + 1) {
                                        derivative = domain.add(
                                            &derivative,
                                            &domain.scale(&next_log[i], (l + 1) as i64, 1),
                                        );
                                    }
                                    rhs[l][i] =
                                        domain.sub(&rhs[l][i], &domain.mul(value, &derivative));
                                }
                            }
                        }
                    } else {
                        for j in 1..=k {
                            for (l, previous) in coefficients[k - j].iter().enumerate() {
                                for &(i, column, value) in &sparse[j] {
                                    if previous[column] != zero {
                                        rhs[l][i] = domain
                                            .add(&rhs[l][i], &domain.mul(value, &previous[column]));
                                        domain.check_value(&rhs[l][i])?;
                                    }
                                }
                            }
                        }
                    }
                    let mut next = vec![vec![domain.zero(); n]; logs];
                    if resonance {
                        let mut result = None;
                        for extra in 0..=n {
                            context.cancellation.check()?;
                            let count = logs + extra;
                            let size = n.checked_mul(count).ok_or_else(|| {
                                Error::Limit("Frobenius resonant dimension overflow".into())
                            })?;
                            domain.preflight_matrix(size)?;
                            let mut system = vec![vec![domain.zero(); size]; size];
                            for l in 0..count {
                                for i in 0..n {
                                    for j in 0..n {
                                        system[l * n + i][l * n + j] = m[i][j].clone();
                                    }
                                    if l + 1 < count {
                                        system[l * n + i][(l + 1) * n + i] =
                                            domain.integer((l + 1) as i64);
                                    }
                                }
                            }
                            let mut extended = rhs.clone();
                            extended.resize(count, vec![domain.zero(); n]);
                            match domain.solve_singular(
                                system,
                                extended.into_iter().flatten().collect(),
                                context,
                            ) {
                                Ok(flat) => {
                                    result = Some(flat.chunks(n).map(|r| r.to_vec()).collect());
                                    break;
                                }
                                Err(Error::Numerical(_)) => continue,
                                Err(e) => return Err(e),
                            }
                        }
                        next = result.ok_or_else(|| {
                            Error::Numerical(
                                "resonant recurrence did not close with sufficient logarithms"
                                    .into(),
                            )
                        })?;
                    } else {
                        // Retain a factorization only if another leading vector
                        // will reuse it. At very high precision a long sequence
                        // of single-use factorizations is otherwise substantial.
                        let fresh_solver;
                        let solver = if !retain_solvers || space.leading.len() == 1 {
                            fresh_solver = domain.solver(m, &self.blocks, context)?;
                            &fresh_solver
                        } else {
                            if solvers[k].is_none() {
                                solvers[k] = Some(domain.solver(m, &self.blocks, context)?);
                            }
                            solvers[k].as_ref().unwrap()
                        };
                        for l in (0..logs).rev() {
                            if l + 1 < logs {
                                for i in 0..n {
                                    rhs[l][i] = domain.sub(
                                        &rhs[l][i],
                                        &domain.scale(&next[l + 1][i], (l + 1) as i64, 1),
                                    );
                                }
                            }
                            next[l] = domain.solve(solver, &rhs[l])?;
                        }
                    }
                    while next.len() > 1
                        && next.last().unwrap().iter().all(|v| domain.insignificant(v))
                    {
                        next.pop();
                    }
                    domain.check_rows(&next)?;
                    logs = logs.max(next.len());
                    domain.preflight_column(n, order, logs)?;
                    coefficients.push(next);
                }
                visitor(RecurrenceColumn {
                    exponent: lambda.clone(),
                    coefficients,
                })?;
            }
        }
        Ok(())
    }
}
impl PreparedFrobenius {
    pub(super) fn transform_recurrence<D: CoefficientDomain>(
        &self,
        domain: &D,
        mut columns: Vec<RecurrenceColumn<D::Scalar>>,
        order: usize,
        context: &RunContext,
    ) -> Result<Vec<RecurrenceColumn<D::Scalar>>> {
        let mut terms = Vec::new();
        let mut low = 0_i64;
        let mut high = 0_i64;
        for (i, row) in self.transformation.iter().enumerate() {
            context.cancellation.check()?;
            for (j, entry) in row.iter().enumerate() {
                if entry.is_zero() {
                    continue;
                }
                domain.preflight_series(entry, self.source.variable, order)?;
                let series = entry
                    .series(self.source.variable, 0, order as i64 + 1)
                    .map_err(|e| Error::Unsupported(e.to_string()))?;
                for (power, coefficient) in series.terms() {
                    let shift = power.to_string().parse::<i64>().map_err(|_| {
                        Error::Unsupported("noninteger basis transformation power".into())
                    })?;
                    low = low.min(shift);
                    high = high.max(shift);
                    terms.push((i, j, shift, domain.evaluate(coefficient)?));
                }
            }
        }
        if high - low > 256 {
            return Err(Error::Limit("normalization power span exceeds 256".into()));
        }
        domain.preflight(
            self.source.matrix.len(),
            order
                .checked_add((high - low) as usize)
                .ok_or_else(|| Error::Limit("Frobenius normalization order overflow".into()))?,
        )?;
        for column in &mut columns {
            context.cancellation.check()?;
            let mut transformed = vec![
                vec![vec![domain.zero(); self.source.matrix.len()]];
                column.coefficients.len() + (high - low) as usize
            ];
            for (k, logs) in column.coefficients.iter().enumerate() {
                for (l, row) in logs.iter().enumerate() {
                    for (i, j, shift, coefficient) in &terms {
                        let destination = &mut transformed[(k as i64 + shift - low) as usize];
                        destination.resize(
                            destination.len().max(l + 1),
                            vec![domain.zero(); self.source.matrix.len()],
                        );
                        destination[l][*i] =
                            domain.add(&destination[l][*i], &domain.mul(coefficient, &row[*j]));
                        domain.check_value(&destination[l][*i])?;
                    }
                }
            }
            column.exponent = (&column.exponent + Atom::num(low)).together().cancel();
            column.coefficients = transformed;
        }
        for column in &columns {
            for rows in &column.coefficients {
                domain.check_rows(rows)?;
            }
        }
        Ok(columns)
    }
}
