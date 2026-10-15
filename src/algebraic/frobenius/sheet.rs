//! Constant sheet actions recovered from a complete exact formal solution space.
use super::*;
use crate::asymptotic::AsymptoticSelector;
use crate::frobenius::exact::RecurrenceColumn;
use crate::frobenius::exact::{self, Gaussian};
use crate::frobenius::{ExactFrobeniusBasis, ExactFrobeniusLimits};

fn zero() -> Gaussian {
    Gaussian::new(Rational::zero(), Rational::zero())
}
fn one() -> Gaussian {
    Gaussian::new(Rational::one(), Rational::zero())
}

impl PreparedAlgebraicFrobenius {
    /// After the last positive integral indicial difference all future
    /// recurrence matrices are invertible. Prefix rank then certifies that
    /// the N constructed exact formal solutions are a complete basis.
    pub(super) fn complete_order(
        &self,
        order: usize,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<bool> {
        let exponents = self
            .prepared
            .exponents()
            .map(|a| exact::gaussian_bounded(a, limits, context))
            .collect::<Result<Vec<_>>>()?;
        if exponents.iter().any(|a| !a.im.is_zero()) {
            return Err(Error::Unsupported(
                "exact sheet completion requires real rational indicial exponents".into(),
            ));
        }
        for a in &exponents {
            for b in &exponents {
                context.cancellation.check()?;
                let difference = &a.re - &b.re;
                if difference.is_integer() && difference > order {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sheet_equations(
        &self,
        basis: &ExactFrobeniusBasis,
        matching: &Atom,
        seeds: &BTreeMap<Symbol, RootSeed>,
        winding: i32,
        p: Precision,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<Option<Vec<Vec<Gaussian>>>> {
        let n = basis.dimension();
        if !self.complete_order(basis.known_order, limits, context)? {
            return Ok(None);
        }
        let mut actions: Vec<Vec<Vec<Gaussian>>> = Vec::new();
        let mut equations = Vec::new();
        // Each subgroup extension doubles the monomial vector in rational_lift;
        // indices 1,2,4,... are precisely its independent XOR generators.
        for index in (0..usize::BITS)
            .map(|i| 1_usize << i)
            .take_while(|&i| i < self.lifted.monomials.len())
        {
            context.cancellation.check()?;
            let mask = self.lifted.monomials[index];
            self.check_covariance(mask, limits, context)?;
            let (v, a, h) = super::sheet_germ::selected_germ(
                &self.lifted,
                mask,
                matching,
                seeds,
                winding,
                p,
                limits,
                context,
            )?;
            let transformed = self.apply_sheet(basis, mask, v, &a, &h, limits, context)?;
            let Some(action) = recover_action(basis, &transformed, limits, context)? else {
                return Ok(None);
            };
            let square = exact::multiply(action.clone(), action.clone(), limits, context)?;
            for (i, row) in square.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    if *value != if i == j { one() } else { zero() } {
                        return Err(Error::Numerical(
                            "exact recovered root sheet action is not involutory".into(),
                        ));
                    }
                }
            }
            for other in &actions {
                if exact::multiply(action.clone(), other.clone(), limits, context)?
                    != exact::multiply(other.clone(), action.clone(), limits, context)?
                {
                    return Err(Error::Numerical(
                        "exact recovered root sheet actions do not commute".into(),
                    ));
                }
            }
            if equations
                .len()
                .checked_add(n)
                .and_then(|rows| rows.checked_mul(n + 1))
                .is_none_or(|cells| cells > limits.max_scalar_cells)
            {
                return Err(Error::Limit(
                    "sheet equation allocation exceeds limit".into(),
                ));
            }
            for (i, source) in action.iter().enumerate() {
                let mut row = source.clone();
                row[i] = &row[i] - &one();
                row.push(zero());
                equations.push(row);
            }
            actions.push(action);
        }
        Ok(Some(equations))
    }

    fn check_covariance(
        &self,
        mask: usize,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<()> {
        let lift = &self.lifted;
        let n = lift.rational.matrix.len();
        if n.checked_mul(n)
            .and_then(|v| v.checked_mul(n))
            .is_none_or(|v| v > limits.max_scalar_cells.saturating_mul(16))
        {
            return Err(Error::Limit(
                "root covariance matrix work exceeds limit".into(),
            ));
        }
        let z = lift.rational.variable;
        let stride = lift.size * lift.count;
        let mut multiplication = vec![vec![Atom::new(); n]; n];
        for (i, &source) in lift.monomials.iter().enumerate() {
            context.cancellation.check()?;
            let target = lift
                .monomials
                .iter()
                .position(|&m| m == source ^ mask)
                .ok_or_else(|| {
                    Error::InvalidInput("root generator leaves lifted subgroup".into())
                })?;
            let factor =
                super::sheet_germ::monomial_radicand(lift, source & mask, limits, context)?;
            exact::preflight_expression(&factor, z, 0, limits, context)?;
            for k in 0..stride {
                multiplication[i * stride + k][target * stride + k] = factor.clone();
            }
        }
        let generator_inputs = lift
            .source
            .roots
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, root)| &root.radicand)
            .collect::<Vec<_>>();
        for entry in lift.rational.matrix.iter().flatten() {
            let mut inputs = generator_inputs.clone();
            inputs.push(entry);
            // M is monomial: each product entry has one contributing term.
            // The factor also covers derivatives and the covariance sum.
            exact::preflight_operation(&inputs, z, 8, n * n, limits, context)?;
        }
        let square = crate::algebra::matmul(&multiplication, &multiplication);
        let radicand = super::sheet_germ::monomial_radicand(lift, mask, limits, context)?;
        let rho = (radicand.derivative(z) / (Atom::num(2) * &radicand))
            .together()
            .cancel();
        let left = crate::algebra::matmul(&lift.rational.matrix, &multiplication);
        let right = crate::algebra::matmul(&multiplication, &lift.rational.matrix);
        for i in 0..n {
            context.cancellation.check()?;
            for j in 0..n {
                let expected = if i == j {
                    radicand.clone()
                } else {
                    Atom::new()
                };
                if !(&square[i][j] - expected).together().cancel().is_zero()
                    || !(multiplication[i][j].derivative(z) - &left[i][j] + &right[i][j]
                        - &rho * &multiplication[i][j])
                        .together()
                        .cancel()
                        .is_zero()
                {
                    return Err(Error::Numerical(
                        "native root multiplication failed its exact covariance identity".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    // Log channels index each power's coefficient row while native polynomial
    // multiplication acts along the separate power axis.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn apply_sheet(
        &self,
        basis: &ExactFrobeniusBasis,
        mask: usize,
        valuation: i64,
        leading: &Gaussian,
        h: &Atom,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<ExactFrobeniusBasis> {
        let lift = &self.lifted;
        let z = lift.rational.variable;
        let order = basis.known_order;
        let n = basis.dimension();
        let stride = lift.size * lift.count;
        let factors = lift
            .monomials
            .iter()
            .map(|&m| super::sheet_germ::monomial_radicand(lift, m & mask, limits, context))
            .collect::<Result<Vec<_>>>()?;
        let low = factors
            .iter()
            .map(|a| super::sheet_germ::valuation(a, z, limits, context).map(|a| a.0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .min()
            .unwrap_or(0);
        let coefficients = factors
            .iter()
            .map(|factor| {
                let leading_atom = Atom::num(leading.clone());
                let shift = Atom::var(z).pow(low);
                exact::preflight_operation(
                    &[factor, &leading_atom, &shift],
                    z,
                    2,
                    1,
                    limits,
                    context,
                )?;
                let regular = (factor / (Atom::num(leading.clone()) * Atom::var(z).pow(low)))
                    .together()
                    .cancel();
                exact::normalized_root_series(h, &regular, z, order, limits, context)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut columns = Vec::with_capacity(n);
        for column in &basis.columns {
            context.cancellation.check()?;
            let logs = column
                .coefficients
                .iter()
                .take(order + 1)
                .map(Vec::len)
                .max()
                .unwrap_or(1);
            if n.checked_mul(n)
                .and_then(|v| v.checked_mul(logs))
                .and_then(|v| v.checked_mul(order + 1))
                .is_none_or(|v| v > limits.max_scalar_cells)
            {
                return Err(Error::Limit(
                    "root action coefficient allocation exceeds limit".into(),
                ));
            }
            let mut values = vec![vec![vec![zero(); n]; logs]; order + 1];
            for (block, &monomial) in lift.monomials.iter().enumerate() {
                let target = lift
                    .monomials
                    .iter()
                    .position(|&m| m == monomial ^ mask)
                    .unwrap();
                for log in 0..logs {
                    for component in 0..stride {
                        let input = column
                            .coefficients
                            .iter()
                            .take(order + 1)
                            .map(|logs| {
                                logs.get(log).map_or_else(zero, |row| {
                                    row[target * stride + component].clone()
                                })
                            })
                            .collect();
                        let product = exact::polynomial_product(
                            coefficients[block].clone(),
                            input,
                            z,
                            limits,
                            context,
                        )?;
                        for (k, value) in product.into_iter().take(order + 1).enumerate() {
                            values[k][log][block * stride + component] = value;
                        }
                    }
                }
            }
            columns.push(RecurrenceColumn {
                exponent: (&column.exponent + Atom::num(low)
                    - Atom::num(Rational::from((valuation, 2))))
                .together()
                .cancel(),
                coefficients: values,
            });
        }
        Ok(ExactFrobeniusBasis {
            columns,
            known_order: order,
            limits: limits.clone(),
            elimination_mode: basis.elimination_mode,
        })
    }
}

fn recover_action(
    basis: &ExactFrobeniusBasis,
    image: &ExactFrobeniusBasis,
    limits: &ExactFrobeniusLimits,
    context: &RunContext,
) -> Result<Option<Vec<Vec<Gaussian>>>> {
    let n = basis.dimension();
    let mut selectors = BTreeSet::new();
    for column in &basis.columns {
        let exponent = exact::gaussian(&column.exponent)?.re;
        for (k, logs) in column
            .coefficients
            .iter()
            .take(basis.known_order + 1)
            .enumerate()
        {
            for (log, values) in logs.iter().enumerate() {
                for (i, value) in values.iter().enumerate() {
                    if !value.is_zero() {
                        if selectors
                            .len()
                            .checked_add(1)
                            .and_then(|rows: usize| rows.checked_mul(n))
                            .is_none_or(|cells| cells > limits.max_scalar_cells)
                        {
                            return Err(Error::Limit(
                                "common sheet selector allocation exceeds limit".into(),
                            ));
                        }
                        selectors.insert((&exponent + &Rational::from(k), log, i));
                    }
                }
            }
        }
    }
    if selectors
        .len()
        .checked_mul(n)
        .is_none_or(|v| v > limits.max_scalar_cells)
    {
        return Err(Error::Limit(
            "common sheet selector allocation exceeds limit".into(),
        ));
    }
    let mut rows = Vec::new();
    let mut images = Vec::new();
    for (power, log_power, component) in selectors {
        context.cancellation.check()?;
        let selector = AsymptoticSelector {
            component,
            power: Atom::num(power),
            log_power,
        };
        let (row, image_row) = match (
            basis.coefficient_row(&selector),
            image.coefficient_row(&selector),
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(Error::Accuracy(_)), _) | (_, Err(Error::Accuracy(_))) => continue,
            (Err(e), _) | (_, Err(e)) => return Err(e),
        };
        rows.push(row);
        if exact::rank(&rows, limits, context)? < rows.len() {
            rows.pop();
            continue;
        }
        images.push(image_row);
        if rows.len() == n {
            return exact::multiply(
                exact::inverse(&rows, limits, context)?,
                images,
                limits,
                context,
            )
            .map(Some);
        }
    }
    Ok(None)
}
