//! Real Horner evaluation on positive matching coordinates in the unit disk.
use super::{C, Error, FrobeniusColumn, Precision, Result};
use symbolica::domains::float::{Float, RoundingDirection::Nearest};
use symbolica::prelude::*;

pub(super) fn logarithm(p: Precision, z: &C, winding: i32) -> Result<C> {
    if !p.finite(z) || *z == p.zero() {
        return Err(Error::InvalidInput(
            "evaluate Frobenius series at a nonzero matching point".into(),
        ));
    }
    Ok(if winding == 0 {
        p.log(z)
    } else {
        p.add(
            &p.log(z),
            &p.scale(&p.log(&p.i(-1)), 2 * i64::from(winding), 1),
        )
    })
}

/// Shared evaluator for a validated column. Both full-basis and streamed
/// evaluation use the same power/log branches and real Horner specialization.
pub(super) fn column_value(
    p: Precision,
    z: &C,
    log: &C,
    parameters: &ahash::HashMap<Atom, C>,
    column: &FrobeniusColumn,
) -> Result<Vec<C>> {
    let exponent = p.eval(&column.exponent, parameters)?;
    let mut power = p.exp(&p.mul(log, &exponent));
    if let Some(values) = real_column(p, z, log, &power, column) {
        return Ok(values);
    }
    let n = column.coefficients[0][0].len();
    let mut values = vec![p.zero(); n];
    for coeff in &column.coefficients {
        let mut logpower = p.i(1);
        for (index, row) in coeff.iter().enumerate() {
            let term_power = if index == 0 {
                power.clone()
            } else {
                p.mul(&power, &logpower)
            };
            for i in 0..n {
                values[i] = p.add(&values[i], &p.mul(&term_power, &row[i]));
            }
            if index + 1 < coeff.len() {
                logpower = p.mul(&logpower, log);
            }
        }
        power = p.mul(&power, z);
    }
    Ok(values)
}

pub(super) fn real_column(
    p: Precision,
    z: &C,
    logarithm: &C,
    prefactor: &C,
    column: &FrobeniusColumn,
) -> Option<Vec<C>> {
    if !z.im.is_zero()
        || z.re <= p.real(0)
        || z.re > p.real(1)
        || !logarithm.im.is_zero()
        || !prefactor.im.is_zero()
        || !p.finite(prefactor)
        || column
            .coefficients
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.im.is_zero())
    {
        return None;
    }
    let dimension = column.coefficients.first()?.first()?.len();
    let mut result = Vec::with_capacity(dimension);
    for component in 0..dimension {
        let mut value = p.real(0);
        for rows in column.coefficients.iter().rev() {
            let mut coefficient = p.real(0);
            for row in rows.iter().rev() {
                coefficient = coefficient
                    .mul_round(&logarithm.re, p.bits, Nearest)
                    .add_round(&row[component].re, p.bits, Nearest);
            }
            value =
                value
                    .mul_round(&z.re, p.bits, Nearest)
                    .add_round(&coefficient, p.bits, Nearest);
        }
        let value: Float = value.mul_round(&prefactor.re, p.bits, Nearest);
        if !value.is_finite() {
            return None;
        }
        result.push(C::new(value, p.real(0)));
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(p: Precision) -> FrobeniusColumn {
        FrobeniusColumn {
            exponent: Atom::num((-3, 5)),
            coefficients: (0..32)
                .map(|n| {
                    (0..3)
                        .map(|k| {
                            (0..2)
                                .map(|i| {
                                    p.rational(&Rational::from((
                                        if (n + k + i) % 2 == 0 { n + 1 } else { -n - 1 },
                                        7 + k + i,
                                    )))
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect(),
        }
    }

    #[test]
    fn real_horner_matches_independent_high_precision_power_sum() -> crate::Result<()> {
        let high = Precision::decimal(160)?;
        for digits in [30, 60, 100] {
            let p = Precision::decimal(digits + 12)?;
            for point in [
                Rational::from((1, 128)),
                Rational::from((7, 10)),
                Rational::one(),
            ] {
                let z = p.rational(&point);
                let log = p.log(&z);
                let prefactor = p.exp(&p.mul(&log, &p.rational(&Rational::from((-3, 5)))));
                let values = real_column(p, &z, &log, &prefactor, &column(p)).unwrap();
                let z = high.rational(&point);
                let log = high.log(&z);
                let source = column(high);
                let mut power = high.pow(&z, &high.rational(&Rational::from((-3, 5))));
                let mut expected = vec![high.zero(); 2];
                for rows in source.coefficients {
                    let mut logpower = high.i(1);
                    for row in rows {
                        for (value, coefficient) in expected.iter_mut().zip(row) {
                            *value = high
                                .add(value, &high.mul(&high.mul(&power, &logpower), &coefficient));
                        }
                        logpower = high.mul(&logpower, &log);
                    }
                    power = high.mul(&power, &z);
                }
                for (actual, expected) in values.iter().zip(expected) {
                    assert!(high.close(actual, &expected, digits));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn real_horner_retains_complex_and_large_coordinate_fallbacks() -> crate::Result<()> {
        let p = Precision::decimal(60)?;
        let z = p.rational(&Rational::from((1, 2)));
        let log = p.log(&z);
        let mut source = column(p);
        source.coefficients[0][0][0].im = p.parse("1e-300", "0")?.re;
        assert!(real_column(p, &z, &log, &p.i(1), &source).is_none());
        let source = column(p);
        for point in [p.i(-1), p.i(2), p.complex(1, 1)] {
            assert!(real_column(p, &point, &log, &p.i(1), &source).is_none());
        }
        assert!(real_column(p, &z, &p.complex(1, 1), &p.i(1), &source).is_none());
        Ok(())
    }
}
