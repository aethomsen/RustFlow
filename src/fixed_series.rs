//! Arithmetic adapter for Symbolica-owned series operations at fixed MPFR precision.
use crate::{ComplexFloat as C, Error, Precision, Result};
use std::fmt;
use symbolica::domains::float::FloatField;
use symbolica::domains::{Ring, RingOps, Set};
use symbolica::poly::series::Series;
use symbolica::prelude::*;
use symbolica::printer::{PrintOptions, PrintState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FixedComplexRing {
    bits: u32,
}
impl FixedComplexRing {
    pub(crate) fn new(p: Precision) -> Self {
        Self { bits: p.bits }
    }
    pub fn precision(self) -> Precision {
        Precision { bits: self.bits }
    }
}
impl fmt::Display for FixedComplexRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MPFR complex, {} working bits", self.bits)
    }
}
impl Set for FixedComplexRing {
    type Element = C;
    fn size(&self) -> Option<Integer> {
        None
    }
}
impl RingOps<&C> for FixedComplexRing {
    fn add(&self, a: &C, b: &C) -> C {
        self.precision().add(a, b)
    }
    fn sub(&self, a: &C, b: &C) -> C {
        self.precision().sub(a, b)
    }
    fn mul(&self, a: &C, b: &C) -> C {
        self.precision().mul(a, b)
    }
    fn neg(&self, a: &C) -> C {
        self.precision().round(&self.precision().neg(a))
    }
    fn add_assign(&self, a: &mut C, b: &C) {
        *a = self.precision().add(a, b);
    }
    fn sub_assign(&self, a: &mut C, b: &C) {
        *a = self.precision().sub(a, b);
    }
    fn mul_assign(&self, a: &mut C, b: &C) {
        *a = self.precision().mul(a, b);
    }
    fn add_mul_assign(&self, a: &mut C, b: &C, c: &C) {
        let p = self.precision();
        *a = p.add(a, &p.mul(b, c));
    }
    fn sub_mul_assign(&self, a: &mut C, b: &C, c: &C) {
        let p = self.precision();
        *a = p.sub(a, &p.mul(b, c));
    }
}
impl RingOps<C> for FixedComplexRing {
    fn add(&self, a: C, b: C) -> C {
        RingOps::<&C>::add(self, &a, &b)
    }
    fn sub(&self, a: C, b: C) -> C {
        RingOps::<&C>::sub(self, &a, &b)
    }
    fn mul(&self, a: C, b: C) -> C {
        RingOps::<&C>::mul(self, &a, &b)
    }
    fn neg(&self, a: C) -> C {
        RingOps::<&C>::neg(self, &a)
    }
    fn add_assign(&self, a: &mut C, b: C) {
        RingOps::<&C>::add_assign(self, a, &b);
    }
    fn sub_assign(&self, a: &mut C, b: C) {
        RingOps::<&C>::sub_assign(self, a, &b);
    }
    fn mul_assign(&self, a: &mut C, b: C) {
        RingOps::<&C>::mul_assign(self, a, &b);
    }
    fn add_mul_assign(&self, a: &mut C, b: C, c: C) {
        RingOps::<&C>::add_mul_assign(self, a, &b, &c);
    }
    fn sub_mul_assign(&self, a: &mut C, b: C, c: C) {
        RingOps::<&C>::sub_mul_assign(self, a, &b, &c);
    }
}
impl Ring for FixedComplexRing {
    fn zero(&self) -> C {
        self.precision().zero()
    }
    fn one(&self) -> C {
        self.precision().i(1)
    }
    fn nth(&self, n: Integer) -> C {
        self.precision().rational(&Rational::from(n))
    }
    fn pow(&self, a: &C, mut n: u64) -> C {
        let p = self.precision();
        let mut result = p.i(1);
        let mut base = p.round(a);
        while n != 0 {
            if n & 1 == 1 {
                result = p.mul(&result, &base);
            }
            n >>= 1;
            if n != 0 {
                base = p.mul(&base, &base);
            }
        }
        result
    }
    fn is_zero(&self, a: &C) -> bool {
        *a == self.precision().zero()
    }
    fn is_one(&self, a: &C) -> bool {
        *a == self.precision().i(1)
    }
    fn one_is_gcd_unit() -> bool {
        true
    }
    fn characteristic(&self) -> Integer {
        0.into()
    }
    fn try_inv(&self, a: &C) -> Option<C> {
        self.try_div(&self.precision().i(1), a)
    }
    fn try_div(&self, a: &C, b: &C) -> Option<C> {
        let p = self.precision();
        if !p.finite(a) || !p.finite(b) || *b == p.zero() {
            return None;
        }
        let value = p.div(a, b);
        p.finite(&value).then_some(value)
    }
    fn format<W: fmt::Write>(
        &self,
        a: &C,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> std::result::Result<bool, fmt::Error> {
        FloatField::from_rep(self.precision().zero()).format(a, opts, state, f)
    }
}

pub type NumericSeries = Series<FixedComplexRing>;
/// Convert a regular exact Symbolica series after native symbolic rpow/series.
/// The retained order is not silently extended; unknown coefficients stay unknown.
#[cfg(test)]
pub fn evaluate_series(
    exact: &Series<symbolica::domains::atom::AtomField>,
    p: Precision,
    values: &ahash::HashMap<Atom, C>,
) -> Result<NumericSeries> {
    if p.bits < 2 {
        return Err(Error::InvalidInput(
            "series precision must be at least two bits".into(),
        ));
    }
    let order = exact.absolute_order();
    if !order.is_integer() || order <= 0 || order > (crate::series_limits::MAX_ORDER + 1) as i64 {
        return Err(Error::Unsupported(
            "numeric coefficient chart requires a positive integer series order at most 200001"
                .into(),
        ));
    }
    let center = p.eval(&exact.get_expansion_point(), values)?;
    let prototype = Series::new(
        &FixedComplexRing::new(p),
        None,
        exact.get_variable(),
        center,
        order,
    );
    let mut result = prototype.zero();
    for (exponent, coefficient) in exact.terms() {
        if coefficient.is_zero() {
            continue;
        }
        if !exponent.is_integer() || exponent < 0 {
            return Err(Error::Unsupported(
                "numeric coefficient chart requires an ordinary Taylor series".into(),
            ));
        }
        result = result + prototype.monomial(p.eval(coefficient, values)?, exponent);
    }
    Ok(result)
}

/// Construct a local series in `variable` about zero. Exactly coefficients.len()
/// powers are known, even when high coefficients vanish. No f64 conversion or
/// source-precision promotion is used as accuracy evidence.
pub fn fixed_series(p: Precision, variable: Symbol, coefficients: &[C]) -> Result<NumericSeries> {
    if coefficients.is_empty()
        || coefficients.len() > crate::series_limits::MAX_ORDER + 1
        || p.bits < 2
    {
        return Err(Error::InvalidInput(
            "fixed series requires 1..=200001 retained coefficients and valid precision".into(),
        ));
    }
    crate::series_limits::admit_complex(
        p,
        coefficients.len() - 1,
        (coefficients.len() as u128).saturating_mul(8),
        "native fixed series construction",
    )?;
    if coefficients.iter().any(|a| !p.finite(a)) {
        return Err(Error::Numerical(
            "nonfinite fixed-series coefficient".into(),
        ));
    }
    let prototype = Series::new(
        &FixedComplexRing::new(p),
        None,
        std::sync::Arc::new(variable.into()),
        p.zero(),
        Rational::from(coefficients.len() as i64),
    );
    // A linear fold repeatedly copies a growing dense native Series. Pairwise
    // native addition preserves the same coefficient operations without that
    // quadratic construction cost. Omit rounded zero monomials so a sparse
    // high-power input does not repeatedly materialize its leading zeros.
    let mut level = coefficients
        .iter()
        .enumerate()
        .map(|(index, coefficient)| (index, p.round(coefficient)))
        .filter(|(_, coefficient)| !coefficient.is_zero())
        .map(|(index, coefficient)| prototype.monomial(coefficient, Rational::from(index as i64)))
        .collect::<Vec<_>>();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut terms = level.into_iter();
        while let Some(left) = terms.next() {
            next.push(if let Some(right) = terms.next() {
                left + right
            } else {
                left
            });
        }
        level = next;
    }
    // A shifted monomial knows more than the supplied finite prefix. Intersect
    // with the original zero prototype to retain exactly its absolute order,
    // also when every coefficient or all leading coefficients vanish.
    Ok(if let Some(result) = level.pop() {
        prototype.zero() + result
    } else {
        prototype.zero()
    })
}

/// Read only known Taylor coefficients. A request crossing the series remainder
/// is an accuracy error rather than an invented string of zero coefficients.
pub fn coefficients(series: &NumericSeries, length: usize) -> Result<Vec<C>> {
    if length > crate::series_limits::MAX_ORDER + 1 {
        return Err(Error::Limit(
            "numeric series extraction exceeds 200001 coefficients".into(),
        ));
    }
    let p = series.get_field().precision();
    crate::series_limits::admit_complex(
        p,
        length.saturating_sub(1),
        (length as u128).saturating_mul(2),
        "native fixed series extraction",
    )?;
    (0..length)
        .map(|index| {
            let coefficient = series
                .coefficient(Rational::from(index as i64))
                .ok_or_else(|| {
                    Error::Accuracy(
                        "requested coefficient is in the unknown series remainder".into(),
                    )
                })?;
            if !p.finite(&coefficient) {
                return Err(Error::Numerical("nonfinite numeric-series result".into()));
            }
            Ok(coefficient)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use symbolica::domains::{Ring, RingOps};
    use symbolica::poly::series::Series;
    fn bits(p: Precision, s: &NumericSeries) {
        for (_, c) in s.terms() {
            assert_eq!(c.re.as_raw().prec(), p.bits);
            assert_eq!(c.im.as_raw().prec(), p.bits);
        }
    }
    #[test]
    fn native_series_preserves_fixed_rounding_and_remainders() {
        let p = Precision::decimal(60).unwrap();
        let x = symbol!("fixed_ring_probe::x");
        let field = FixedComplexRing::new(p);
        let prototype = Series::new(
            &field,
            None,
            Arc::new(x.into()),
            p.zero(),
            Rational::from(161),
        );
        let near = p.sub(&p.i(1), &p.powi(&p.i(10), -40));
        let a = prototype.constant(p.i(1)) + prototype.monomial(near.clone(), Rational::from(1));
        let b = prototype.constant(p.i(1)) + prototype.monomial(p.i(-1), Rational::from(1));
        let c = &a * &b;
        bits(p, &c);
        assert_eq!(c.coefficient(1.into()).unwrap(), p.sub(&near, &p.i(1)));

        let exact_a = Atom::num(Complex::new(Rational::from(1), Rational::from(2)))
            + Atom::var(x).pow(13)
                * Atom::num(Complex::new(Rational::from((3, 2)), Rational::from(-7)))
            + Atom::var(x).pow(97) * Atom::num((7, 11));
        let exact_b = Atom::num(Complex::new(Rational::from(-2), Rational::from(1)))
            + Atom::var(x).pow(19) * Atom::num((2, 5))
            + Atom::var(x).pow(89) * Atom::num(Complex::new(Rational::from(7), Rational::from(3)));
        let a =
            evaluate_series(&exact_a.series(x, 0, 160).unwrap(), p, &Default::default()).unwrap();
        let b =
            evaluate_series(&exact_b.series(x, 0, 160).unwrap(), p, &Default::default()).unwrap();
        let c = &a * &b;
        let expected = (&exact_a * &exact_b).series(x, 0, 160).unwrap();
        bits(p, &c);
        for k in 0..=160 {
            let actual = c.coefficient(Rational::from(k)).unwrap();
            let expected = p
                .eval(
                    &expected.coefficient(Rational::from(k)).unwrap(),
                    &Default::default(),
                )
                .unwrap();
            assert!(p.close(&actual, &expected, 55));
        }
        assert!(c.coefficient(161.into()).is_none());

        let low = Precision { bits: 16 };
        let high = Precision { bits: 512 };
        let varied = prototype.constant(low.complex(2, 3))
            + prototype.monomial(high.complex(7, -2), 13.into());
        let product = &varied * &varied;
        bits(p, &product);
        assert_eq!(product.coefficient(0.into()).unwrap(), p.complex(-5, 12));
        assert_eq!(product.coefficient(13.into()).unwrap(), p.complex(40, 34));
        assert_eq!(product.coefficient(26.into()).unwrap(), p.complex(45, -28));
        let neg = RingOps::<&C>::neg(&field, &low.complex(2, 3));
        assert_eq!(neg.re.as_raw().prec(), p.bits);
        assert!(field.try_inv(&p.zero()).is_none());

        let root = (Atom::num(1) + Atom::var(x))
            .series(x, 0, 80)
            .unwrap()
            .rpow(Rational::from((1, 2)))
            .unwrap();
        let numeric = evaluate_series(&root, p, &Default::default()).unwrap();
        let squared = &numeric * &numeric;
        bits(p, &squared);
        for k in 0..=80 {
            let expected = if k < 2 { p.i(1) } else { p.zero() };
            assert!(p.close(&squared.coefficient(k.into()).unwrap(), &expected, 55));
        }

        let data = vec![p.zero(), p.complex(2, 3), p.zero(), p.zero()];
        let retained = fixed_series(p, x, &data).unwrap();
        assert_eq!(retained.absolute_order(), Rational::from(4));
        assert_eq!(coefficients(&retained, 4).unwrap(), data);
        assert!(coefficients(&retained, 5).is_err());
        let zero = fixed_series(p, x, &vec![p.zero(); 8]).unwrap();
        assert_eq!(zero.absolute_order(), Rational::from(8));
        assert_eq!(coefficients(&zero, 8).unwrap(), vec![p.zero(); 8]);
    }

    #[test]
    fn balanced_native_construction_preserves_sparse_prefix_and_products() {
        let variable = symbol!("balanced_series_contract::x");
        let stored = Precision::decimal(120).unwrap();
        for working_bits in [201, 333] {
            let p = Precision { bits: working_bits };
            for length in [1, 4, 7, 64, 96, 256, 512] {
                for kind in 0..4 {
                    let input = (0..length)
                        .map(|index| {
                            if kind == 2
                                || (kind == 1 && index < length / 2)
                                || (kind == 3 && index + 1 < length)
                            {
                                stored.zero()
                            } else {
                                C::new(
                                    stored.rational(&Rational::from((index + 1, index + 3))).re,
                                    stored.rational(&Rational::from((-index - 2, index + 5))).re,
                                )
                            }
                        })
                        .collect::<Vec<_>>();
                    let prototype = Series::new(
                        &FixedComplexRing::new(p),
                        None,
                        std::sync::Arc::new(variable.into()),
                        p.zero(),
                        Rational::from(length),
                    );
                    // Retain the preceding native linear construction as an
                    // independent representation oracle, including its remainder.
                    let reference =
                        input
                            .iter()
                            .enumerate()
                            .fold(prototype.zero(), |sum, (index, value)| {
                                sum + prototype
                                    .monomial(p.round(value), Rational::from(index as i64))
                            });
                    let result = fixed_series(p, variable, &input).unwrap();
                    assert_eq!(result.absolute_order(), Rational::from(length));
                    assert_eq!(result.absolute_order(), reference.absolute_order());
                    assert_eq!(
                        coefficients(&result, length as usize).unwrap(),
                        coefficients(&reference, length as usize).unwrap(),
                    );
                    assert!(result.coefficient(Rational::from(length)).is_none());
                    bits(p, &result);
                    if length <= 96 {
                        let actual_product = &result * &result;
                        let reference_product = &reference * &reference;
                        assert_eq!(
                            actual_product.absolute_order(),
                            reference_product.absolute_order()
                        );
                        assert_eq!(
                            coefficients(&actual_product, length as usize).unwrap(),
                            coefficients(&reference_product, length as usize).unwrap(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_precision_and_nonfinite_series_are_errors() {
        let p = Precision::decimal(30).unwrap();
        let x = symbol!("fixed_series_validation::x");
        let exact = Atom::var(x).series(x, 0, 8).unwrap();
        assert!(evaluate_series(&exact, Precision { bits: 1 }, &Default::default()).is_err());
        assert!(fixed_series(Precision { bits: 1 }, x, &[p.i(1)]).is_err());
        assert!(fixed_series(p, x, &[p.parse("inf", "0").unwrap()]).is_err());
        assert!(fixed_series(p, x, &[]).is_err());
    }

    #[test]
    fn extended_native_series_preserves_known_high_coefficient_and_remainder() {
        let p = Precision::decimal(40).unwrap();
        let x = symbol!("extended_native_series::x");
        let mut input = vec![p.zero(); 10_002];
        input[0] = p.i(1);
        input[10_001] = p.i(3);
        let series = fixed_series(p, x, &input).unwrap();
        assert_eq!(coefficients(&series, input.len()).unwrap(), input);
        assert!(coefficients(&series, input.len() + 1).is_err());
        assert!(matches!(
            fixed_series(Precision { bits: u32::MAX }, x, &input),
            Err(Error::Limit(_))
        ));
    }
}
