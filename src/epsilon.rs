//! Numerical Laurent reconstruction from exact, nonzero regulator samples.
use crate::{ComplexFloat as C, Error, Precision, Result};
use std::{collections::BTreeMap, sync::Arc};
use symbolica::domains::float::FloatField;
use symbolica::prelude::*;

#[derive(Clone, Debug)]
pub struct LaurentExpansion {
    pub coefficients: BTreeMap<i32, C>,
    /// Set only after independent precision/sample refinement. Relative digits
    /// for resolved nonzero coefficients; absolute digits near numerical zero.
    pub verified_digits: Option<u32>,
    pub working_bits: u32,
    pub samples: usize,
    pub validation_samples: usize,
    pub refinements: usize,
    /// Absolute coefficient changes between the last independent fits.
    pub comparison_errors: BTreeMap<i32, Float>,
}

/// Fit using all samples, returning powers `leading..=last`.
/// This operation alone does not certify accuracy; use independent refinements.
pub fn fit_epsilon(
    samples: &[Rational],
    values: &[C],
    leading: i32,
    last: i32,
    p: Precision,
) -> Result<LaurentExpansion> {
    if last < leading
        || samples.len() != values.len()
        || samples.len() < (last as i64 - leading as i64 + 1) as usize
    {
        return Err(Error::InvalidInput(
            "epsilon fit dimensions or power range".into(),
        ));
    }
    if samples.iter().any(|v| v.is_zero()) {
        return Err(Error::InvalidInput(
            "epsilon samples must be nonzero".into(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != samples.len() {
        return Err(Error::InvalidInput(
            "epsilon samples must be distinct".into(),
        ));
    }
    if values.iter().any(|v| !p.finite(v)) {
        return Err(Error::InvalidInput("nonfinite epsilon sample".into()));
    }
    if samples.len() > 10000 {
        return Err(Error::Limit(
            "epsilon reconstruction exceeds 10000 samples".into(),
        ));
    }
    // Normalize exact coordinates before converting to the numerical field.
    // Small regulator radii must not create a badly scaled interpolation basis.
    let radius = samples
        .iter()
        .map(|r| {
            if *r < Rational::zero() {
                -r.clone()
            } else {
                r.clone()
            }
        })
        .max()
        .unwrap();
    let coordinates = samples
        .iter()
        .map(|r| p.rational(&(r / &radius)))
        .collect::<Vec<_>>();
    if coordinates.iter().any(|value| !p.finite(value)) {
        return Err(Error::Numerical(
            "nonfinite normalized epsilon interpolation coordinate".into(),
        ));
    }
    if coordinates
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != samples.len()
    {
        return Err(Error::InsufficientPrecision {
            minimum_bits: p.bits.saturating_add(32),
            context: "epsilon interpolation coordinates round to the same value".into(),
        });
    }
    let numeric_radius = p.rational(&radius);
    if !p.finite(&numeric_radius) || numeric_radius == p.zero() {
        return Err(Error::Numerical(
            "epsilon interpolation radius exceeds the numerical backend range".into(),
        ));
    }
    let n = samples.len();
    // Symbolica owns Newton interpolation, polynomial arithmetic and its
    // precision-tracking complex field. No dense Vandermonde matrix is built.
    let field = FloatField::from_rep(p.zero());
    let prototype = MultivariatePolynomial::<_, u16>::new(
        &field,
        None,
        Arc::new(vec![
            symbol!("symbolica_amflow::epsilon_interpolation_coordinate").into(),
        ]),
    );
    let regular_values = samples
        .iter()
        .zip(values)
        .map(|(eps, value)| {
            prototype.constant(p.mul(value, &p.powi(&p.rational(eps), -i64::from(leading))))
        })
        .collect::<Vec<_>>();
    if regular_values
        .iter()
        .flat_map(|polynomial| polynomial.coefficients.iter())
        .any(|value| !p.finite(value))
    {
        return Err(Error::Numerical(
            "nonfinite pole-subtracted epsilon sample".into(),
        ));
    }
    let fitted = MultivariatePolynomial::newton_interpolation(&coordinates, &regular_values, 0);
    let mut coefficients = (leading..=last)
        .map(|k| (k, p.zero()))
        .collect::<BTreeMap<_, _>>();
    for term in &fitted {
        let offset = i32::from(term.exponents[0]);
        if i64::from(offset) <= i64::from(last) - i64::from(leading) {
            let coefficient = p.div(
                term.coefficient,
                &p.powi(&numeric_radius, i64::from(offset)),
            );
            if !p.finite(&coefficient) {
                return Err(Error::Numerical(
                    "nonfinite epsilon interpolation coefficient".into(),
                ));
            }
            coefficients.insert(leading + offset, coefficient);
        }
    }
    Ok(LaurentExpansion {
        coefficients,
        verified_digits: None,
        working_bits: p.bits,
        samples: n,
        validation_samples: 0,
        refinements: 0,
        comparison_errors: BTreeMap::new(),
    })
}

pub fn epsilon_samples(count: usize, denominator: i64) -> Result<Vec<Rational>> {
    if count == 0 || count > 10000 || denominator <= 0 {
        return Err(Error::InvalidInput("epsilon sampling parameters".into()));
    }
    let total = denominator
        .checked_mul(count as i64)
        .ok_or_else(|| Error::InvalidInput("epsilon denominator overflow".into()))?;
    Ok((1..=count)
        .map(|i| Rational::from((i as i64, total)))
        .collect())
}

/// Balanced real nodes within an arbitrary exact positive regulator radius.
/// Both signs are
/// analytic regulator samples, not different physical i0 prescriptions. Avoid
/// zero and retain exact rational coordinates for reduction and independent fits.
pub fn symmetric_epsilon_samples(count: usize, radius: &Rational) -> Result<Vec<Rational>> {
    if count == 0 || count > 10000 || *radius <= Rational::zero() {
        return Err(Error::InvalidInput("epsilon sampling parameters".into()));
    }
    let half = count.div_ceil(2) as i64;
    Ok((0..count)
        .map(|index| {
            let magnitude = (index / 2 + 1) as i64;
            radius
                * &Rational::from((
                    if index % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    },
                    half,
                ))
        })
        .collect())
}

/// A bounded-size proposal for independently refined numerical Laurent fits.
/// The radius balances omitted-order suppression against the guard precision
/// needed to extract the requested coefficients. It is not an accuracy claim.
#[derive(Clone, Debug)]
pub struct EpsilonFitProfile {
    pub samples: Vec<Rational>,
    pub radius: Rational,
    pub working_digits: u32,
}

impl EpsilonFitProfile {
    pub fn new(
        digits: u32,
        guard_digits: u32,
        leading: i32,
        last: i32,
        refinement: usize,
    ) -> Result<Self> {
        if digits == 0 || last < leading || refinement > 16 {
            return Err(Error::InvalidInput("epsilon fit profile parameters".into()));
        }
        let width = (i64::from(last) - i64::from(leading) + 1) as u64;
        let initial_count = (2 * width).max(32);
        let count = initial_count + 4 * refinement as u64;
        if count > 10000 {
            return Err(Error::Limit(
                "epsilon reconstruction exceeds 10000 samples".into(),
            ));
        }
        let radius_digits = (u64::from(digits) + 64)
            .div_ceil(initial_count - width)
            .max(3)
            + refinement as u64;
        let fit_guard = (width - 1) * radius_digits + 64 + 32 * refinement as u64;
        let working_digits = u64::from(digits) + fit_guard.max(u64::from(guard_digits));
        let working_digits = u32::try_from(working_digits)
            .map_err(|_| Error::Limit("epsilon fit precision overflow".into()))?;
        // Validate the backend's representable bit count before constructing
        // a potentially large exact denominator.
        Precision::decimal(working_digits)?;
        let radius = Rational::one() / Rational::from(Integer::from(10).pow(radius_digits));
        Ok(Self {
            samples: symmetric_epsilon_samples(count as usize, &radius)?,
            radius,
            working_digits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_rational_sampling_preserves_tiny_distinct_balanced_nodes() {
        let radius = Rational::one() / Rational::from(Integer::from(10).pow(1000));
        let samples = symmetric_epsilon_samples(32, &radius).unwrap();
        assert_eq!(samples.len(), 32);
        for pair in samples.chunks_exact(2) {
            assert_eq!(pair[0], -pair[1].clone());
            assert!(pair[0] > Rational::zero() && pair[0] <= radius);
        }
        assert_eq!(samples[30], radius);
        assert!(symmetric_epsilon_samples(0, &radius).is_err());
        assert!(symmetric_epsilon_samples(32, &Rational::zero()).is_err());
    }

    #[test]
    fn twenty_thousand_digit_profiles_keep_sample_count_bounded_and_refine_independently() {
        let first = EpsilonFitProfile::new(20000, 40, -1, 4, 0).unwrap();
        let next = EpsilonFitProfile::new(20000, 40, -1, 4, 1).unwrap();
        assert_eq!(first.samples.len(), 32);
        assert_eq!(next.samples.len(), 36);
        assert!(first.working_digits >= 23800);
        assert!(next.working_digits > first.working_digits);
        assert_eq!(first.radius, &next.radius * &Rational::from(10));
        assert!(
            first
                .samples
                .iter()
                .all(|sample| !next.samples.contains(sample))
        );
        assert!(EpsilonFitProfile::new(0, 40, -1, 4, 0).is_err());
        assert!(EpsilonFitProfile::new(20, 40, 1, 0, 0).is_err());
        assert!(EpsilonFitProfile::new(u32::MAX, 40, -1, 4, 0).is_err());
    }

    #[test]
    fn native_newton_reconstructs_complex_laurent_coefficients_at_twenty_thousand_digits() {
        // A convergent nonpolynomial with known coefficients tests truncation,
        // cancellation, native interpolation and regulator rescaling together.
        // The 20,000-digit kernel test takes only a bounded 32/36 samples.
        for attempt in 0..=1 {
            let profile = EpsilonFitProfile::new(20000, 40, -1, 4, attempt).unwrap();
            let p = Precision::decimal(profile.working_digits).unwrap();
            let values = profile
                .samples
                .iter()
                .map(|eps| {
                    let e = p.rational(eps);
                    p.div(
                        &p.complex(3, 4),
                        &p.mul(&e, &p.sub(&p.i(1), &p.scale(&e, 2, 1))),
                    )
                })
                .collect::<Vec<_>>();
            let fitted = fit_epsilon(&profile.samples, &values, -1, 4, p).unwrap();
            assert_eq!(fitted.verified_digits, None);
            for power in -1..=4 {
                let expected = p.scale(&p.complex(3, 4), 1 << (power + 1), 1);
                assert!(
                    p.close(&fitted.coefficients[&power], &expected, 20000),
                    "epsilon^{power}, refinement {attempt}"
                );
            }
        }
    }

    #[test]
    fn native_newton_preserves_sparse_coefficients_and_validates_nodes() {
        let p = Precision::decimal(100).unwrap();
        let samples = symmetric_epsilon_samples(12, &Rational::from((1, 100))).unwrap();
        let values = samples
            .iter()
            .map(|eps| {
                let e = p.rational(eps);
                p.add(
                    &p.scale(&p.powi(&e, -2), 7, 3),
                    &p.scale(&p.powi(&e, 2), -5, 7),
                )
            })
            .collect::<Vec<_>>();
        let fitted = fit_epsilon(&samples, &values, -2, 2, p).unwrap();
        for power in -2..=2 {
            let expected = match power {
                -2 => p.scale(&p.i(1), 7, 3),
                2 => p.scale(&p.i(1), -5, 7),
                _ => p.zero(),
            };
            assert!(p.close(&fitted.coefficients[&power], &expected, 70));
        }
        assert!(fit_epsilon(&[Rational::zero()], &[p.i(1)], 0, 0, p).is_err());
        assert!(
            fit_epsilon(
                &[Rational::one(), Rational::one()],
                &[p.i(1), p.i(1)],
                0,
                0,
                p
            )
            .is_err()
        );
        let nearly_one =
            Rational::one() + Rational::one() / Rational::from(Integer::from(10).pow(1000));
        assert!(matches!(
            fit_epsilon(&[Rational::one(), nearly_one], &[p.i(1), p.i(1)], 0, 0, p),
            Err(Error::InsufficientPrecision { .. })
        ));
    }
}
