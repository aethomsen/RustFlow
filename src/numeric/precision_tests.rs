//! Exact rational rounding oracles and unchanged general complex formulas.
use super::*;

fn rounded(p: Precision, value: Rational) -> Float {
    // The portable arithmetic backend stores full words, whereas with_val
    // rounds to the requested bit count. Compare at that requested precision.
    let value = Float::from_rational_round(&value, p.bits, Nearest);
    Float::with_val(p.bits, value.as_raw())
}

fn legacy_mul(p: Precision, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
    let ac = a.re.mul_round(&b.re, p.bits, Nearest);
    let bd = a.im.mul_round(&b.im, p.bits, Nearest);
    let ad = a.re.mul_round(&b.im, p.bits, Nearest);
    let bc = a.im.mul_round(&b.re, p.bits, Nearest);
    Complex::new(
        ac.sub_round(&bd, p.bits, Nearest),
        ad.add_round(&bc, p.bits, Nearest),
    )
}

fn legacy_div(p: Precision, a: &ComplexFloat, b: &ComplexFloat) -> ComplexFloat {
    let conjugate = Complex::new(b.re.clone(), -b.im.clone());
    let numerator = legacy_mul(p, a, &conjugate);
    let denominator = legacy_mul(p, b, &conjugate).re;
    Complex::new(
        numerator.re.div_round(&denominator, p.bits, Nearest),
        numerator.im.div_round(&denominator, p.bits, Nearest),
    )
}

#[test]
fn real_operand_shortcuts_use_correct_rounding_at_mixed_precisions() -> Result<()> {
    for bits in [53, 80, 201, 400] {
        let p = Precision { bits };
        for source_bits in [40, 201, 700] {
            let storage = Precision { bits: source_bits };
            let tiny = storage.powi(&storage.i(2), -1000);
            let large = storage.powi(&storage.i(2), 1000);
            let third = storage.rational(&Rational::from((1, 3)));
            let values = [
                storage.zero(),
                storage.i(1),
                storage.i(-3),
                third,
                storage.complex(0, -2),
                storage.complex(2, -3),
                tiny,
                large,
            ];
            for a in &values {
                let expected = if a.im.is_zero() {
                    rounded(p, a.re.to_rational().abs())
                } else {
                    legacy_mul(p, a, &Complex::new(a.re.clone(), -a.im.clone()))
                        .re
                        .sqrt()
                };
                assert_eq!(p.norm(a), expected);
                for b in &values {
                    assert_eq!(p.mul(a, b), legacy_mul(p, a, b));
                    if *b != storage.zero() {
                        let expected = if b.im.is_zero() {
                            Complex::new(
                                rounded(p, a.re.to_rational() / b.re.to_rational()),
                                rounded(p, a.im.to_rational() / b.re.to_rational()),
                            )
                        } else {
                            legacy_div(p, a, b)
                        };
                        let actual = p.div(a, b);
                        if b.im.is_zero() {
                            assert_eq!(p.round(&actual), expected);
                        } else {
                            assert_eq!(actual, expected);
                        }
                    }
                }
                for (n, d) in [(1, 3), (-7, 11), (0, 5)] {
                    let numerator = if n == 1 {
                        a.clone()
                    } else {
                        legacy_mul(p, a, &p.i(n))
                    };
                    let expected = Complex::new(
                        rounded(p, numerator.re.to_rational() / Rational::from(d)),
                        rounded(p, numerator.im.to_rational() / Rational::from(d)),
                    );
                    assert_eq!(p.round(&p.scale(a, n, d)), expected);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn nonfinite_values_keep_general_complex_arithmetic_behavior() -> Result<()> {
    let p = Precision::decimal(40)?;
    for real in ["inf", "-inf", "NaN"] {
        let a = p.parse(real, "0")?;
        let b = p.i(3);
        let actual = p.mul(&a, &b);
        let expected = legacy_mul(p, &a, &b);
        assert_eq!(actual.re.to_string(), expected.re.to_string());
        assert_eq!(actual.im.to_string(), expected.im.to_string());
        let actual = p.div(&a, &b);
        let expected = legacy_div(p, &a, &b);
        assert_eq!(actual.re.to_string(), expected.re.to_string());
        assert_eq!(actual.im.to_string(), expected.im.to_string());
    }
    Ok(())
}

#[cfg(feature = "automatic")]
#[test]
fn gamma_request_measurement_for_equal_mass_two_loop_boundary() -> Result<()> {
    use crate::*;
    let family = IntegralFamily {
        name: "gamma_request_measurement_sunset".into(),
        loops: vec!["k1".into(), "k2".into()],
        external: vec![],
        external_gram: vec![],
        propagators: vec![
            Propagator::quadratic(&[1, 0], &[], Atom::one(), &[])?,
            Propagator::quadratic(&[0, 1], &[], Atom::one(), &[])?,
            Propagator::quadratic(&[1, 1], &[], Atom::one(), &[])?,
        ],
        physical_propagators: 3,
        epsilon: symbol!("gamma_measurement::eps"),
        dimension: 4,
    };
    let options = FlowOptions {
        digits: 107,
        guard_digits: 71,
        series_order: 300,
        sampled_reduction: false,
        mass_mode: MassMode::Propagators(vec![0]),
        ..Default::default()
    };
    let backend = RustRedBackend::default();
    let context = RunContext::default();
    let prepared = PreparedFlow::new(
        &family,
        &[Integral(vec![1, 1, 1]), Integral(vec![0, 1, 1])],
        &KinematicPoint::default(),
        &backend,
        &options,
        &context,
    )?;
    let boundary = recursive::RecursiveBoundary::new(&backend, &options, &context);
    GAMMA_VALUES.with(|values| values.borrow_mut().clear());
    GAMMA_REQUEST_COUNTS.with(|counts| *counts.borrow_mut() = Some(Default::default()));
    let result = prepared.evaluate(
        &Rational::from((1, 1_600_000)),
        &options,
        &boundary,
        &context,
    );
    let counts = GAMMA_REQUEST_COUNTS.with(|counts| counts.borrow_mut().take().unwrap());
    let result = result?;
    assert_eq!(result.len(), 2);
    let total = counts.values().sum::<usize>();
    let repeated = counts.values().filter(|count| **count > 1).sum::<usize>();
    eprintln!(
        "gamma_request_measurement total={total} unique={} repeated_requests={repeated} counts={counts:?}",
        counts.len()
    );
    assert!(
        total > counts.len(),
        "expected recursive boundary reuse is absent"
    );
    Ok(())
}

#[test]
fn gamma_cache_reuses_high_precision_arguments_and_keeps_precision_profiles_separate() -> Result<()>
{
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
    GAMMA_EVALUATIONS.with(|evaluations| evaluations.set(Some(0)));
    let p = Precision { bits: 5000 };
    let argument = p.rational(&Rational::from((3, 2))).re;
    let first = p.gamma_real(&argument)?;
    assert_eq!(p.gamma_real(&argument)?, first);
    assert_eq!(GAMMA_EVALUATIONS.with(|count| count.get()), Some(1));
    let refined = Precision { bits: 5100 };
    let higher = refined.gamma_real(&argument)?;
    assert_eq!(higher.re.prec(), refined.bits);
    assert_eq!(GAMMA_EVALUATIONS.with(|count| count.get()), Some(2));
    p.gamma_real(&p.rational(&Rational::from((5, 2))).re)?;
    assert_eq!(GAMMA_EVALUATIONS.with(|count| count.get()), Some(3));
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
    assert_eq!(p.gamma_real(&argument)?, first);
    assert_eq!(GAMMA_EVALUATIONS.with(|count| count.replace(None)), Some(4));
    assert!(p.close(&first, &p.round(&higher), 1400));
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
    Ok(())
}

#[test]
fn gamma_cache_evicts_by_entry_count_and_precision_aware_byte_estimate() {
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
    let p = Precision { bits: 80 };
    for value in 1..=65 {
        remember_gamma(p.bits, p.real(value), p.i(1));
    }
    GAMMA_VALUES.with(|cache| {
        let cache = cache.borrow();
        assert_eq!(cache.len(), GAMMA_CACHE_MAX_ENTRIES);
        assert_eq!(cache.front().unwrap().1, p.real(2));
    });
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
    // Synthetic precision tags exercise the budget without actually reserving
    // that much memory in the unit test; value correctness is tested above.
    for value in 1..=40 {
        remember_gamma(1_000_000, p.real(value), p.i(1));
    }
    GAMMA_VALUES.with(|cache| {
        let cache = cache.borrow();
        let bytes = cache
            .iter()
            .map(|(bits, _, _)| gamma_cache_entry_bytes(*bits))
            .sum::<u64>();
        assert!(bytes <= GAMMA_CACHE_MAX_BYTES);
        assert!(cache.len() < 40);
        assert!(bytes + gamma_cache_entry_bytes(1_000_000) > GAMMA_CACHE_MAX_BYTES);
    });
    let before = GAMMA_VALUES.with(|cache| cache.borrow().len());
    remember_gamma(u32::MAX, p.real(41), p.i(1));
    assert_eq!(GAMMA_VALUES.with(|cache| cache.borrow().len()), before);
    GAMMA_VALUES.with(|cache| cache.borrow_mut().clear());
}
