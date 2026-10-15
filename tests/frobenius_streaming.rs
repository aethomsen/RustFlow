use std::sync::Arc;
use symbolica::prelude::*;
use symbolica_amflow::*;

fn weighted(
    p: Precision,
    matrix: Vec<Vec<ComplexFloat>>,
    constants: &[ComplexFloat],
) -> Vec<ComplexFloat> {
    matrix
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(constants)
                .fold(p.zero(), |sum, (value, weight)| {
                    p.add(&sum, &p.mul(value, weight))
                })
        })
        .collect()
}

#[test]
fn streamed_columns_match_full_basis_for_radial_sunset_resonant_and_sheared_systems() -> Result<()>
{
    let x = symbol!("streaming_comparison::x");
    let e = parse!("streaming_comparison::epsilon");
    let z = Atom::var(x);
    let eta = symbol!("streaming_comparison::eta");
    let t = Atom::var(eta);
    let denominator = (&t - Atom::num(3)) * (&t + Atom::one());
    let driving = (&e - Atom::one()) / &denominator;
    let sunset = DifferentialSystem {
        variable: eta,
        matrix: vec![
            vec![Atom::new(); 4],
            vec![
                Atom::new(),
                (Atom::one() - &e) / (&t + Atom::one()),
                Atom::new(),
                Atom::new(),
            ],
            vec![
                Atom::new(),
                Atom::new(),
                (Atom::one() - &e) / (&t + Atom::one()),
                Atom::new(),
            ],
            vec![
                -Atom::num(2) * &driving,
                driving.clone(),
                driving,
                -(-&t - Atom::num(2) * &e + Atom::one() + Atom::num(2) * &e * &t) / denominator,
            ],
        ],
    }
    .invert_variable(x);
    let systems = vec![
        DifferentialSystem {
            variable: x,
            matrix: vec![
                vec![
                    (Atom::one() - &e) / &z - &e / (Atom::one() - &z),
                    Atom::new(),
                ],
                vec![Atom::one(), Atom::new()],
            ],
        },
        DifferentialSystem {
            variable: x,
            matrix: vec![
                vec![
                    &e / &z - (Atom::one() - &e) / (Atom::one() - &z),
                    Atom::new(),
                ],
                vec![Atom::num(-1), Atom::new()],
            ],
        },
        DifferentialSystem {
            variable: x,
            matrix: vec![
                vec![&e / &z, Atom::one() / &z, Atom::new()],
                vec![Atom::new(), &e / &z, Atom::new()],
                vec![Atom::one(), Atom::new(), (&e + Atom::one()) / &z],
            ],
        },
        DifferentialSystem {
            variable: x,
            matrix: vec![
                vec![z.clone().pow(-2), z.clone().pow(-2)],
                vec![-z.clone().pow(-2), -z.clone().pow(-2)],
            ],
        },
        sunset,
    ];
    for digits in [75, 110] {
        let p = Precision::decimal(digits)?;
        let parameters =
            ahash::HashMap::from_iter([(e.clone(), p.rational(&Rational::from((1, 97))))]);
        for system in &systems {
            let prepared = system.prepare_frobenius(&RunContext::default())?;
            let full = prepared.evaluate(p, &parameters, 72, &RunContext::default())?;
            let constants = (0..system.matrix.len())
                .map(|i| p.rational(&Rational::from((i as i64 + 2, 3))))
                .collect::<Vec<_>>();
            for point in [
                p.rational(&Rational::from((1, 10))),
                p.parse("0.05", "0.07")?,
            ] {
                for winding in [-1, 0, 2] {
                    let expected = weighted(
                        p,
                        full.evaluate_with_winding(&point, &parameters, winding)?,
                        &constants,
                    );
                    let actual = prepared.evaluate_solution_with_winding(
                        p,
                        &parameters,
                        72,
                        &constants,
                        &point,
                        winding,
                        &RunContext::default(),
                    )?;
                    let recovered = prepared.match_values_with_winding(
                        p,
                        &parameters,
                        72,
                        &point,
                        &actual,
                        winding,
                        &RunContext::default(),
                    )?;
                    for (actual, expected) in recovered.iter().zip(&constants) {
                        assert!(p.close(actual, expected, digits - 20));
                    }
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!(p.close(actual, &expected, digits - 15));
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn short_asymptotic_matching_retains_logarithmic_branch_and_derivative() -> Result<()> {
    let x = symbol!("streaming_exact::x");
    let e = parse!("streaming_exact::epsilon");
    let z = Atom::var(x);
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![vec![&e / &z, Atom::one() / &z], vec![Atom::new(), &e / &z]],
    };
    let p = Precision::decimal(100)?;
    let parameters = ahash::HashMap::from_iter([(e.clone(), p.rational(&Rational::from((1, 31))))]);
    let prepared = system.prepare_frobenius(&RunContext::default())?;
    let short = prepared.evaluate(p, &parameters, 2, &RunContext::default())?;
    let constants = short.match_constraints(&[
        AsymptoticConstraint {
            component: 0,
            power: e.clone(),
            log_power: 0,
            coefficient: p.i(2),
        },
        AsymptoticConstraint {
            component: 1,
            power: e.clone(),
            log_power: 0,
            coefficient: p.i(3),
        },
    ])?;
    let expressions = [
        parse!("streaming_exact::x^streaming_exact::epsilon*(2+3*log(streaming_exact::x))"),
        parse!("3*streaming_exact::x^streaming_exact::epsilon"),
    ];
    let point = p.rational(&Rational::from((1, 5)));
    let actual = prepared.evaluate_solution(
        p,
        &parameters,
        40,
        &constants,
        &point,
        &RunContext::default(),
    )?;
    let mut values = parameters.clone();
    values.insert(z, point.clone());
    for (i, expression) in expressions.iter().enumerate() {
        assert!(p.close(&actual[i], &p.eval(expression, &values)?, 85));
        let derivative = system.matrix[i].iter().zip(&actual).try_fold(
            p.zero(),
            |sum, (entry, value)| -> Result<_> {
                Ok(p.add(&sum, &p.mul(&p.eval(entry, &values)?, value)))
            },
        )?;
        assert!(p.close(
            &derivative,
            &p.eval(&expression.derivative(x), &values)?,
            85
        ));
    }
    Ok(())
}

#[test]
fn streaming_retains_domain_cancellation_and_allocation_guards() -> Result<()> {
    let x = symbol!("streaming_guard::x");
    let e = parse!("streaming_guard::epsilon");
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![vec![parse!(
            "(streaming_guard::epsilon^2-1)/((streaming_guard::epsilon-1)*streaming_guard::x)"
        )]],
    };
    let prepared = system.prepare_frobenius(&RunContext::default())?;
    let p = Precision::decimal(60)?;
    let constants = vec![p.i(2)];
    let point = p.rational(&Rational::from((1, 2)));
    let excluded = ahash::HashMap::from_iter([(e.clone(), p.i(1))]);
    assert!(matches!(
        prepared.evaluate_solution(p, &excluded, 8, &constants, &point, &RunContext::default()),
        Err(Error::Numerical(_))
    ));
    let parameters = ahash::HashMap::from_iter([(e, p.i(2))]);
    // Refuse impossible precision before allocating a matrix, result vector,
    // or a rounded logarithm at that precision.
    let oversized = Precision { bits: u32::MAX };
    assert!(matches!(
        prepared.evaluate_solution(
            oversized,
            &parameters,
            8,
            &constants,
            &point,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        prepared.match_values(
            oversized,
            &parameters,
            8,
            &point,
            &constants,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        prepared.evaluate_solution(
            p,
            &parameters,
            200_001,
            &constants,
            &point,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        prepared.evaluate_solution(p, &parameters, 8, &[], &point, &RunContext::default()),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        prepared.match_values(p, &parameters, 8, &point, &[], &RunContext::default()),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        prepared.evaluate_solution(
            p,
            &parameters,
            8,
            &constants,
            &p.zero(),
            &RunContext::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    let token = CancellationToken::default();
    let cancel = token.clone();
    let context = RunContext {
        cancellation: token,
        progress: Some(Arc::new(move |_| cancel.cancel())),
    };
    assert!(matches!(
        prepared.evaluate_solution(p, &parameters, 8, &constants, &point, &context),
        Err(Error::Cancelled)
    ));
    assert!(p.close(
        &prepared.evaluate_solution(
            p,
            &parameters,
            8,
            &constants,
            &point,
            &RunContext::default()
        )?[0],
        &p.rational(&Rational::from((1, 4))),
        50
    ));
    Ok(())
}
