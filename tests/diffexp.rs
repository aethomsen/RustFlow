use symbolica::prelude::*;
use symbolica_amflow::diffexp::*;
use symbolica_amflow::*;

#[test]
fn pinned_diffexp_polylogarithm_oracle_from_independent_boundary_series() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/diffexp/mpl-101-4.json")).unwrap();
    let matrix = fixture["system"]["matrix_epsilon0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|a| {
                    Atom::parse(a.as_str().unwrap(), "mpl_oracle", Default::default()).unwrap()
                })
                .collect()
        })
        .collect();
    let system = EpsilonSystem {
        variable: symbol!("mpl_oracle::t"),
        matrices: vec![matrix],
    };
    let result = transport_epsilon(
        &system,
        |p| {
            // These convergent Taylor sums follow directly from G_1'=1/(t-1),
            // G_01'=G_1/t and G_101'=G_01/(t-1), with zero regularized values.
            // No upstream numerical reference enters this boundary construction.
            let point = p.rational(&Rational::from((1, 8)));
            let mut values = vec![p.zero(), p.zero(), p.zero(), p.i(1)];
            let mut power = p.i(1);
            let mut harmonic = p.zero();
            for n in 1..=i64::from(p.bits) {
                power = p.mul(&power, &point);
                values[0] = p.add(&values[0], &p.scale(&p.mul(&power, &harmonic), 1, n));
                values[1] = p.sub(&values[1], &p.scale(&power, 1, n * n));
                values[2] = p.sub(&values[2], &p.scale(&power, 1, n));
                harmonic = p.add(&harmonic, &p.scale(&p.i(1), 1, n * n));
            }
            Ok(EpsilonBoundary {
                point,
                leading: 0,
                coefficients: vec![values],
            })
        },
        &[
            Atom::num(symbolica::domains::float::Complex::new(
                Rational::from(1),
                Rational::from((-1, 2)),
            )),
            Atom::num(symbolica::domains::float::Complex::new(
                Rational::from(2),
                Rational::from((-1, 2)),
            )),
            Atom::num(4),
        ],
        &FlowOptions {
            digits: 30,
            ..Default::default()
        },
        &RunContext::default(),
        true,
    )
    .unwrap();
    assert_eq!(result.verified_digits, Some(30));
    let p = Precision {
        bits: result.diagnostics.working_bits,
    };
    let references = fixture["references"].as_array().unwrap();
    let refined = references.iter().find(|v| v["order"] == 75).unwrap();
    for (actual, reference) in result.coefficients[0]
        .iter()
        .zip(refined["values"].as_array().unwrap())
    {
        let expected = p
            .parse(
                reference["real"].as_str().unwrap(),
                reference["imaginary"].as_str().unwrap(),
            )
            .unwrap();
        // Upstream's 250-digit arithmetic is not its achieved accuracy. Its
        // order75 result was checked against a 35–36-digit notebook reference.
        assert!(p.close(actual, &expected, 30), "{actual} != {expected}");
    }
    let logarithm = p.add(&p.log(&p.i(3)), &p.log(&p.i(-1)));
    assert!(p.close(&result.coefficients[0][2], &logarithm, 50));
}

#[test]
fn direct_laurent_transport_and_saved_intermediate_points() {
    let system = EpsilonSystem::from_differential_system(
        &DifferentialSystem {
            variable: symbol!("x"),
            matrix: vec![vec![parse!("(2+eps)/(1+x)")]],
        },
        symbol!("eps"),
        4,
    )
    .unwrap();
    let result = transport_epsilon(
        &system,
        |p| {
            Ok(EpsilonBoundary {
                point: p.zero(),
                leading: -2,
                coefficients: vec![
                    vec![p.i(1)],
                    vec![p.zero()],
                    vec![p.zero()],
                    vec![p.zero()],
                    vec![p.zero()],
                ],
            })
        },
        &[Atom::num(1)],
        &FlowOptions::default(),
        &RunContext::default(),
        true,
    )
    .unwrap();
    assert_eq!(result.verified_digits, Some(20));
    assert_eq!(result.leading, -2);
    assert!(!result.comparison_errors.is_empty());
    let p = Precision {
        bits: result.diagnostics.working_bits,
    };
    let mut factorial = 1;
    for (k, row) in result.coefficients.iter().enumerate() {
        if k > 0 {
            factorial *= k as i64;
        }
        assert!(p.close(
            &row[0],
            &p.scale(&p.powi(&p.log(&p.i(2)), k as i64), 4, factorial),
            40
        ));
    }
    for (i, segment) in result.segments.iter().enumerate() {
        let point = p.scale(&p.add(&segment.center, &segment.end), 1, 2);
        let value = result.evaluate_segment(i, &point).unwrap();
        let argument = p.add(&p.i(1), &point);
        let log = p.log(&argument);
        let mut factorial = 1;
        for (k, row) in value.iter().enumerate() {
            if k > 0 {
                factorial *= k as i64;
            }
            let expected = p.scale(
                &p.mul(&p.powi(&argument, 2), &p.powi(&log, k as i64)),
                1,
                factorial,
            );
            assert!(p.close(&row[0], &expected, 35));
        }
        assert!(
            result
                .evaluate_segment(i, &p.add(&segment.end, &p.i(100)))
                .is_err()
        );
    }
}

#[test]
fn coupled_noncommuting_epsilon_orders() {
    let system = EpsilonSystem::from_differential_system(
        &DifferentialSystem {
            variable: symbol!("x"),
            matrix: vec![
                vec![Atom::new(), Atom::num(1)],
                vec![parse!("eps"), Atom::new()],
            ],
        },
        symbol!("eps"),
        5,
    )
    .unwrap();
    let p = Precision::decimal(65).unwrap();
    let mut coefficients = vec![vec![p.zero(); 2]; 6];
    coefficients[0][0] = p.i(1);
    let result = system
        .compile(p, &Default::default())
        .unwrap()
        .transport(
            &EpsilonBoundary {
                point: p.zero(),
                leading: 0,
                coefficients,
            },
            &[p.i(2)],
            &FlowOptions::default(),
            &RunContext::default(),
            false,
        )
        .unwrap();
    for (k, row) in result.coefficients.iter().enumerate() {
        let factorial = (1..=2 * k).product::<usize>() as i64;
        assert!(p.close(
            &row[0],
            &p.scale(&p.powi(&p.i(2), 2 * k as i64), 1, factorial),
            45
        ));
        let expected = if k == 0 {
            p.zero()
        } else {
            p.scale(
                &p.powi(&p.i(2), (2 * k - 1) as i64),
                1,
                (1..2 * k).product::<usize>() as i64,
            )
        };
        assert!(p.close(&row[1], &expected, 45));
    }
    assert_eq!(result.verified_digits, None);
}

#[test]
fn rational_epsilon_dependence_and_rejected_matrix_poles() {
    let source = DifferentialSystem {
        variable: symbol!("x"),
        matrix: vec![vec![parse!("eps/((1-eps)*(1+x))")]],
    };
    let system = EpsilonSystem::from_differential_system(&source, symbol!("eps"), 4).unwrap();
    assert!(system.matrices[0][0][0].is_zero());
    for matrix in system.matrices.iter().skip(1) {
        assert!(
            (&matrix[0][0] - parse!("1/(1+x)"))
                .together()
                .cancel()
                .is_zero()
        );
    }
    assert!(matches!(
        EpsilonSystem::from_differential_system(
            &DifferentialSystem {
                variable: symbol!("x"),
                matrix: vec![vec![parse!("1/eps+x")]],
            },
            symbol!("eps"),
            4
        ),
        Err(Error::Unsupported(_))
    ));
    assert!(EpsilonSystem::from_differential_system(&source, symbol!("x"), 1).is_err());
}

#[test]
fn coefficient_transport_preserves_logarithm_monodromy() {
    let system = EpsilonSystem::from_differential_system(
        &DifferentialSystem {
            variable: symbol!("x"),
            matrix: vec![vec![parse!("eps/x")]],
        },
        symbol!("eps"),
        4,
    )
    .unwrap();
    let p = Precision::decimal(65).unwrap();
    let compiled = system.compile(p, &Default::default()).unwrap();
    for side in [-1, 1] {
        let path = compiled.plan_path(&p.i(1), &p.i(-1), side).unwrap();
        let result = compiled
            .transport(
                &EpsilonBoundary {
                    point: p.i(1),
                    leading: 0,
                    coefficients: vec![
                        vec![p.i(1)],
                        vec![p.zero()],
                        vec![p.zero()],
                        vec![p.zero()],
                        vec![p.zero()],
                    ],
                },
                &path,
                &FlowOptions::default(),
                &RunContext::default(),
                true,
            )
            .unwrap();
        // Path normal is relative to the leftward directed segment: side +1
        // goes below the origin and therefore gives log(-1)=-i*pi.
        let logarithm = p.scale(&p.log(&p.i(-1)), -side, 1);
        let mut factorial = 1;
        for (k, row) in result.coefficients.iter().enumerate() {
            if k > 0 {
                factorial *= k as i64;
            }
            assert!(p.close(
                &row[0],
                &p.scale(&p.powi(&logarithm, k as i64), 1, factorial),
                35
            ));
        }
        // Restarting at an accepted intermediate point must preserve branch.
        let middle = result.segments.len() / 2;
        let point = result.segments[middle].end.clone();
        let restarted = compiled
            .transport(
                &EpsilonBoundary {
                    coefficients: result.evaluate_segment(middle, &point).unwrap(),
                    point,
                    leading: 0,
                },
                &result.segments[middle + 1..]
                    .iter()
                    .map(|s| s.end.clone())
                    .collect::<Vec<_>>(),
                &FlowOptions::default(),
                &RunContext::default(),
                false,
            )
            .unwrap();
        for (a, b) in result
            .coefficients
            .iter()
            .flatten()
            .zip(restarted.coefficients.iter().flatten())
        {
            assert!(p.close(a, b, 35));
        }
    }
}

fn epsilon_domain_source(text: &str) -> DifferentialSystem {
    DifferentialSystem {
        variable: symbol!("epsilon_input_domain::x"),
        matrix: vec![vec![
            Atom::parse(text, "epsilon_input_domain", Default::default()).unwrap(),
        ]],
    }
}

#[test]
fn epsilon_domain_rejects_fractional_power() {
    assert!(matches!(
        EpsilonSystem::from_differential_system(
            &epsilon_domain_source("eps^(1/2)"),
            symbol!("epsilon_input_domain::eps"),
            5,
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn epsilon_domain_rejects_fractional_power_above_requested_order() {
    assert!(matches!(
        EpsilonSystem::from_differential_system(
            &epsilon_domain_source("eps^(101/2)"),
            symbol!("epsilon_input_domain::eps"),
            2,
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn epsilon_domain_rejects_fractional_function() {
    assert!(matches!(
        EpsilonSystem::from_differential_system(
            &epsilon_domain_source("symbolica::exp(eps^(1/2))"),
            symbol!("epsilon_input_domain::eps"),
            4,
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn epsilon_domain_rejects_mixed_integer_and_fractional_input() {
    assert!(matches!(
        EpsilonSystem::from_differential_system(
            &epsilon_domain_source("1/(1-eps)+eps^(1/2)"),
            symbol!("epsilon_input_domain::eps"),
            3,
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn epsilon_domain_preserves_independent_exact_coefficients() {
    for coefficient in ["x^(1/2)", "symbolica::gamma(1/3)"] {
        let source = epsilon_domain_source(&format!("({coefficient})/(1-eps)"));
        let expected =
            Atom::parse(coefficient, "epsilon_input_domain", Default::default()).unwrap();
        let expanded = EpsilonSystem::from_differential_system(
            &source,
            symbol!("epsilon_input_domain::eps"),
            5,
        )
        .unwrap();
        assert_eq!(expanded.matrices.len(), 6);
        for matrix in expanded.matrices {
            assert!((&matrix[0][0] - &expected).together().cancel().is_zero());
        }
    }
}
