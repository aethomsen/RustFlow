use symbolica::prelude::*;
use symbolica_amflow::*;

#[test]
fn polynomial_frobenius_rows_match_exact_expanded_recurrence() -> Result<()> {
    let p = Precision::decimal(110)?;
    let x = symbol!("x");
    let systems = [
        vec![
            vec![parse!("99/(100*x)-1/(100*(1-x))"), Atom::new()],
            vec![Atom::num(1), Atom::new()],
        ],
        vec![
            vec![parse!("-1/(101*x)-102/(101*(1-x))"), Atom::new()],
            vec![Atom::num(-1), Atom::new()],
        ],
        vec![
            vec![parse!("1/x+1/(1-x)"), parse!("1/x+1/(2-x)")],
            vec![Atom::new(), parse!("1/x")],
        ],
        vec![
            vec![Atom::new(), Atom::new()],
            vec![parse!("1/(1-x)"), parse!("1/x")],
        ],
        vec![vec![parse!("1/(x*(1+x^2))")]],
    ];
    for matrix in systems {
        let prepared = DifferentialSystem {
            variable: x,
            matrix,
        }
        .prepare_frobenius(&RunContext::default())?;
        let numerical = prepared.evaluate(p, &Default::default(), 48, &RunContext::default())?;
        // The exact coefficient domain retains the rational-entry power-series
        // recurrence, giving an independent convolution for the same ODE.
        let reference = prepared
            .exact_coefficients(
                48,
                &frobenius::ExactFrobeniusLimits::default(),
                &RunContext::default(),
            )?
            .numerical(p);
        for (actual, expected) in numerical.columns.iter().zip(&reference.columns) {
            assert_eq!(actual.exponent, expected.exponent);
            assert_eq!(actual.coefficients.len(), expected.coefficients.len());
            for (actual, expected) in actual.coefficients.iter().zip(&expected.coefficients) {
                for log in 0..actual.len().max(expected.len()) {
                    for component in 0..numerical.columns.len() {
                        let a = actual
                            .get(log)
                            .map(|r| r[component].clone())
                            .unwrap_or_else(|| p.zero());
                        let b = expected
                            .get(log)
                            .map(|r| r[component].clone())
                            .unwrap_or_else(|| p.zero());
                        assert!(
                            p.close(&a, &b, 85),
                            "coefficient mismatch at log {log}, component {component}"
                        );
                    }
                }
            }
        }
        let point = p.rational(&Rational::from((1, 5)));
        let actual = numerical.evaluate(&point, &Default::default())?;
        let expected = reference.evaluate(&point, &Default::default())?;
        for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
            assert!(p.close(a, b, 85));
        }
    }
    Ok(())
}

#[test]
fn extended_frobenius_order_obeys_allocation_admission() -> Result<()> {
    let prepared = DifferentialSystem {
        variable: symbol!("x"),
        matrix: vec![vec![parse!("1/x")]],
    }
    .prepare_frobenius(&RunContext::default())?;
    let p = Precision::decimal(40)?;
    let result = prepared.evaluate(p, &Default::default(), 10_001, &RunContext::default())?;
    assert_eq!(result.columns[0].coefficients.len(), 10_002);
    assert!(matches!(
        prepared.evaluate(p, &Default::default(), 200_001, &RunContext::default()),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        prepared.evaluate(
            Precision { bits: u32::MAX },
            &Default::default(),
            10_001,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
    Ok(())
}

#[test]
fn sparse_high_rank_logarithmic_recurrence_matches_matrix_exponential() {
    let n = 9;
    let x = symbol!("sparse_log_x");
    let variable = Atom::var(x);
    let mut matrix = vec![vec![Atom::new(); n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = Atom::num(2) * &variable;
        if i + 1 < n {
            row[i + 1] = Atom::num(1) / &variable;
        }
    }
    let p = Precision::decimal(60).unwrap();
    let basis = DifferentialSystem {
        variable: x,
        matrix,
    }
    .frobenius(p, &Default::default(), 32)
    .unwrap();
    let point = p.rational(&Rational::from((1, 10)));
    let actual = basis.evaluate(&point, &Default::default()).unwrap();
    let exponential = p.exp(&p.mul(&point, &point));
    let logarithm = p.log(&point);
    for (i, row) in actual.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let expected = if j < i {
                p.zero()
            } else {
                let factorial = (1..=j - i).product::<usize>();
                p.scale(
                    &p.mul(&exponential, &p.powi(&logarithm, (j - i) as i64)),
                    1,
                    factorial as i64,
                )
            };
            assert!(p.close(value, &expected, 40));
        }
    }
}

#[test]
fn jordan_and_integer_resonance() {
    let p = Precision::decimal(70).unwrap();
    let params = ahash::HashMap::default();
    let options = FlowOptions::default();
    for matrix in [
        vec![
            vec![parse!("1/x"), parse!("1/x")],
            vec![Atom::new(), parse!("1/x")],
        ],
        vec![
            vec![Atom::new(), Atom::new()],
            vec![Atom::num(1), parse!("1/x")],
        ],
    ] {
        let system = DifferentialSystem {
            variable: symbol!("x"),
            matrix,
        };
        let basis = system.frobenius(p, &params, 20).unwrap();
        let start = p.parse("0.1", "0").unwrap();
        let end = p.parse("0.2", "0").unwrap();
        let initial = basis.evaluate(&start, &params).unwrap();
        let expected = basis.evaluate(&end, &params).unwrap();
        let compiled = system.compile(p, &params).unwrap();
        for j in 0..2 {
            let result = compiled
                .transport(
                    &BoundaryData {
                        point: start.clone(),
                        values: initial.iter().map(|r| r[j].clone()).collect(),
                    },
                    std::slice::from_ref(&end),
                    &options,
                    &RunContext::default(),
                )
                .unwrap();
            for (i, value) in result.values.iter().enumerate() {
                assert!(
                    p.close(value, &expected[i][j], 24),
                    "{value} != {}",
                    expected[i][j]
                );
            }
        }
    }
}

#[test]
fn bubble_endpoint_and_infinity() {
    let p = Precision::decimal(70).unwrap();
    let params = ahash::HashMap::from_iter([(parse!("eps"), p.parse("0.1", "0").unwrap())]);
    let system = DifferentialSystem {
        variable: symbol!("eta"),
        matrix: vec![
            vec![parse!("(1-eps)/eta"), Atom::new()],
            vec![
                parse!("2*(eps-1)/(eta*(4*eta-1))"),
                parse!("-2*(2*eps-1)/(4*eta-1)"),
            ],
        ],
    };
    let endpoint = system.frobenius(p, &params, 40).unwrap();
    assert_eq!(endpoint.columns.len(), 2);
    let inverted = system.invert_variable(symbol!("z"));
    let infinity = inverted.frobenius(p, &params, 40).unwrap();
    assert_eq!(infinity.columns.len(), 2);
}

#[test]
fn diagonal_shearing_and_basis_derivative() {
    let system = DifferentialSystem {
        variable: symbol!("x"),
        matrix: vec![
            vec![Atom::new(), parse!("1/x^2")],
            vec![Atom::new(), Atom::new()],
        ],
    };
    let (normal, shifts) = system.diagonal_fuchsian_form().unwrap();
    assert_eq!(shifts, vec![0, 1]);
    let residue = normal.residue().unwrap();
    assert_eq!(residue[0][1], Atom::num(1));
    assert_eq!(residue[1][1], Atom::num(-1));
    let p = Precision::decimal(60).unwrap();
    let params = ahash::HashMap::default();
    let basis = system.frobenius(p, &params, 10).unwrap();
    let at = basis
        .evaluate(&p.parse("0.1", "0").unwrap(), &params)
        .unwrap();
    assert_eq!(at.len(), 2);
    let constant = basis
        .match_values(&p.parse("0.1", "0").unwrap(), &[p.i(-10), p.i(1)], &params)
        .unwrap();
    let at = basis
        .evaluate(&p.parse("0.2", "0").unwrap(), &params)
        .unwrap();
    let answer = at
        .iter()
        .map(|row| {
            row.iter()
                .zip(&constant)
                .fold(p.zero(), |s, (a, b)| p.add(&s, &p.mul(a, b)))
        })
        .collect::<Vec<_>>();
    assert!(p.close(&answer[0], &p.i(-5), 40));
    assert!(p.close(&answer[1], &p.i(1), 40));
}

#[test]
fn hidden_nilpotent_pole_is_rotated_before_shearing() {
    let x = symbol!("hidden_x");
    let a = Atom::var(x).pow(-2);
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![vec![a.clone(), a.clone()], vec![-a.clone(), -a]],
    };
    let p = Precision::decimal(70).unwrap();
    let parameters = ahash::HashMap::default();
    let basis = system.frobenius(p, &parameters, 30).unwrap();
    let initial = vec![p.i(2), p.i(3)];
    let constants = basis.match_values(&p.i(1), &initial, &parameters).unwrap();
    let matrix = basis.evaluate(&p.i(2), &parameters).unwrap();
    let y = matrix
        .iter()
        .map(|r| {
            r.iter()
                .zip(&constants)
                .fold(p.zero(), |a, (b, c)| p.add(&a, &p.mul(b, c)))
        })
        .collect::<Vec<_>>();
    assert!(p.close(&y[0], &p.scale(&p.i(1), 9, 2), 50));
    assert!(p.close(&y[1], &p.scale(&p.i(1), 1, 2), 50));
}

#[test]
fn successive_nilpotent_balances_remove_a_cubic_pole() {
    let x = symbol!("balanced_x");
    let z = Atom::var(x);
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![
            vec![
                z.clone().pow(-2),
                z.clone().pow(-1) - z.clone().pow(-2) - z.clone().pow(-3),
            ],
            vec![z.clone().pow(-1), -z.clone().pow(-2)],
        ],
    };
    assert!(system.diagonal_fuchsian_form().is_err());
    let (normal, transformation) = system.fuchsian_form(16).unwrap();
    normal.residue().unwrap();
    assert_eq!(
        normal.matrix,
        system.change_basis(&transformation).unwrap().matrix
    );
    let p = Precision::decimal(70).unwrap();
    let parameters = ahash::HashMap::default();
    let basis = system.frobenius(p, &parameters, 40).unwrap();
    let constants = basis
        .match_values(&p.i(1), &[p.i(2), p.i(1)], &parameters)
        .unwrap();
    let matrix = basis.evaluate(&p.i(2), &parameters).unwrap();
    let values = matrix
        .iter()
        .map(|r| {
            r.iter()
                .zip(&constants)
                .fold(p.zero(), |a, (b, c)| p.add(&a, &p.mul(b, c)))
        })
        .collect::<Vec<_>>();
    // Base solution (x,x), transformed by [[1,1/x],[0,1]].
    assert!(p.close(&values[0], &p.i(3), 40));
    assert!(p.close(&values[1], &p.i(2), 40));
}

#[test]
fn endpoint_projection_combines_divergences_and_handles_half_powers() {
    use symbolica_amflow::frobenius::{FrobeniusBasis, FrobeniusColumn};
    let p = Precision::decimal(60).unwrap();
    let basis = FrobeniusBasis {
        precision: p,
        columns: vec![
            FrobeniusColumn {
                exponent: Atom::num(-1),
                coefficients: vec![vec![vec![p.i(1), p.zero()]], vec![vec![p.i(1), p.zero()]]],
            },
            FrobeniusColumn {
                exponent: Atom::num(-1),
                coefficients: vec![vec![vec![p.i(-1), p.zero()]], vec![vec![p.zero(), p.i(1)]]],
            },
        ],
    };
    let limit = basis
        .physical_limit(&[p.i(1), p.i(1)], symbol!("endpoint_eps"))
        .unwrap();
    assert_eq!(limit, vec![p.i(1), p.i(1)]);
    assert!(matches!(
        basis.physical_limit(&[p.i(1), p.zero()], symbol!("endpoint_eps")),
        Err(Error::Numerical(_))
    ));
    let half = FrobeniusBasis {
        precision: p,
        columns: vec![FrobeniusColumn {
            exponent: Atom::num((1, 2)),
            coefficients: vec![vec![vec![p.i(1)]]],
        }],
    };
    assert_eq!(
        half.physical_limit(&[p.i(1)], symbol!("endpoint_eps"))
            .unwrap(),
        vec![p.zero()]
    );
}

#[test]
fn moser_projector_mixes_jordan_chains() {
    let system = DifferentialSystem {
        variable: symbol!("x"),
        matrix: vec![
            vec![Atom::new(), parse!("1/x^2"), Atom::new(), Atom::new()],
            vec![parse!("1/x"), Atom::new(), parse!("1/x"), Atom::new()],
            vec![Atom::new(), Atom::new(), Atom::new(), parse!("1/x^2")],
            vec![parse!("-1/x"), Atom::new(), parse!("-1/x"), Atom::new()],
        ],
    };
    let (normal, transform) = system.fuchsian_form(32).unwrap();
    let restored = system.change_basis(&transform).unwrap();
    assert_eq!(normal.matrix, restored.matrix);
    assert!(normal.diagonal_fuchsian_form().is_ok());
    let p = Precision::decimal(70).unwrap();
    let parameters = ahash::HashMap::default();
    let basis = system.frobenius(p, &parameters, 40).unwrap();
    let constants = basis
        .match_values(&p.i(1), &[p.i(-1), p.zero(), p.i(2), p.zero()], &parameters)
        .unwrap();
    let values = basis.evaluate(&p.i(2), &parameters).unwrap();
    let log = p.log(&p.i(2));
    let a = p.scale(&p.add(&p.i(1), &log), 1, 2);
    let expected = [p.neg(&a), log.clone(), p.add(&p.i(1), &a), p.neg(&log)];
    for (row, expected) in values.iter().zip(expected) {
        let actual = row
            .iter()
            .zip(&constants)
            .fold(p.zero(), |sum, (a, b)| p.add(&sum, &p.mul(a, b)));
        assert!(p.close(&actual, &expected, 40), "{actual} != {expected}");
    }
}
