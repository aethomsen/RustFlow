//! Regressions for solution contraction; independent special-function oracles
//! catch small logarithm rows that a basis-only comparison can obscure.
use super::*;

fn prepare(x: Symbol, matrix: Vec<Vec<Atom>>) -> Result<PreparedFrobenius> {
    DifferentialSystem {
        variable: x,
        matrix,
    }
    .prepare_frobenius(&RunContext::default())
}

fn weighted(p: Precision, matrix: Vec<Vec<C>>, constants: &[C]) -> Vec<C> {
    matrix
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(constants)
                .fold(p.zero(), |sum, (v, c)| p.add(&sum, &p.mul(v, c)))
        })
        .collect()
}

fn legacy(
    prepared: &PreparedFrobenius,
    p: Precision,
    parameters: &ahash::HashMap<Atom, C>,
    order: usize,
    constants: &[C],
    point: &C,
    winding: i32,
) -> Result<Vec<C>> {
    let mut result = vec![p.zero(); constants.len()];
    prepared.visit_values(
        p,
        parameters,
        order,
        point,
        winding,
        &RunContext::default(),
        None,
        |indices, values| {
            assert_eq!(indices.len(), 1);
            for (sum, value) in result.iter_mut().zip(values) {
                *sum = p.add(sum, &p.mul(&constants[indices.start], &value));
            }
            Ok(())
        },
    )?;
    Ok(result)
}

fn ranges(
    prepared: &PreparedFrobenius,
    p: Precision,
    parameters: &ahash::HashMap<Atom, C>,
    order: usize,
    constants: &[C],
    point: &C,
) -> Result<Vec<usize>> {
    let mut result = Vec::new();
    prepared.visit_values(
        p,
        parameters,
        order,
        point,
        0,
        &RunContext::default(),
        Some(constants),
        |indices, _| {
            result.push(indices.len());
            Ok(())
        },
    )?;
    Ok(result)
}

// For these identity-gauge fixtures the leading power-zero vectors form the
// original coordinate basis. Reuse native solving rather than assume its order.
fn initial_constants(prepared: &PreparedFrobenius, p: Precision, initial: &[C]) -> Result<Vec<C>> {
    assert!(prepared.identity_transformation());
    let columns = prepared
        .eigenspaces
        .iter()
        .flat_map(|s| &s.leading)
        .collect::<Vec<_>>();
    let matrix = (0..initial.len())
        .map(|i| {
            columns
                .iter()
                .map(|c| p.eval(&c[0][i], &Default::default()))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    solve(p, matrix, initial.to_vec())
}

fn relative(p: Precision, actual: &C, expected: &C, digits: u32) {
    assert!(p.finite(actual) && p.finite(expected));
    assert_ne!(*expected, p.zero());
    let error = p.norm(&p.sub(actual, expected)) / p.norm(expected);
    assert!(
        error < p.tolerance(digits),
        "relative error {error} at {digits} digits"
    );
}

#[test]
fn public_grouped_solution_matches_legacy_basis_across_branches() -> Result<()> {
    let x = symbol!("contracted_solution_tests::x");
    let eps = symbol!("contracted_solution_tests::epsilon");
    let z = Atom::var(x);
    let e = Atom::var(eps);
    let mut repeated = vec![vec![Atom::Zero; 4]; 4];
    let mut jordan = repeated.clone();
    let mut distinct = repeated.clone();
    for i in 0..4 {
        repeated[i][i] = &e / &z + Atom::one();
        jordan[i][i] = &e / &z;
        if i < 2 {
            jordan[i][i + 1] = Atom::one() / &z;
        }
        distinct[i][i] = Atom::num(if i < 2 { 1 } else { 2 }) * &e / &z;
    }
    let resonant = vec![
        vec![&e / &z, Atom::one() / &z, Atom::Zero],
        vec![Atom::Zero, &e / &z, Atom::Zero],
        vec![Atom::one(), Atom::Zero, (&e + Atom::one()) / &z],
    ];
    let sheared = vec![
        vec![z.clone().pow(-2), z.clone().pow(-2)],
        vec![-z.clone().pow(-2), -z.clone().pow(-2)],
    ];
    let mut checks = 0;
    for matrix in [repeated, jordan, resonant, sheared, distinct] {
        let prepared = prepare(x, matrix)?;
        let n = prepared.source.matrix.len();
        for digits in [80, 140] {
            let p = Precision::decimal(digits)?;
            let parameters =
                ahash::HashMap::from_iter([(e.clone(), p.rational(&Rational::from((1, 97))))]);
            let constants = (0..n)
                .map(|i| p.parse(&(i + 2).to_string(), &(i as i64 - 1).to_string()))
                .collect::<Result<Vec<_>>>()?;
            let basis = prepared.evaluate(p, &parameters, 28, &RunContext::default())?;
            for point in [
                p.rational(&Rational::from((1, 10))),
                p.parse("0.05", "0.07")?,
            ] {
                for winding in [-1, 0, 2] {
                    let actual = prepared.evaluate_solution_with_winding(
                        p,
                        &parameters,
                        28,
                        &constants,
                        &point,
                        winding,
                        &RunContext::default(),
                    )?;
                    let old = legacy(&prepared, p, &parameters, 28, &constants, &point, winding)?;
                    let full = weighted(
                        p,
                        basis.evaluate_with_winding(&point, &parameters, winding)?,
                        &constants,
                    );
                    for ((a, b), c) in actual.iter().zip(old).zip(full) {
                        assert!(p.close(a, &b, digits - 25));
                        assert!(p.close(a, &c, digits - 25));
                    }
                    let matched = prepared.match_values_with_winding(
                        p,
                        &parameters,
                        28,
                        &point,
                        &actual,
                        winding,
                        &RunContext::default(),
                    )?;
                    for (a, c) in matched.iter().zip(&constants) {
                        assert!(p.close(a, c, digits - 25));
                    }
                    checks += 1;
                }
            }
        }
    }
    assert_eq!(checks, 60);
    Ok(())
}

#[test]
fn nonresonant_tiny_log_rows_survive_amplification() -> Result<()> {
    let x = symbol!("contracted_tiny_log::x");
    let z = Atom::var(x);
    let a = -Atom::num(12) / &z + Atom::one();
    let prepared = prepare(
        x,
        vec![vec![a.clone(), Atom::one() / &z], vec![Atom::Zero, a]],
    )?;
    let mut refined: Option<Vec<C>> = None;
    for digits in [100, 160] {
        let p = Precision::decimal(digits)?;
        let point = p.parse("1e-10", "0")?;
        let c = p.parse("1e-120", "0")?;
        let constants = initial_constants(&prepared, p, &[p.zero(), c])?;
        assert_eq!(
            ranges(&prepared, p, &Default::default(), 28, &constants, &point)?,
            vec![2]
        );
        for winding in [-1, 0, 2] {
            let actual = prepared.evaluate_solution_with_winding(
                p,
                &Default::default(),
                28,
                &constants,
                &point,
                winding,
                &RunContext::default(),
            )?;
            // Independent exp/log oracle: x^-12*c=1 exactly in the mathematical fixture.
            let exponential = p.exp(&point);
            let logarithm = if winding == 0 {
                p.log(&point)
            } else {
                p.add(
                    &p.log(&point),
                    &p.scale(&p.log(&p.i(-1)), 2 * i64::from(winding), 1),
                )
            };
            relative(p, &actual[0], &p.mul(&exponential, &logarithm), digits - 25);
            relative(p, &actual[1], &exponential, digits - 25);
            if winding == 0 {
                if let Some(old) = &refined {
                    for (a, b) in actual.iter().zip(old) {
                        relative(p, a, &p.round(b), 75);
                    }
                }
                refined = Some(actual);
            }
        }
    }
    Ok(())
}

#[test]
fn resonant_tiny_log_keeps_the_legacy_route() -> Result<()> {
    let x = symbol!("contracted_resonant_log::x");
    let z = Atom::var(x);
    let prepared = prepare(
        x,
        vec![
            vec![-Atom::num(12) / &z, Atom::one()],
            vec![Atom::Zero, -Atom::num(13) / &z],
        ],
    )?;
    for digits in [100, 160] {
        let p = Precision::decimal(digits)?;
        let point = p.parse("1e-10", "0")?;
        let constants = initial_constants(&prepared, p, &[p.zero(), p.parse("1e-120", "0")?])?;
        assert_eq!(
            ranges(&prepared, p, &Default::default(), 8, &constants, &point)?,
            vec![1, 1]
        );
        let actual = prepared.evaluate_solution(
            p,
            &Default::default(),
            8,
            &constants,
            &point,
            &RunContext::default(),
        )?;
        relative(p, &actual[0], &p.log(&point), digits - 25);
        relative(p, &actual[1], &p.parse("1e10", "0")?, digits - 25);
    }
    Ok(())
}

#[test]
fn nonresonant_contraction_preserves_cancellation_and_tiny_scalars() -> Result<()> {
    let x = symbol!("contracted_cancellation::x");
    let z = Atom::var(x);
    let jordan = prepare(
        x,
        vec![
            vec![Atom::Zero, Atom::one() / &z],
            vec![Atom::Zero, Atom::Zero],
        ],
    )?;
    for digits in [160, 220] {
        let p = Precision::decimal(digits)?;
        let point = p.rational(&Rational::from((1, 10)));
        let tiny = p.parse("1e-50", "0")?;
        let log = p.log(&point);
        for target in [p.zero(), tiny.clone()] {
            let constants = initial_constants(&jordan, p, &[p.sub(&target, &log), p.i(1)])?;
            assert_eq!(
                ranges(&jordan, p, &Default::default(), 12, &constants, &point)?,
                vec![2]
            );
            let actual = jordan.evaluate_solution(
                p,
                &Default::default(),
                12,
                &constants,
                &point,
                &RunContext::default(),
            )?;
            let old = legacy(&jordan, p, &Default::default(), 12, &constants, &point, 0)?;
            if target == p.zero() {
                assert!(p.norm(&actual[0]) < p.tolerance(digits - 15));
            } else {
                relative(p, &actual[0], &target, digits - 65);
            }
            assert!(p.close(&actual[0], &old[0], digits - 15));
            relative(p, &actual[1], &p.i(1), digits - 15);
        }
    }
    let scalar = prepare(x, vec![vec![-Atom::num(12) / &z]])?;
    let p = Precision::decimal(100)?;
    let actual = scalar.evaluate_solution(
        p,
        &Default::default(),
        8,
        &[p.parse("1e-120", "0")?],
        &p.parse("1e-10", "0")?,
        &RunContext::default(),
    )?;
    relative(p, &actual[0], &p.i(1), 75);
    Ok(())
}

#[test]
fn requested_order_controls_resonance_fallback_and_exact_groups_stay_separate() -> Result<()> {
    let x = symbol!("contracted_order_scope::x");
    let z = Atom::var(x);
    let p = Precision::decimal(80)?;
    let point = p.rational(&Rational::from((1, 10)));
    let prepared = prepare(
        x,
        vec![
            vec![Atom::Zero; 3],
            vec![Atom::Zero; 3],
            vec![Atom::Zero, Atom::Zero, Atom::num(2) / &z],
        ],
    )?;
    let constants = vec![p.i(1); 3];
    assert_eq!(
        ranges(&prepared, p, &Default::default(), 1, &constants, &point)?,
        vec![2, 1]
    );
    assert_eq!(
        ranges(&prepared, p, &Default::default(), 2, &constants, &point)?,
        vec![1, 1, 1]
    );
    let eps = symbol!("contracted_order_scope::epsilon");
    let e = Atom::var(eps);
    let mut matrix = vec![vec![Atom::Zero; 4]; 4];
    for i in 0..4 {
        matrix[i][i] = if i < 2 {
            e.clone() / &z
        } else {
            Atom::num(2) * &e / &z
        };
    }
    let distinct = prepare(x, matrix)?;
    let parameters = ahash::HashMap::from_iter([(e, p.rational(&Rational::from((1, 97))))]);
    assert_eq!(
        ranges(&distinct, p, &parameters, 12, &vec![p.i(1); 4], &point)?,
        vec![2, 2]
    );
    Ok(())
}

#[test]
fn unchanged_genericity_rejects_near_positive_integer_resonance() -> Result<()> {
    let x = symbol!("contracted_near_resonance::x");
    let e = parse!("contracted_near_resonance::epsilon");
    let z = Atom::var(x);
    let prepared = prepare(
        x,
        vec![
            vec![&e / &z, Atom::Zero, Atom::Zero],
            vec![Atom::Zero, &e / &z, Atom::Zero],
            vec![Atom::Zero, Atom::Zero, Atom::num(2) / &z],
        ],
    )?;
    let p = Precision::decimal(80)?;
    let parameters = ahash::HashMap::from_iter([(e, p.parse("1e-100", "0")?)]);
    assert!(matches!(
        prepared.evaluate_solution(
            p,
            &parameters,
            4,
            &vec![p.zero(); 3],
            &p.i(1),
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}

#[test]
fn zero_constants_still_enforce_source_point_cancellation_and_resource_guards() -> Result<()> {
    let x = symbol!("contracted_zero_guards::x");
    let e = parse!("contracted_zero_guards::epsilon");
    let a = parse!(
        "(contracted_zero_guards::epsilon^2-1)/((contracted_zero_guards::epsilon-1)*contracted_zero_guards::x)"
    );
    let prepared = prepare(x, vec![vec![a.clone(), Atom::Zero], vec![Atom::Zero, a]])?;
    let p = Precision::decimal(80)?;
    let constants = vec![p.zero(); 2];
    let point = p.rational(&Rational::from((1, 10)));
    let excluded = ahash::HashMap::from_iter([(e.clone(), p.i(1))]);
    assert!(matches!(
        prepared.evaluate_solution(p, &excluded, 8, &constants, &point, &RunContext::default()),
        Err(Error::Numerical(_))
    ));
    let parameters = ahash::HashMap::from_iter([(e, p.i(2))]);
    for badpoint in [p.zero(), p.parse("NaN", "0")?, p.parse("inf", "0")?] {
        assert!(matches!(
            prepared.evaluate_solution(
                p,
                &parameters,
                8,
                &constants,
                &badpoint,
                &RunContext::default()
            ),
            Err(Error::InvalidInput(_))
        ));
    }
    assert!(matches!(
        prepared.evaluate_solution(p, &parameters, 8, &[], &point, &RunContext::default()),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        prepared.evaluate_solution(
            p,
            &parameters,
            8,
            &[p.parse("NaN", "0")?, p.zero()],
            &point,
            &RunContext::default()
        ),
        Err(Error::InvalidInput(_))
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
        prepared.evaluate_solution(
            Precision { bits: u32::MAX },
            &parameters,
            8,
            &constants,
            &point,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
    let context = RunContext::default();
    context.cancellation.cancel();
    assert!(matches!(
        prepared.evaluate_solution(p, &parameters, 8, &constants, &point, &context),
        Err(Error::Cancelled)
    ));
    let actual = prepared.evaluate_solution(
        p,
        &parameters,
        8,
        &constants,
        &point,
        &RunContext::default(),
    )?;
    assert!(actual.iter().all(|v| *v == p.zero()));
    let admission = StreamingAdmission::new(p, 8, u128::MAX, true);
    assert!(matches!(admission.column(2, 8, 2), Err(Error::Limit(_))));
    let admission = StreamingAdmission::new(p, 8, 0, true);
    assert!(matches!(admission.matrix(1_000_000), Err(Error::Limit(_))));
    Ok(())
}

#[test]
fn bounded_synthetic_comparison_reports_work_without_a_speed_assertion() -> Result<()> {
    let x = symbol!("contracted_timing::x");
    let z = Atom::var(x);
    let n = 8;
    let mut matrix = vec![vec![Atom::Zero; n]; n];
    for i in 0..n {
        matrix[i][i] = Atom::one() / &z + Atom::one();
    }
    let prepared = prepare(x, matrix)?;
    let p = Precision::decimal(120)?;
    let constants = vec![p.i(1); n];
    let point = p.rational(&Rational::from((1, 10)));
    let start = std::time::Instant::now();
    let old = legacy(
        &prepared,
        p,
        &Default::default(),
        600,
        &constants,
        &point,
        0,
    )?;
    let old_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let actual = prepared.evaluate_solution(
        p,
        &Default::default(),
        600,
        &constants,
        &point,
        &RunContext::default(),
    )?;
    let grouped_seconds = start.elapsed().as_secs_f64();
    for (a, b) in actual.iter().zip(old) {
        relative(p, a, &b, 95);
    }
    assert_eq!(
        ranges(&prepared, p, &Default::default(), 4, &constants, &point)?,
        vec![8]
    );
    println!(
        "nonresonant-contraction synthetic n={n} order=600 requested_digits=120 original_sweeps=8 contracted_sweeps=1 legacy_seconds={old_seconds:.6} grouped_seconds={grouped_seconds:.6}"
    );
    Ok(())
}
