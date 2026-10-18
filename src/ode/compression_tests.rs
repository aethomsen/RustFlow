use super::*;

fn regular_with_spurious_map_pole() -> Result<StateCompression> {
    let x = symbol!("compression_test::x");
    DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::Zero, Atom::one()], vec![Atom::Zero, Atom::Zero]],
    }
    .compress_states(
        &[
            vec![Atom::one()],
            vec![Atom::one() / (Atom::var(x) - Atom::num(3))],
        ],
        &[0],
        &[],
        &RunContext::default(),
    )
}

#[test]
fn nonconstant_embedding_preserves_full_transport_and_finite_endpoint() -> Result<()> {
    let x = symbol!("compression_transport::x");
    let t = Atom::var(x);
    // y1=x, y2=2, y3=x*y1+y2/x is a nonconstant invariant embedding.
    let old = DifferentialSystem {
        variable: x,
        matrix: vec![
            vec![Atom::one() / &t, Atom::Zero, Atom::Zero],
            vec![Atom::Zero, Atom::Zero, Atom::Zero],
            vec![Atom::num(2), -Atom::one() / t.clone().pow(2), Atom::Zero],
        ],
    };
    let embedding = vec![
        vec![Atom::one(), Atom::Zero],
        vec![Atom::Zero, Atom::one()],
        vec![t.clone(), Atom::one() / &t],
    ];
    let context = RunContext::default();
    let exact = old.compress_states(&embedding, &[0, 1], &[], &context)?;
    assert_eq!(exact.selected_old_rows(), &[0, 1]);
    assert_eq!(
        exact.reduced_system().matrix,
        vec![
            vec![Atom::one() / &t, Atom::Zero],
            vec![Atom::Zero, Atom::Zero]
        ]
    );
    let p = Precision::decimal(80)?;
    let compressed = exact.compile(p, &Default::default(), &context)?;
    let full = BoundaryData {
        point: p.i(1),
        values: vec![p.i(1), p.i(2), p.i(3)],
    };
    let small = compressed.project_boundary(&full, 60)?;
    let options = FlowOptions {
        digits: 50,
        guard_digits: 30,
        series_order: 80,
        ..Default::default()
    };
    let full_end =
        old.compile(p, &Default::default())?
            .transport(&full, &[p.i(2)], &options, &context)?;
    let small_end = compressed
        .reduced_system()
        .transport(&small, &[p.i(2)], &options, &context)?;
    let restored = compressed.reconstruct(&small_end.point, &small_end.values)?;
    for ((a, b), expected) in full_end
        .values
        .iter()
        .zip(&restored)
        .zip([p.i(2), p.i(2), p.i(5)])
    {
        assert!(p.close(a, b, 50));
        assert!(p.close(b, &expected, 50));
    }
    Ok(())
}

#[test]
fn spurious_map_pole_is_preserved_even_for_a_regular_original_system() -> Result<()> {
    let exact = regular_with_spurious_map_pole()?;
    let x = Atom::var(symbol!("compression_test::x"));
    assert!(
        (exact.reduced_system().matrix[0][0].clone() - Atom::one() / (&x - Atom::num(3)))
            .together()
            .cancel()
            .is_zero()
    );
    assert!(
        exact
            .nonzero_conditions()
            .iter()
            .any(|a| (a - (&x - Atom::num(3))).together().cancel().is_zero())
    );
    let p = Precision::decimal(60)?;
    let compiled = exact.compile(p, &Default::default(), &RunContext::default())?;
    assert!(
        compiled
            .reduced_system()
            .poles
            .iter()
            .any(|z| p.close(z, &p.i(3), 50))
    );
    assert!(compiled.reconstruct(&p.i(3), &[p.zero()]).is_err());
    assert!(
        compiled
            .project_boundary(
                &BoundaryData {
                    point: p.i(3),
                    values: vec![p.zero(), p.i(1)]
                },
                40
            )
            .is_err()
    );
    let projected = compiled.project_boundary(
        &BoundaryData {
            point: p.i(4),
            values: vec![p.i(1), p.i(1)],
        },
        40,
    )?;
    assert_eq!(projected.values, vec![p.i(1)]);
    Ok(())
}

#[test]
fn original_canceled_domain_and_declared_parameter_condition_survive() -> Result<()> {
    let x = symbol!("compression_domains::x");
    let epsilon = symbol!("compression_domains::epsilon");
    let old = DifferentialSystem {
        variable: x,
        matrix: vec![
            vec![Atom::Zero, Atom::one() / (Atom::var(x) - Atom::num(2))],
            vec![Atom::Zero, Atom::Zero],
        ],
    };
    let exact = old.compress_states(
        &[vec![Atom::one()], vec![Atom::Zero]],
        &[0],
        &[Atom::var(epsilon) - Atom::one()],
        &RunContext::default(),
    )?;
    assert_eq!(exact.reduced_system().matrix, vec![vec![Atom::Zero]]);
    let p = Precision::decimal(60)?;
    let values = ahash::HashMap::from_iter([(Atom::var(epsilon), p.i(2))]);
    let compiled = exact.compile(p, &values, &RunContext::default())?;
    assert!(
        compiled
            .reduced_system()
            .poles
            .iter()
            .any(|z| p.close(z, &p.i(2), 50))
    );
    assert!(compiled.reconstruct(&p.i(2), &[p.i(1)]).is_err());
    assert!(
        exact
            .compile(
                p,
                &ahash::HashMap::from_iter([(Atom::var(epsilon), p.i(1))]),
                &RunContext::default()
            )
            .is_err()
    );
    assert_eq!(
        compiled.reconstruct(&p.i(3), &[p.i(5)])?,
        vec![p.i(5), p.zero()]
    );
    Ok(())
}

#[test]
fn invalid_shape_selector_identity_or_intertwining_is_rejected() -> Result<()> {
    let x = symbol!("compression_rejections::x");
    let zero = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::Zero; 2]; 2],
    };
    let identity = vec![vec![Atom::one(), Atom::Zero], vec![Atom::Zero, Atom::one()]];
    let context = RunContext::default();
    assert!(
        zero.compress_states(&identity, &[0, 0], &[], &context)
            .is_err()
    );
    assert!(
        zero.compress_states(&identity, &[0, 2], &[], &context)
            .is_err()
    );
    assert!(
        zero.compress_states(&[vec![Atom::one()], vec![]], &[0], &[], &context)
            .is_err()
    );
    assert!(
        zero.compress_states(&[vec![Atom::num(2)], vec![Atom::Zero]], &[0], &[], &context)
            .is_err()
    );
    assert!(
        zero.compress_states(
            &[vec![Atom::one()], vec![Atom::var(x)]],
            &[0],
            &[],
            &context
        )
        .is_err()
    );
    assert!(
        zero.compress_states(
            &[vec![Atom::one()], vec![Atom::Zero]],
            &[0],
            &[Atom::Zero],
            &context
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn incompatible_boundary_and_invalid_finite_state_are_rejected() -> Result<()> {
    let exact = regular_with_spurious_map_pole()?;
    let p = Precision::decimal(60)?;
    let compiled = exact.compile(p, &Default::default(), &RunContext::default())?;
    assert!(
        compiled
            .project_boundary(
                &BoundaryData {
                    point: p.i(4),
                    values: vec![p.i(1), p.i(2)]
                },
                40
            )
            .is_err()
    );
    assert!(
        compiled
            .project_boundary(
                &BoundaryData {
                    point: p.i(4),
                    values: vec![p.i(1)]
                },
                40
            )
            .is_err()
    );
    assert!(
        compiled
            .project_boundary(
                &BoundaryData {
                    point: p.i(4),
                    values: vec![p.i(1), p.i(1)]
                },
                0
            )
            .is_err()
    );
    assert!(
        compiled
            .project_boundary(
                &BoundaryData {
                    point: p.i(4),
                    values: vec![p.i(1), p.i(1)]
                },
                1000
            )
            .is_err()
    );
    assert!(compiled.reconstruct(&p.i(4), &[]).is_err());
    assert!(
        exact
            .compile(
                p,
                &ahash::HashMap::from_iter([(Atom::var(symbol!("compression_test::x")), p.i(4))]),
                &RunContext::default()
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn canceled_context_stops_exact_preparation_and_specialization() -> Result<()> {
    let exact = regular_with_spurious_map_pole()?;
    let context = RunContext::default();
    context.cancellation.cancel();
    assert!(
        exact
            .compile(Precision::decimal(40)?, &Default::default(), &context)
            .is_err()
    );
    assert!(
        exact
            .reduced_system()
            .compress_states(&[vec![Atom::one()]], &[0], &[], &context)
            .is_err()
    );
    Ok(())
}
