use super::*;

#[test]
fn extended_taylor_order_transports_a_nonpolynomial_solution() -> Result<()> {
    let x = symbol!("extended_taylor::x");
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::one() / (Atom::one() - Atom::var(x))]],
    };
    let p = Precision::decimal(80)?;
    let compiled = system.compile(p, &Default::default())?;
    let options = FlowOptions {
        digits: 35,
        guard_digits: 40,
        series_order: 10_001,
        ..Default::default()
    };
    let end = p.rational(&Rational::from((1, 8)));
    let result = compiled.transport(
        &BoundaryData {
            point: p.zero(),
            values: vec![p.i(1)],
        },
        &[end.clone()],
        &options,
        &RunContext::default(),
    )?;
    let expected = p.div(&p.i(1), &p.sub(&p.i(1), &end));
    assert!(p.close(&result.values[0], &expected, 35));
    assert_eq!(result.diagnostics.expansion_order, 10_001);
    Ok(())
}

#[test]
fn extended_taylor_allocation_refuses_precision_or_channel_growth_before_arrays() -> Result<()> {
    let x = symbol!("extended_taylor_budget::x");
    let p = Precision::decimal(40)?;
    let mut compiled = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::one()]],
    }
    .compile(p, &Default::default())?;
    compiled.p = Precision { bits: u32::MAX };
    assert!(matches!(
        compiled.taylor(&p.zero(), &[p.i(1)], 10_001),
        Err(Error::Limit(_))
    ));
    compiled.p = p;
    assert!(matches!(
        compiled.taylor(&p.zero(), &[p.i(1)], 200_001),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        compiled.taylor_channels(&p.zero(), &[p.i(1)], 10_001, usize::MAX),
        Err(Error::InvalidInput(_)) | Err(Error::Limit(_))
    ));
    Ok(())
}

#[test]
fn extended_taylor_estimate_counts_the_full_exact_source_degree() -> Result<()> {
    let x = symbol!("extended_taylor_source::x");
    let p = Precision::decimal(30)?;
    let a = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::one()]],
    }
    .compile(p, &Default::default())?;
    let b = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::var(x).pow(12000)]],
    }
    .compile(p, &Default::default())?;
    assert!(b.taylor_workspace_bytes(10_001, 1) > 2 * a.taylor_workspace_bytes(10_001, 1));
    assert_eq!(
        a.taylor_workspace_bytes(10_001, 3),
        3 * a.taylor_workspace_bytes(10_001, 1)
    );
    Ok(())
}

#[test]
fn extended_transport_counts_retained_segment_storage_before_the_next_chart() -> Result<()> {
    let p = Precision::decimal(40)?;
    let system = DifferentialSystem {
        variable: symbol!("retained_chart_budget::x"),
        matrix: vec![vec![Atom::one()]],
    }
    .compile(p, &Default::default())?;
    // Exaggerated declared storage precision exercises admission without
    // allocating tens of GiB in a regression test.
    let mut saved = vec![TaylorSegment {
        center: p.zero(),
        end: p.i(1),
        coefficients: vec![vec![p.i(1)]; 40],
        pade: None,
        working_bits: u32::MAX,
        coordinate: TaylorCoordinate::Identity,
        conditioning_digits: 20,
    }];
    let options = FlowOptions {
        series_order: 10_001,
        ..Default::default()
    };
    assert!(matches!(
        transport_series(
            &system,
            &BoundaryData {
                point: p.zero(),
                values: vec![p.i(1)]
            },
            &[p.i(1)],
            &options,
            &RunContext::default(),
            Some(&mut saved)
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(saved.len(), 1);
    Ok(())
}

#[test]
fn algebraic_transport_keeps_the_previous_order_limit_without_an_owner_estimate() -> Result<()> {
    use crate::algebraic::AlgebraicSystem;
    let p = Precision::decimal(40)?;
    let system = AlgebraicSystem::ordinary(
        DifferentialSystem {
            variable: symbol!("unestimated_chart_budget::x"),
            matrix: vec![vec![Atom::one()]],
        },
        vec![],
    )
    .compile(p)?;
    let options = FlowOptions {
        series_order: 10_001,
        ..Default::default()
    };
    assert!(matches!(
        system.transport_ordinary(
            &BoundaryData {
                point: p.zero(),
                values: vec![p.i(1)]
            },
            &[p.i(1)],
            &Default::default(),
            &options,
            &RunContext::default(),
            false
        ),
        Err(Error::Unsupported(_))
    ));
    Ok(())
}
