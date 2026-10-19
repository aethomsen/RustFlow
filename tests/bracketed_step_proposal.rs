use symbolica_amflow::{local_coordinates::LocalCoordinate, symbolica::prelude::*, *};

#[test]
fn bracketed_identity_uses_fewer_charts_for_polynomial_solution() -> Result<()> {
    let p = Precision::decimal(70)?;
    let x = symbol!("bracket_polynomial::x");
    let system = DifferentialSystem {
        variable: x,
        matrix: vec![vec![Atom::num(2) / (Atom::num(1) + Atom::var(x))]],
    }
    .compile(p, &Default::default())?;
    let boundary = BoundaryData {
        point: p.zero(),
        values: vec![p.i(1)],
    };
    let mut results = Vec::new();
    for strategy in [StepSizeStrategy::Halving, StepSizeStrategy::Bracketed] {
        let answer = system.transport(
            &boundary,
            &[p.i(100)],
            &FlowOptions {
                step_size_strategy: strategy,
                local_coordinate: LocalCoordinate::Identity,
                digits: 40,
                guard_digits: 20,
                series_order: 12,
                max_steps: 3000,
                ..Default::default()
            },
            &RunContext::default(),
        )?;
        assert_eq!(answer.point, p.i(100));
        assert!(p.close(&answer.values[0], &p.i(10201), 45));
        assert_eq!(
            answer.diagnostics.predicate_evaluations,
            answer.diagnostics.steps
                + answer.diagnostics.rejected_steps
                + answer.diagnostics.superseded_successes
        );
        results.push(answer);
    }
    assert!(p.close(&results[0].values[0], &results[1].values[0], 45));
    eprintln!(
        "halving_charts={} bracketed_charts={}",
        results[0].diagnostics.steps, results[1].diagnostics.steps
    );
    assert!(results[1].diagnostics.steps < results[0].diagnostics.steps);
    Ok(())
}
