use symbolica::prelude::*;
use symbolica_amflow::{local_coordinates::LocalCoordinate, *};

#[test]
fn finite_epsilon_consumer_retains_canceled_domain_poles() -> Result<()> {
    let eta = symbol!("compiled_domain_consumer::eta");
    let p = Precision::decimal(70)?;
    // A reduced connection can be completely regular after cancellation while
    // its derivation still requires eta != 1. Exercise the public point-flow
    // consumer, without supplying an epsilon-expansion cache or fake boundary.
    let system = DifferentialSystem {
        variable: eta,
        matrix: vec![vec![Atom::zero()]],
    };
    let mut compiled = system.compile(p, &Default::default())?;
    assert!(compiled.poles.is_empty());
    compiled.exclude_polynomials(&[Atom::var(eta) - Atom::one()])?;
    assert!(
        compiled
            .poles
            .iter()
            .any(|value| p.close(value, &p.i(1), 55))
    );
    assert!(compiled.endpoint_radius(&p.zero()) < p.real(1));
    for side in [1, -1] {
        let path = compiled.plan_path(&p.zero(), &p.i(2), side)?;
        assert!(path.iter().any(|point| point.im != p.real(0)));
        for coordinate in [LocalCoordinate::Identity, LocalCoordinate::BalancedMobius] {
            let result = compiled.transport(
                &BoundaryData {
                    point: p.zero(),
                    values: vec![p.i(7)],
                },
                &path,
                &FlowOptions {
                    digits: 30,
                    guard_digits: 40,
                    series_order: 48,
                    local_coordinate: coordinate,
                    ..Default::default()
                },
                &RunContext::default(),
            )?;
            assert_eq!(result.point, p.i(2));
            assert!(p.close(&result.values[0], &p.i(7), 50));
        }
    }
    assert!(matches!(
        compiled.exclude_polynomials(&[Atom::zero()]),
        Err(Error::InvalidInput(_))
    ));
    Ok(())
}
