use super::*;

fn series(expression: &str) -> impl Iterator<Item = (Rational, Atom)> {
    let z = symbol!("exact_root_series_grid::z");
    Atom::parse(expression, "exact_root_series_grid", Default::default())
        .unwrap()
        .series(z, 0, 5)
        .unwrap()
        .terms()
        .map(|(power, coefficient)| (power, coefficient.clone()))
        .collect::<Vec<_>>()
        .into_iter()
}

#[test]
fn native_nonconstant_unit_root_grid_ignores_only_exact_zero_fractional_slots() {
    let terms = series("(1-z)^(-1/2)").collect::<Vec<_>>();
    assert!(
        terms
            .iter()
            .any(|(power, coefficient)| !power.is_integer() && coefficient.is_zero())
    );
    let limits = ExactFrobeniusLimits::default();
    let context = RunContext::default();
    let coefficients = regular_root_coefficients(
        terms.iter().map(|(p, c)| (p.clone(), c)),
        4,
        &ExactDomain {
            limits: &limits,
            context: &context,
            elimination_mode: ExactEliminationMode::ConservativeDense,
        },
    )
    .unwrap();
    for (value, expected) in coefficients.iter().zip([
        Rational::one(),
        Rational::from((1, 2)),
        Rational::from((3, 8)),
        Rational::from((5, 16)),
        Rational::from((35, 128)),
    ]) {
        assert_eq!(*value, Gaussian::new(expected, Rational::zero()));
    }
}

#[test]
fn nonzero_fractional_power_is_not_discarded_as_a_ramification_slot() {
    let terms = series("z^(1/2)+(1-z)^(-1/2)").collect::<Vec<_>>();
    assert!(
        terms
            .iter()
            .any(|(power, coefficient)| !power.is_integer() && !coefficient.is_zero())
    );
    let limits = ExactFrobeniusLimits::default();
    let context = RunContext::default();
    assert!(matches!(
        regular_root_coefficients(
            terms.iter().map(|(p, c)| (p.clone(), c)),
            4,
            &ExactDomain {
                limits: &limits,
                context: &context,
                elimination_mode: ExactEliminationMode::ConservativeDense
            }
        ),
        Err(Error::Unsupported(_))
    ));
}
