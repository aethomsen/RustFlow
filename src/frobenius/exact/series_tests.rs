use super::*;

#[test]
fn explicit_32m_ceiling_keeps_default_and_other_admission_bounds() {
    assert_eq!(ExactFrobeniusLimits::default().max_scalar_cells, 4_000_000);
    for (cells, accepted) in [
        (0, false),
        (16_000_000, true),
        (16_000_001, true),
        (32_000_000, true),
        (32_000_001, false),
    ] {
        let limits = ExactFrobeniusLimits {
            max_dimension: 128,
            max_coefficient_bits: 1_048_576,
            max_scalar_cells: cells,
            ..Default::default()
        };
        assert_eq!(limits.validate().is_ok(), accepted);
    }
    let context = RunContext::default();
    let old = ExactFrobeniusLimits {
        max_dimension: 128,
        max_coefficient_bits: 1_048_576,
        max_scalar_cells: 16_000_000,
        ..Default::default()
    };
    let larger = ExactFrobeniusLimits {
        max_scalar_cells: 32_000_000,
        ..old.clone()
    };
    // PR15: requested order7 plus the independently proved gauge span5.
    assert_eq!(127usize * 127 * 128 * (7 + 5 + 1), 26_838_656);
    let old_domain = ExactDomain {
        limits: &old,
        context: &context,
        elimination_mode: ExactEliminationMode::ObservedSparse,
    };
    let domain = ExactDomain {
        limits: &larger,
        context: &context,
        elimination_mode: ExactEliminationMode::ObservedSparse,
    };
    assert!(matches!(
        old_domain.preflight(127, 12),
        Err(Error::Limit(_))
    ));
    assert!(domain.preflight(127, 12).is_ok());
    assert!(matches!(domain.preflight(127, 15), Err(Error::Limit(_))));
    assert!(matches!(domain.preflight(129, 1), Err(Error::Limit(_))));
    let excess_bits = ExactFrobeniusLimits {
        max_coefficient_bits: 1_048_577,
        ..larger
    };
    assert!(excess_bits.validate().is_err());
}

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
