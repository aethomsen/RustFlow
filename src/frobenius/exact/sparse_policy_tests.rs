use super::*;
use crate::asymptotic::{AsymptoticSelector, ExactAsymptoticConstraints, ExactAsymptoticRelation};
fn g(n: i64) -> Gaussian {
    Gaussian::new(Rational::from(n), Rational::zero())
}
fn domain<'a>(limits: &'a ExactFrobeniusLimits, context: &'a RunContext) -> ExactDomain<'a> {
    ExactDomain {
        limits,
        context,
        elimination_mode: ExactEliminationMode::ObservedSparse,
    }
}
fn atom(s: &str) -> Atom {
    Atom::parse(s, "sparse_policy_tests", Default::default()).unwrap()
}
#[test]
fn sparse_policy_reuses_native_singular_solve_and_rejects_inconsistent_rhs() {
    let limits = ExactFrobeniusLimits::default();
    let context = RunContext::default();
    let d = domain(&limits, &context);
    for (a, b, good) in [
        (vec![vec![g(0)]], vec![g(1)], false),
        (
            vec![vec![g(1), g(0)], vec![g(1), g(0)]],
            vec![g(1), g(2)],
            false,
        ),
        (
            vec![vec![g(1), g(0)], vec![g(1), g(0)]],
            vec![g(1), g(1)],
            true,
        ),
        (vec![vec![g(0)]], vec![g(0)], true),
    ] {
        assert_eq!(d.sparse_solve_any(a, b, &context).is_ok(), good);
    }
    let rows = d
        .sparse_reduced_rows(vec![vec![g(1), g(0), g(1)], vec![g(1), g(0), g(2)]])
        .unwrap();
    assert!(
        rows.iter()
            .any(|r| r[..2].iter().all(Gaussian::is_zero) && !r[2].is_zero())
    );
}
#[test]
fn sparse_policy_preserves_default_and_matches_small_exact_bases() {
    let limits = ExactFrobeniusLimits::default();
    let context = RunContext::default();
    for rows in [
        vec![vec!["1/z", "0"], vec!["0", "2/z"]],
        vec![vec!["0", "1/z"], vec!["0", "0"]],
        vec![vec!["0", "0"], vec!["1", "1/z"]],
    ] {
        let prepared = DifferentialSystem {
            variable: symbol!("sparse_policy_tests::z"),
            matrix: rows
                .iter()
                .map(|r| r.iter().map(|s| atom(s)).collect())
                .collect(),
        }
        .prepare_frobenius(&context)
        .unwrap();
        let old = prepared.exact_coefficients(6, &limits, &context).unwrap();
        let new = prepared
            .exact_coefficients_with_elimination(
                6,
                &limits,
                ExactEliminationMode::ObservedSparse,
                &context,
            )
            .unwrap();
        assert_eq!(
            old.elimination_mode(),
            ExactEliminationMode::ConservativeDense
        );
        assert_eq!(new.elimination_mode(), ExactEliminationMode::ObservedSparse);
        assert_eq!(old.known_order, new.known_order);
        for (a, b) in old.columns.iter().zip(&new.columns) {
            assert_eq!(a.exponent, b.exponent);
            assert_eq!(a.coefficients, b.coefficients);
        }
        let constraints = ExactAsymptoticConstraints {
            relations: vec![ExactAsymptoticRelation::coefficient(
                AsymptoticSelector {
                    component: 0,
                    power: Atom::num(0),
                    log_power: 0,
                },
                Atom::num(0),
            )],
            provenance: "exact synthetic zero".into(),
        };
        let a = old.constrain(&constraints, &limits, &context).unwrap();
        let b = new.constrain(&constraints, &limits, &context).unwrap();
        assert_eq!(a.offset(), b.offset());
        assert_eq!(a.directions(), b.directions());
    }
}
#[test]
fn sparse_policy_observes_input_growth_cells_cancellation_and_blocks() {
    let context = RunContext::default();
    let limits = ExactFrobeniusLimits {
        max_coefficient_bits: 10,
        ..Default::default()
    };
    let d = domain(&limits, &context);
    assert!(matches!(
        d.sparse_reduced_rows(vec![vec![g(2048), g(1)]]),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        d.sparse_inverse(vec![vec![g(1), g(511)], vec![g(511), g(1)]], &context),
        Err(Error::Limit(_))
    ));
    let small = ExactFrobeniusLimits {
        max_scalar_cells: 3,
        ..Default::default()
    };
    assert!(matches!(
        domain(&small, &context).sparse_reduced_rows(vec![vec![g(1), g(1)]]),
        Err(Error::Limit(_))
    ));
    assert!(
        d.validate_sparse_blocks(&[vec![g(1), g(0)], vec![g(1), g(1)]], &[vec![0], vec![1]])
            .is_ok()
    );
    for blocks in [
        vec![vec![1], vec![0]],
        vec![vec![0], vec![0]],
        vec![vec![2], vec![0]],
        vec![vec![0]],
    ] {
        assert!(
            d.validate_sparse_blocks(&[vec![g(1), g(0)], vec![g(1), g(1)]], &blocks)
                .is_err()
        );
    }
    context.cancellation.cancel();
    assert!(d.sparse_reduced_rows(vec![vec![g(1), g(1)]]).is_err());
}
#[test]
fn sparse_policy_dense_height_cap_is_still_default_only() {
    let context = RunContext::default();
    let limits = ExactFrobeniusLimits {
        max_coefficient_bits: 128,
        ..Default::default()
    };
    let rows = (0..8)
        .map(|i| (0..8).map(|j| g(if i == j { 1 } else { 0 })).collect())
        .collect::<Vec<Vec<_>>>();
    let blocks = (0..8).map(|i| vec![i]).collect::<Vec<_>>();
    assert!(matches!(
        ExactDomain::dense(&limits, &context).solver(rows.clone(), &blocks, &context),
        Err(Error::Limit(_))
    ));
    assert!(
        domain(&limits, &context)
            .solver(rows, &blocks, &context)
            .is_ok()
    );
}
