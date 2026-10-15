#![cfg(feature = "automatic")]
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use symbolica::prelude::*;
use symbolica_amflow::symbolica;
use symbolica_amflow::{reduction::Reduction, *};
const EVIDENCE: &str = "1111111111111111111111111111111111111111111111111111111111111111";
fn eps() -> Symbol {
    symbol!("native_seed_test::epsilon")
}
fn family(mass: Atom) -> IntegralFamily {
    IntegralFamily {
        name: "native_seed_test_tadpole".into(),
        loops: vec!["k".into()],
        external: vec![],
        external_gram: vec![],
        propagators: vec![Propagator {
            constant: -mass,
            scalar_products: vec![Atom::one()],
        }],
        physical_propagators: 1,
        epsilon: eps(),
        dimension: 4,
    }
}
fn rule(coefficient: Atom) -> Reduction {
    Reduction {
        rules: BTreeMap::from([(
            Integral(vec![2]),
            BTreeMap::from([(Integral(vec![1]), coefficient)]),
        )]),
        residuals: vec![Integral(vec![1])],
        nonzero_conditions: vec![],
    }
}
fn bank(f: &IntegralFamily, r: Reduction) -> NativeIdentityBank {
    NativeIdentityBank::new("proved_tadpole", f, r, EVIDENCE, Default::default()).unwrap()
}
fn coefficient(mass: Atom) -> Atom {
    (Atom::one() - Atom::var(eps())) / mass
}
fn counted() -> (RunContext, Arc<AtomicUsize>) {
    let n = Arc::new(AtomicUsize::new(0));
    let count = n.clone();
    (
        RunContext {
            progress: Some(Arc::new(move |event| {
                if let Progress::SectorReduction { integrals, .. } = event {
                    count.fetch_add(integrals, Ordering::Relaxed);
                }
            })),
            ..Default::default()
        },
        n,
    )
}
fn backend() -> RustRedBackend {
    RustRedBackend {
        parametric_rules: false,
        symmetry_rules: false,
        bubble_subloops: false,
        ..Default::default()
    }
}
fn directory(label: &str) -> std::path::PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "rustflow-seed-{}-{stamp}-{label}",
        std::process::id()
    ));
    std::fs::create_dir(&p).expect("preserve prior test output");
    p
}
#[test]
fn exact_seed_matches_fresh_native_result_and_avoids_a_search() {
    let f = family(Atom::num(2));
    let target = [Integral(vec![2])];
    let point = Rational::from((1, 10));
    let fresh = backend()
        .reduce_at_epsilon(&f, &target, &point, &RunContext::default())
        .unwrap();
    let mut native = backend();
    native.max_targets = 1;
    let seeded = native.with_identity_bank(bank(&f, rule(coefficient(Atom::num(2)))));
    let (ctx, count) = counted();
    let result = seeded.reduce_at_epsilon(&f, &target, &point, &ctx).unwrap();
    assert_eq!(
        result.expand(&target[0]).unwrap(),
        fresh.expand(&target[0]).unwrap()
    );
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(result.residuals, vec![Integral(vec![1])]);
}
#[test]
fn supplied_residuals_remain_unsearched_and_limited() {
    let f = family(Atom::one());
    let mut native = backend();
    native.max_targets = 0;
    let seeded = native.with_identity_bank(bank(&f, rule(coefficient(Atom::one()))));
    assert!(matches!(
        seeded.reduce(&f, &[Integral(vec![2])], &RunContext::default()),
        Err(Error::Limit(_))
    ));
}
#[test]
fn supplied_zero_is_distinct_from_an_unknown_leaf() {
    let f = family(Atom::Zero);
    let r = Reduction {
        rules: BTreeMap::from([(Integral(vec![2]), BTreeMap::new())]),
        residuals: vec![],
        nonzero_conditions: vec![],
    };
    let mut native = backend();
    native.max_targets = 0;
    let out = native
        .with_identity_bank(bank(&f, r))
        .reduce(&f, &[Integral(vec![2])], &RunContext::default())
        .unwrap();
    assert!(out.expand(&Integral(vec![2])).unwrap().is_empty());
    assert!(out.residuals.is_empty());
}
#[test]
fn wrong_family_dimension_and_parameter_specialization_reject() {
    let mass = Atom::one() + Atom::var(eps());
    let f = family(mass.clone());
    let seeded = backend().with_identity_bank(bank(&f, rule(coefficient(mass))));
    for wrong in [
        family(Atom::num(2)),
        {
            let mut x = f.clone();
            x.dimension = 2;
            x
        },
        f.at(&KinematicPoint(BTreeMap::from([(
            Atom::var(eps()),
            Atom::num(Rational::from((1, 10))),
        )]))),
    ] {
        assert!(matches!(
            seeded.reduce(&wrong, &[Integral(vec![2])], &RunContext::default()),
            Err(Error::IncompleteReduction(_))
        ));
    }
    let out = seeded
        .reduce_at_epsilon(
            &f,
            &[Integral(vec![2])],
            &Rational::from((1, 10)),
            &RunContext::default(),
        )
        .unwrap();
    assert!(
        (&out.expand(&Integral(vec![2])).unwrap()[&Integral(vec![1])]
            - Atom::num(Rational::from((9, 11))))
        .together()
        .cancel()
        .is_zero()
    );
}
#[test]
fn sampled_zero_condition_rejects_and_symbolic_condition_survives() {
    let f = family(Atom::one());
    let mut r = rule(coefficient(Atom::one()));
    r.nonzero_conditions.push(Atom::var(eps()));
    let seeded = backend().with_identity_bank(bank(&f, r));
    assert!(matches!(
        seeded.reduce_at_epsilon(
            &f,
            &[Integral(vec![2])],
            &Rational::from(0),
            &RunContext::default()
        ),
        Err(Error::Reduction(_))
    ));
    let out = seeded
        .reduce(&f, &[Integral(vec![2])], &RunContext::default())
        .unwrap();
    assert!(
        out.nonzero_conditions
            .iter()
            .any(|a| a.derivative(eps()) != Atom::Zero)
    );
}
#[test]
fn native_order_and_cyclic_rules_reject() {
    let f = family(Atom::one());
    let upward = Reduction {
        rules: BTreeMap::from([(
            Integral(vec![1]),
            BTreeMap::from([(Integral(vec![2]), Atom::one())]),
        )]),
        residuals: vec![Integral(vec![2])],
        nonzero_conditions: vec![],
    };
    assert!(matches!(
        backend()
            .with_identity_bank(bank(&f, upward.clone()))
            .reduce(&f, &[Integral(vec![1])], &RunContext::default()),
        Err(Error::InvalidInput(_))
    ));
    let mut cyclic = upward;
    cyclic.rules.insert(
        Integral(vec![2]),
        BTreeMap::from([(Integral(vec![1]), Atom::one())]),
    );
    assert!(NativeIdentityBank::new("cycle", &f, cyclic, EVIDENCE, Default::default()).is_err());
}
#[test]
fn bank_budget_scope_domain_and_evidence_are_bound() {
    let f = family(Atom::one());
    let r = rule(coefficient(Atom::one()));
    let first = bank(&f, r.clone());
    let mut guarded = r.clone();
    guarded.nonzero_conditions.push(Atom::var(eps()));
    let second = bank(&f, guarded);
    assert_ne!(first.identity(), second.identity());
    let evidence = EVIDENCE.replace('1', "2");
    let other = NativeIdentityBank::new(
        "proved_tadpole",
        &f,
        r.clone(),
        &evidence,
        Default::default(),
    )
    .unwrap();
    assert_ne!(first.identity(), other.identity());
    assert!(matches!(
        NativeIdentityBank::new(
            "budget",
            &f,
            r.clone(),
            EVIDENCE,
            NativeIdentityLimits {
                max_rules: 0,
                ..Default::default()
            }
        ),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        NativeIdentityBank::new(
            "budget",
            &f,
            r.clone(),
            EVIDENCE,
            NativeIdentityLimits {
                max_terms: 0,
                ..Default::default()
            }
        ),
        Err(Error::Limit(_))
    ));
    assert!(NativeIdentityBank::new("bad_receipt", &f, r, "invalid", Default::default()).is_err());
    assert_eq!(first.rule_count(), 1);
    let raw = backend();
    let before = raw.identity();
    let seeded = raw.clone().with_identity_bank(first);
    assert_eq!(before, raw.identity());
    assert_ne!(before, seeded.identity());
}
#[test]
fn unsupported_nonfactorized_mode_rejects() {
    let f = family(Atom::one());
    let mut raw = backend();
    raw.factorized = false;
    let seeded = raw.with_identity_bank(bank(&f, rule(coefficient(Atom::one()))));
    assert!(matches!(
        seeded.reduce(&f, &[Integral(vec![2])], &RunContext::default()),
        Err(Error::Unsupported(_))
    ));
}
#[test]
fn cancellation_does_not_admit_or_search() {
    let f = family(Atom::one());
    let seeded = backend().with_identity_bank(bank(&f, rule(coefficient(Atom::one()))));
    let ctx = RunContext::default();
    ctx.cancellation.cancel();
    assert!(seeded.reduce(&f, &[Integral(vec![2])], &ctx).is_err());
}
#[test]
fn seeded_checkpoints_restart_and_never_reuse_no_bank_scope() {
    let f = family(Atom::num(2));
    let targets = [Integral(vec![2])];
    let point = Rational::from((1, 10));
    let directory = directory("checkpoint_scopes");
    let mut raw = backend();
    raw.checkpoints = Some(directory.clone());
    let expected = raw
        .reduce_at_epsilon(&f, &targets, &point, &RunContext::default())
        .unwrap();
    let old = std::fs::read_dir(&directory)
        .unwrap()
        .map(|p| p.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    let mut one = raw.clone();
    one.max_targets = 1;
    let seeded = one.with_identity_bank(bank(&f, rule(coefficient(Atom::num(2)))));
    let (ctx, count) = counted();
    let actual = seeded
        .reduce_at_epsilon(&f, &targets, &point, &ctx)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(
        actual.expand(&targets[0]).unwrap(),
        expected.expand(&targets[0]).unwrap()
    );
    let after = std::fs::read_dir(&directory)
        .unwrap()
        .map(|p| p.unwrap().file_name())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(old.is_subset(&after));
    assert!(
        after
            .difference(&old)
            .all(|s| s.to_str().unwrap().contains("identity-seed-v1"))
    );
    assert!(after.len() > old.len());
    let again = seeded
        .reduce_at_epsilon(&f, &targets, &point, &ctx)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(
        again.expand(&targets[0]).unwrap(),
        actual.expand(&targets[0]).unwrap()
    );
    let mut changed = rule(coefficient(Atom::num(2)));
    changed.nonzero_conditions.push(Atom::var(eps()));
    let modified = raw.with_identity_bank(bank(&f, changed));
    let (ctx, other_count) = counted();
    modified
        .reduce_at_epsilon(&f, &targets, &point, &ctx)
        .unwrap();
    assert_eq!(other_count.load(Ordering::Relaxed), 1);
}

#[test]
fn cut_measure_is_explicitly_unsupported() {
    use symbolica_amflow::cuts::*;
    let f = family(Atom::one());
    let seeded = backend().with_identity_bank(bank(&f, rule(coefficient(Atom::one()))));
    let cuts = CutMetadata::new(
        [CutLine {
            propagator: 0,
            definition: CutDefinition::Distribution,
        }],
        vec![LoopPrescription::Insensitive],
    )
    .unwrap();
    let cut = CutFamily::new(f, cuts).unwrap();
    assert!(matches!(
        seeded.reduce_cut(&cut, &[Integral(vec![2])], &RunContext::default()),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        seeded.reduce_cut_at_epsilon(
            &cut,
            &[Integral(vec![2])],
            &Rational::from((1, 10)),
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn interrupted_seeded_checkpoint_restarts_with_unknown_leaf_still_unsearched() {
    let f = family(Atom::num(2));
    let targets = [Integral(vec![2])];
    let point = Rational::from((1, 10));
    let directory = directory("interrupted_checkpoint");
    let mut raw = backend();
    raw.checkpoints = Some(directory.clone());
    raw.max_targets = 1;
    let seeded = raw.with_identity_bank(bank(&f, rule(coefficient(Atom::num(2)))));
    let token = CancellationToken::default();
    let child_token = token.clone();
    let ctx = RunContext {
        cancellation: token,
        progress: Some(Arc::new(move |event| {
            if matches!(event, Progress::SectorReduction { .. }) {
                child_token.cancel();
            }
        })),
    };
    assert!(
        seeded
            .reduce_at_epsilon(&f, &targets, &point, &ctx)
            .is_err()
    );
    assert!(std::fs::read_dir(&directory).unwrap().next().is_some());
    let (fresh_ctx, count) = counted();
    let resumed = seeded
        .reduce_at_epsilon(&f, &targets, &point, &fresh_ctx)
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 1);
    let fresh = backend()
        .reduce_at_epsilon(&f, &targets, &point, &RunContext::default())
        .unwrap();
    assert_eq!(
        resumed.expand(&targets[0]).unwrap(),
        fresh.expand(&targets[0]).unwrap()
    );
}

#[test]
fn inferred_denominator_conditions_obey_the_final_budget() {
    let eta = Atom::var(symbol!("native_seed_test::eta"));
    let f = family(eta.clone());
    // A constructor-admission negative control, never used for evaluation.
    // The input condition list is empty; 1/eta introduces an inferred domain.
    let r = rule(Atom::one() / eta);
    assert!(r.nonzero_conditions.is_empty());
    let result = NativeIdentityBank::new(
        "inferred_budget",
        &f,
        r,
        EVIDENCE,
        NativeIdentityLimits {
            max_conditions: 0,
            ..Default::default()
        },
    );
    assert!(
        matches!(result, Err(Error::Limit(ref message)) if message.contains("admitted conditions"))
    );
}
