#![cfg(feature = "automatic")]
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use symbolica::prelude::*;
use symbolica_amflow::reduction::Reduction;
use symbolica_amflow::*;

struct Unsupported {
    calls: AtomicUsize,
    identity: &'static str,
    budget: usize,
}
impl NativeIdentityProvider for Unsupported {
    fn identity(&self) -> &str {
        self.identity
    }
    fn max_calls(&self) -> usize {
        self.budget
    }
    fn provide(
        &self,
        _: &IntegralFamily,
        _: &Integral,
        _: &RunContext,
    ) -> Result<Option<NativeIdentityBank>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(None)
    }
}
fn family() -> IntegralFamily {
    IntegralFamily {
        name: "provider_fallback_tadpole".into(),
        loops: vec!["k".into()],
        external: vec![],
        external_gram: vec![],
        propagators: vec![Propagator {
            constant: Atom::num(-1),
            scalar_products: vec![Atom::one()],
        }],
        physical_propagators: 1,
        epsilon: symbol!("provider_fallback::epsilon"),
        dimension: 4,
    }
}
#[test]
fn unsupported_provider_is_bounded_and_falls_back_to_native_reduction() {
    let f = family();
    let targets = [Integral(vec![2])];
    let raw = RustRedBackend {
        parametric_rules: false,
        symmetry_rules: false,
        bubble_subloops: false,
        ..Default::default()
    };
    let expected = raw.reduce(&f, &targets, &RunContext::default()).unwrap();
    let provider = Arc::new(Unsupported {
        calls: AtomicUsize::new(0),
        identity: "unsupported-v1",
        budget: 1,
    });
    let provided = raw
        .clone()
        .with_identity_provider(provider.clone())
        .unwrap();
    let actual = provided
        .reduce(&f, &targets, &RunContext::default())
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        actual.expand_many(&targets).unwrap(),
        expected.expand_many(&targets).unwrap()
    );
    assert_eq!(actual.residuals, expected.residuals);
    assert_eq!(actual.nonzero_conditions, expected.nonzero_conditions);
    assert_ne!(raw.identity(), provided.identity());
}
#[test]
fn provider_identity_budget_and_sampling_scope_are_explicit() {
    let provider = |identity, budget| {
        Arc::new(Unsupported {
            calls: AtomicUsize::new(0),
            identity,
            budget,
        })
    };
    assert!(
        RustRedBackend::default()
            .with_identity_provider(provider("", 1))
            .is_err()
    );
    assert!(
        RustRedBackend::default()
            .with_identity_provider(provider("id", 0))
            .is_err()
    );
    let a = RustRedBackend::default()
        .with_identity_provider(provider("id", 1))
        .unwrap();
    let b = RustRedBackend::default()
        .with_identity_provider(provider("id", 2))
        .unwrap();
    let c = RustRedBackend::default()
        .with_identity_provider(provider("other", 1))
        .unwrap();
    assert_ne!(a.identity(), b.identity());
    assert_ne!(a.identity(), c.identity());
    assert!(matches!(
        a.reduce_at_epsilon(
            &family(),
            &[Integral(vec![2])],
            &Rational::from((1, 10)),
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
}

struct TadpoleIdentity {
    bank: NativeIdentityBank,
    calls: AtomicUsize,
    supplied: AtomicUsize,
}
impl NativeIdentityProvider for TadpoleIdentity {
    fn identity(&self) -> &str {
        "native-proved-tadpole-provider-v1"
    }
    fn max_calls(&self) -> usize {
        4
    }
    fn provide(
        &self,
        _: &IntegralFamily,
        target: &Integral,
        _: &RunContext,
    ) -> Result<Option<NativeIdentityBank>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if target == &Integral(vec![2]) {
            self.supplied.fetch_add(1, Ordering::Relaxed);
            Ok(Some(self.bank.clone()))
        } else {
            Ok(None)
        }
    }
}

#[test]
fn supplied_native_identity_searches_its_unknown_leaf_and_survives_restart() {
    let f = family();
    let target = Integral(vec![2]);
    let targets = [target.clone()];
    let raw = RustRedBackend {
        parametric_rules: false,
        symmetry_rules: false,
        bubble_subloops: false,
        ..Default::default()
    };
    // Obtain the supplied equation independently from the native backend.
    let expected = raw.reduce(&f, &targets, &RunContext::default()).unwrap();
    assert_eq!(expected.residuals, vec![Integral(vec![1])]);
    let equation = Reduction {
        rules: BTreeMap::from([(target.clone(), expected.expand(&target).unwrap())]),
        residuals: expected.residuals.clone(),
        nonzero_conditions: expected.nonzero_conditions.clone(),
    };
    let provider = Arc::new(TadpoleIdentity {
        bank: NativeIdentityBank::new(
            "native-tadpole-control",
            &f,
            equation,
            "1111111111111111111111111111111111111111111111111111111111111111",
            Default::default(),
        )
        .unwrap(),
        calls: AtomicUsize::new(0),
        supplied: AtomicUsize::new(0),
    });
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("rustflow-provider-{}-{stamp}", std::process::id()));
    let provided = RustRedBackend {
        max_targets: 1,
        checkpoints: Some(directory.clone()),
        checkpoint_interval: std::time::Duration::ZERO,
        ..raw
    }
    .with_identity_provider(provider.clone())
    .unwrap();
    let searches = Arc::new(AtomicUsize::new(0));
    let counted = searches.clone();
    let context = RunContext {
        progress: Some(Arc::new(move |event| {
            if let Progress::SectorReduction { integrals, .. } = event {
                counted.fetch_add(integrals, Ordering::Relaxed);
            }
        })),
        ..Default::default()
    };
    let actual = provided.reduce(&f, &targets, &context).unwrap();
    assert_eq!(provider.supplied.load(Ordering::Relaxed), 1);
    assert_eq!(
        searches.load(Ordering::Relaxed),
        1,
        "the unresolved I(1) must still be searched"
    );
    assert_eq!(
        actual.expand(&target).unwrap(),
        expected.expand(&target).unwrap()
    );
    assert_eq!(actual.residuals, expected.residuals);
    assert!(
        expected
            .nonzero_conditions
            .iter()
            .all(|c| actual.nonzero_conditions.contains(c))
    );
    let calls = provider.calls.load(Ordering::Relaxed);
    let restarted = provided.reduce(&f, &targets, &context).unwrap();
    assert_eq!(
        provider.calls.load(Ordering::Relaxed),
        calls,
        "a completed target checkpoint avoids provider replay"
    );
    assert_eq!(searches.load(Ordering::Relaxed), 1);
    assert_eq!(actual.rules, restarted.rules);
    assert_eq!(actual.residuals, restarted.residuals);
    assert_eq!(actual.nonzero_conditions, restarted.nonzero_conditions);
    std::fs::remove_dir_all(directory).unwrap();
}
