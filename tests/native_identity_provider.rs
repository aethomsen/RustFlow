#![cfg(feature = "automatic")]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use symbolica::prelude::*;
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
