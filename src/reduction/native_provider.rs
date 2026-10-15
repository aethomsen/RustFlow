//! Optional, exact-family identities for newly discovered native frontier keys.
//!
//! Mathematical certification belongs to the provider, like NativeIdentityBank.
//! The native owner reuses its scope/domain/graph/order admission and checkpoint
//! codec. A provider must have immutable behavior under its identity: definitions,
//! proof engine, limits, serializer and evidence policy all belong in that identity.
//! None means unsupported, never zero. Errors and cancellation are not swallowed.
use super::{
    NativeIdentityBank, Reduction, ReductionBackend, RustRedBackend, SeededRustRedBackend,
};
use crate::{Error, Integral, IntegralFamily, Result, RunContext};
use std::sync::Arc;
use symbolica::prelude::*;

pub trait NativeIdentityProvider: Send + Sync {
    fn identity(&self) -> &str;
    /// A per-call budget, separate from visited/search accounting. Exhaustion falls
    /// back to ordinary reduction; persistent producer caches have their own bounds.
    fn max_calls(&self) -> usize;
    fn provide(
        &self,
        family: &IntegralFamily,
        target: &Integral,
        context: &RunContext,
    ) -> Result<Option<NativeIdentityBank>>;
}

#[derive(Clone)]
pub struct ProvidedRustRedBackend {
    backend: RustRedBackend,
    seed: Option<Arc<NativeIdentityBank>>,
    provider: Arc<dyn NativeIdentityProvider>,
}
impl RustRedBackend {
    pub fn with_identity_provider(
        self,
        provider: Arc<dyn NativeIdentityProvider>,
    ) -> Result<ProvidedRustRedBackend> {
        ProvidedRustRedBackend::new(self, None, provider)
    }
}
impl SeededRustRedBackend {
    pub fn with_identity_provider(
        self,
        provider: Arc<dyn NativeIdentityProvider>,
    ) -> Result<ProvidedRustRedBackend> {
        ProvidedRustRedBackend::new(self.backend, Some(self.bank), provider)
    }
}
impl ProvidedRustRedBackend {
    fn new(
        backend: RustRedBackend,
        seed: Option<Arc<NativeIdentityBank>>,
        provider: Arc<dyn NativeIdentityProvider>,
    ) -> Result<Self> {
        if provider.identity().is_empty() || provider.max_calls() == 0 {
            return Err(Error::InvalidInput(
                "provider identity and positive call budget required".into(),
            ));
        }
        Ok(Self {
            backend,
            seed,
            provider,
        })
    }
}
impl ReductionBackend for ProvidedRustRedBackend {
    fn identity(&self) -> String {
        format!(
            "native-provider-v1:{}:{:?}:{}:{}",
            self.backend.identity(),
            self.seed.as_ref().map(|s| s.identity()),
            self.provider.identity(),
            self.provider.max_calls()
        )
    }
    fn reduce(
        &self,
        family: &IntegralFamily,
        targets: &[Integral],
        context: &RunContext,
    ) -> Result<Reduction> {
        self.backend.reduce_native_with_seed(
            family,
            targets,
            None,
            context,
            None,
            self.seed.as_deref(),
            Some(self.provider.as_ref()),
        )
    }
    fn reduce_at_epsilon(
        &self,
        _: &IntegralFamily,
        _: &[Integral],
        _: &Rational,
        _: &RunContext,
    ) -> Result<Reduction> {
        Err(Error::Unsupported(
            "on-demand identities currently require symbolic epsilon".into(),
        ))
    }
}
