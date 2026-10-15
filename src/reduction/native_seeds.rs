//! Exact supplied identities admitted before native frontier search.
//!
//! Algebraic validity remains the caller's contract, as for ScopedTableBackend.
//! The evidence digest binds that external proof receipt; it is not itself a
//! proof. Scope, coefficient domains, graph validity, native order and budgets
//! are checked here and by the existing native search owner.
use super::{Reduction, ReductionBackend, RustRedBackend, ScopedTableBackend};
use crate::{Error, Integral, IntegralFamily, Result, RunContext};
use std::{collections::BTreeSet, sync::Arc};
use symbolica::prelude::*;

/// Supplied equations have a separate bounded budget from solver searches.
#[derive(Clone, Copy, Debug)]
pub struct NativeIdentityLimits {
    pub max_rules: usize,
    pub max_terms: usize,
    pub max_residuals: usize,
    pub max_conditions: usize,
}
impl Default for NativeIdentityLimits {
    fn default() -> Self {
        Self {
            max_rules: 65_536,
            max_terms: 262_144,
            max_residuals: 65_536,
            max_conditions: 131_072,
        }
    }
}

/// Immutable exact-family table with an authenticated content/domain identity.
#[derive(Clone, Debug)]
pub struct NativeIdentityBank {
    table: ScopedTableBackend,
    targets: Vec<Integral>,
    identity: String,
    rules: usize,
    limits: NativeIdentityLimits,
}
impl NativeIdentityBank {
    /// The SHA256 names the caller's independent proof receipt. The complete
    /// admitted mathematical table and its original domain are additionally
    /// hashed by ScopedTableBackend; changing either changes the bank identity.
    pub fn new(
        name: impl Into<String>,
        family: &IntegralFamily,
        reduction: Reduction,
        evidence_sha256: &str,
        limits: NativeIdentityLimits,
    ) -> Result<Self> {
        if evidence_sha256.len() != 64 || !evidence_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::InvalidInput(
                "identity-bank evidence must be a SHA256 digest".into(),
            ));
        }
        let terms = reduction
            .rules
            .values()
            .try_fold(0usize, |n, r| n.checked_add(r.len()))
            .ok_or_else(|| Error::Limit("identity-bank term count overflow".into()))?;
        for (name, actual, limit) in [
            ("rules", reduction.rules.len(), limits.max_rules),
            ("terms", terms, limits.max_terms),
            ("residuals", reduction.residuals.len(), limits.max_residuals),
            (
                "conditions",
                reduction.nonzero_conditions.len(),
                limits.max_conditions,
            ),
        ] {
            if actual > limit {
                return Err(Error::Limit(format!(
                    "identity-bank {name} requires {actual}; limit {limit}"
                )));
            }
        }
        let rules = reduction.rules.len();
        let targets = reduction
            .rules
            .keys()
            .chain(&reduction.residuals)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let table = ScopedTableBackend::from_table(name, family, reduction)?;
        // Scope admission infers restrictions from every rational denominator.
        // Those conditions also consume the advertised bank-domain budget.
        let admitted_conditions = table.table(family)?.reduction.nonzero_conditions.len();
        if admitted_conditions > limits.max_conditions {
            return Err(Error::Limit(format!(
                "identity-bank admitted conditions requires {admitted_conditions}; limit {}",
                limits.max_conditions
            )));
        }

        let identity = format!(
            "native-identities-v1:{}:{}:{limits:?}",
            table.identity(),
            evidence_sha256.to_ascii_lowercase()
        );
        Ok(Self {
            table,
            targets,
            identity,
            rules,
            limits,
        })
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn rule_count(&self) -> usize {
        self.rules
    }
    pub fn limits(&self) -> NativeIdentityLimits {
        self.limits
    }

    pub(crate) fn resolve(
        &self,
        family: &IntegralFamily,
        epsilon: Option<&Rational>,
        context: &RunContext,
    ) -> Result<ResolvedIdentityBank> {
        let reduction = if let Some(epsilon) = epsilon {
            self.table
                .reduce_at_epsilon(family, &self.targets, epsilon, context)?
        } else {
            self.table.reduce(family, &self.targets, context)?
        };
        // Explicitly distinguish symbolic and sampled domains even if a
        // specialized family's remaining affine coefficients happen to agree.
        let identity = format!("{}:epsilon={epsilon:?}", self.identity);
        Ok(ResolvedIdentityBank {
            identity,
            reduction,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedIdentityBank {
    pub identity: String,
    pub reduction: Reduction,
}

/// A native backend with one strict, exact-family identity scope.
/// Other families require their own wrapper; cuts and the legacy nonfactorized
/// solver are explicitly unsupported. The original RustRedBackend is unchanged.
#[derive(Clone, Debug)]
pub struct SeededRustRedBackend {
    pub(crate) backend: RustRedBackend,
    pub(crate) bank: Arc<NativeIdentityBank>,
}
impl RustRedBackend {
    pub fn with_identity_bank(self, bank: NativeIdentityBank) -> SeededRustRedBackend {
        SeededRustRedBackend {
            backend: self,
            bank: Arc::new(bank),
        }
    }
}
impl ReductionBackend for SeededRustRedBackend {
    fn identity(&self) -> String {
        format!(
            "native-seeded-backend-v1:{}:{}",
            self.backend.identity(),
            self.bank.identity()
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
            Some(&self.bank),
            None,
        )
    }
    fn reduce_at_epsilon(
        &self,
        family: &IntegralFamily,
        targets: &[Integral],
        epsilon: &Rational,
        context: &RunContext,
    ) -> Result<Reduction> {
        self.backend.reduce_native_with_seed(
            family,
            targets,
            Some(epsilon),
            context,
            None,
            Some(&self.bank),
            None,
        )
    }
    // ReductionBackend's explicit Unsupported defaults retain the cut measure.
}
