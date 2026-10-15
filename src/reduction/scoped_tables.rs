//! Family-bound supplied reductions, using the existing exact table applier.
//!
//! Each entry is valid for one exact ordered family. Auxiliary-mass deformations,
//! kinematic specializations and recursive boundary families require their own
//! entries. Mathematical validity of the supplied identities remains the caller's
//! contract; this adapter validates scope, domains and the reduction graph.
use super::{Reduction, ReductionBackend, TableBackend};
use crate::{Error, Integral, IntegralFamily, Result, RunContext};
use std::collections::{BTreeMap, BTreeSet};
use symbolica::prelude::*;

/// A collection of exact reduction tables bound to their original families.
///
/// Unlike [`TableBackend`], this backend never applies the same integral-index
/// table to another family. Denominator order, momentum labels/routing, scalar
/// symbol namespaces, dimension convention and numerator-slot roles are part of
/// the scope. Equivalent routings need explicit entries; no unproved symmetry or
/// parameter renaming is inferred.
/// Algebraically identical rational coefficient forms share a scope, while
/// original denominator restrictions remain part of its identity and evidence.
///
/// Admission does not call an IBP solver, require a complete scalar-product
/// denominator basis, or depend on RustRed's compiled runtime arities. The
/// selected evaluation workflow can impose its own family requirements later.
#[derive(Clone, Debug)]
pub struct ScopedTableBackend {
    name: String,
    tables: BTreeMap<String, TableBackend>,
}

impl ScopedTableBackend {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tables: BTreeMap::new(),
        }
    }

    pub fn from_table(
        name: impl Into<String>,
        family: &IntegralFamily,
        reduction: Reduction,
    ) -> Result<Self> {
        let mut result = Self::new(name);
        result.insert(family, reduction)?;
        Ok(result)
    }

    /// Admit a table without replacing any previously admitted scope.
    ///
    /// Rules must use the family's original scalar symbols and epsilon, with
    /// `D = family.dimension - 2 epsilon` already substituted. Original rational
    /// denominator restrictions are retained before coefficient cancellation.
    /// All rule leaves must be declared residuals, and reachable cycles fail.
    pub fn insert(&mut self, family: &IntegralFamily, mut reduction: Reduction) -> Result<()> {
        let expressions = family_expressions(family)?;
        let key = scope_key(family)?;
        if self.tables.contains_key(&key) {
            return Err(Error::InvalidInput(format!(
                "a supplied reduction table already exists for exact family {} ({key})",
                family.name
            )));
        }
        let variables = family_variables(family, &expressions)?;
        let mut coefficients = expressions;
        for (target, terms) in &reduction.rules {
            family.validate_integral(target)?;
            for (integral, coefficient) in terms {
                family.validate_integral(integral)?;
                coefficients.push(coefficient.clone());
            }
        }
        for integral in &reduction.residuals {
            family.validate_integral(integral)?;
        }
        let mut conditions = reduction.nonzero_conditions.clone();
        conditions.extend(crate::physical_conditions::rational_denominator_conditions(
            &coefficients,
            &variables,
        )?);
        reduction.nonzero_conditions =
            crate::physical_conditions::canonical_conditions(&conditions, &variables)?;
        let targets = reduction
            .rules
            .keys()
            .chain(&reduction.residuals)
            .cloned()
            .collect::<Vec<_>>();
        // The same native sparse graph planner used by every public Reduction
        // call validates the complete immutable supplied table once at admission.
        super::table_graph::validate(&reduction, &targets)?;
        self.tables.insert(
            key,
            TableBackend {
                name: self.name.clone(),
                reduction,
            },
        );
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub(super) fn table(&self, family: &IntegralFamily) -> Result<&TableBackend> {
        let key = scope_key(family)?;
        self.tables.get(&key).ok_or_else(|| Error::IncompleteReduction(format!(
            "no supplied reduction table for exact family {} ({key}); auxiliary deformations, specializations and recursive boundary families require separate entries",
            family.name
        )))
    }
}

impl ReductionBackend for ScopedTableBackend {
    fn identity(&self) -> String {
        let entries = self
            .tables
            .iter()
            .map(|(key, table)| (key, table.identity()))
            .collect::<Vec<_>>();
        let content = format!("{:?}:{entries:?}", self.name);
        format!(
            "scoped-table-v1:{}",
            blake3::hash(content.as_bytes()).to_hex()
        )
    }

    fn reduce(
        &self,
        family: &IntegralFamily,
        targets: &[Integral],
        context: &RunContext,
    ) -> Result<Reduction> {
        context.cancellation.check()?;
        for target in targets {
            family.validate_integral(target)?;
        }
        self.table(family)?.reduce(family, targets, context)
    }

    fn reduce_at_epsilon(
        &self,
        family: &IntegralFamily,
        targets: &[Integral],
        epsilon: &Rational,
        context: &RunContext,
    ) -> Result<Reduction> {
        context.cancellation.check()?;
        for target in targets {
            family.validate_integral(target)?;
        }
        let table = self.table(family)?;
        let replacement = BTreeMap::from([(Atom::var(family.epsilon), Atom::num(epsilon.clone()))]);
        for condition in &table.reduction.nonzero_conditions {
            if crate::family::substitute(condition, &replacement)
                .together()
                .cancel()
                .is_zero()
            {
                return Err(Error::Reduction(format!(
                    "a supplied reduction nonzero condition vanishes at epsilon={epsilon}"
                )));
            }
        }
        table.reduce_at_epsilon(family, targets, epsilon, context)
    }
}

fn scope_key(family: &IntegralFamily) -> Result<String> {
    let expressions = family_expressions(family)?;
    let variables = family_variables(family, &expressions)?;
    // Retain the original domain before normalization can cancel denominators.
    // Equal rational coefficients with different original poles are distinct
    // scopes, even if their normalized expressions are identical.
    let conditions =
        crate::physical_conditions::rational_denominator_conditions(&expressions, &variables)?;
    let mut normalized = family.clone();
    for propagator in &mut normalized.propagators {
        propagator.constant = propagator.constant.together().cancel();
        for coefficient in &mut propagator.scalar_products {
            *coefficient = coefficient.together().cancel();
        }
    }
    for coefficient in normalized.external_gram.iter_mut().flatten() {
        *coefficient = coefficient.together().cancel();
    }
    // Reuse the existing exact original-family identity owner. Its metadata
    // includes every ordered coefficient and original symbol namespace, and
    // does not invoke native family admission or a runtime-arity bridge.
    crate::physical_family::physical_family_fingerprint(&normalized, &BTreeSet::new(), &conditions)
}

fn family_variables(family: &IntegralFamily, expressions: &[Atom]) -> Result<BTreeSet<Symbol>> {
    let mut symbols = BTreeSet::new();
    for atom in expressions {
        crate::family::scalar_symbols(atom.as_view(), &mut symbols)?;
    }
    symbols.insert(Atom::var(family.epsilon));
    symbols.remove(&Atom::var(crate::family::imaginary_parameter()));
    Ok(symbols
        .into_iter()
        .map(|atom| match atom.as_view() {
            AtomView::Var(v) => v.get_symbol(),
            _ => unreachable!("scalar_symbols returns only variables"),
        })
        .collect())
}

fn family_expressions(family: &IntegralFamily) -> Result<Vec<Atom>> {
    let loops = family.loops.len();
    let external = family.external.len();
    let products = loops
        .checked_add(1)
        .and_then(|next| loops.checked_mul(next))
        .map(|count| count / 2)
        .and_then(|count| {
            loops
                .checked_mul(external)
                .and_then(|mixed| count.checked_add(mixed))
        })
        .ok_or_else(|| Error::Limit("supplied family scalar-product count overflow".into()))?;
    if loops == 0
        || family.propagators.is_empty()
        || family.physical_propagators > family.propagators.len()
        || family
            .propagators
            .iter()
            .any(|p| p.scalar_products.len() != products)
        || family.external_gram.len() != external
        || family.external_gram.iter().any(|row| row.len() != external)
    {
        return Err(Error::InvalidInput(
            "inconsistent supplied family affine metadata".into(),
        ));
    }
    for i in 0..external {
        for j in 0..i {
            if (&family.external_gram[i][j] - &family.external_gram[j][i])
                .together()
                .cancel()
                != Atom::Zero
            {
                return Err(Error::InvalidInput(
                    "supplied external Gram matrix is not symmetric".into(),
                ));
            }
        }
    }
    Ok(family
        .propagators
        .iter()
        .flat_map(|p| std::iter::once(p.constant.clone()).chain(p.scalar_products.iter().cloned()))
        .chain(family.external_gram.iter().flatten().cloned())
        .collect())
}
