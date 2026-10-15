//! Explicit exact relations among generalized-series coefficients.
use crate::frobenius::exact::{
    Gaussian, gaussian, gaussian_bounded, multiply, row_reduce_with_mode,
};
use crate::frobenius::{ExactFrobeniusBasis, ExactFrobeniusLimits};
use crate::{Error, Result, RunContext};
use std::collections::BTreeMap;
use symbolica::prelude::*;

#[cfg(test)]
#[path = "exact_tests.rs"]
mod tests;

/// One coefficient in the common declared power/logarithm branch.
/// Components may represent an explicitly flattened epsilon hierarchy.
#[derive(Clone, Debug)]
pub struct AsymptoticSelector {
    pub component: usize,
    pub power: Atom,
    pub log_power: usize,
}

/// The explicitly asserted equality `sum(weight * selected coefficient) = value`.
/// Weights and values must be exact Gaussian-rational constants. Numerical
/// boundary values and their digit labels never imply one of these relations.
#[derive(Clone, Debug)]
pub struct ExactAsymptoticRelation {
    pub terms: Vec<(AsymptoticSelector, Atom)>,
    pub value: Atom,
}
impl ExactAsymptoticRelation {
    pub fn coefficient(selector: AsymptoticSelector, value: Atom) -> Self {
        Self {
            terms: vec![(selector, Atom::one())],
            value,
        }
    }
}

/// Exact physical information asserted separately from numerical input evidence.
#[derive(Clone, Debug)]
pub struct ExactAsymptoticConstraints {
    pub relations: Vec<ExactAsymptoticRelation>,
    pub provenance: String,
}

/// Exact affine integration constants `offset + directions * free_amplitudes`.
/// This describes correlations; it does not determine numerical amplitudes,
/// assert finiteness, or certify the accuracy of a matched regular boundary.
pub struct ExactAsymptoticSpace {
    pub(crate) offset: Vec<Gaussian>,
    pub(crate) directions: Vec<Vec<Gaussian>>,
    pub(crate) free_coordinates: Vec<usize>,
}
impl ExactAsymptoticSpace {
    pub fn dimension(&self) -> usize {
        self.offset.len()
    }
    pub fn free_amplitudes(&self) -> usize {
        self.directions.first().map_or(0, Vec::len)
    }
    pub fn offset(&self) -> &[Gaussian] {
        &self.offset
    }
    pub fn directions(&self) -> &[Vec<Gaussian>] {
        &self.directions
    }
}

fn zero() -> Gaussian {
    Gaussian::new(Rational::zero(), Rational::zero())
}
fn one() -> Gaussian {
    Gaussian::new(Rational::one(), Rational::zero())
}

impl ExactAsymptoticConstraints {
    pub(crate) fn key(&self) -> Result<String> {
        let relations = self
            .relations
            .iter()
            .map(|row| {
                (
                    row.terms
                        .iter()
                        .map(|(selector, weight)| {
                            (
                                selector.component,
                                selector.power.to_canonical_string(),
                                selector.log_power,
                                weight.to_canonical_string(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    row.value.to_canonical_string(),
                )
            })
            .collect::<Vec<_>>();
        let bytes =
            serde_json::to_vec(&("asserted-exact-asymptotic-v1", &self.provenance, relations))
                .map_err(|e| Error::Cache(e.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
    pub(crate) fn validate(
        &self,
        dimension: usize,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<()> {
        limits.validate()?;
        if self.provenance.trim().is_empty() || self.relations.is_empty() {
            return Err(Error::InvalidInput(
                "exact asymptotic constraints require explicit relations and provenance".into(),
            ));
        }
        let cells = dimension
            .checked_add(1)
            .and_then(|n| self.relations.len().checked_mul(n));
        let terms = self
            .relations
            .iter()
            .try_fold(0usize, |sum, row| sum.checked_add(row.terms.len()));
        if dimension > limits.max_dimension
            || cells.is_none_or(|v| v > limits.max_scalar_cells)
            || terms.is_none_or(|v| v > limits.max_scalar_cells)
        {
            return Err(Error::Limit(
                "exact asymptotic relation allocation exceeds limit".into(),
            ));
        }
        for relation in &self.relations {
            context.cancellation.check()?;
            if relation.terms.is_empty() {
                return Err(Error::InvalidInput(
                    "empty exact asymptotic relation".into(),
                ));
            }
            for (selector, weight) in &relation.terms {
                context.cancellation.check()?;
                if selector.component >= dimension {
                    return Err(Error::InvalidInput(
                        "exact asymptotic selector component out of range".into(),
                    ));
                }
                let power = gaussian_bounded(&selector.power, limits, context)?;
                if !power.im.is_zero() {
                    return Err(Error::Unsupported(
                        "exact coefficient selectors require real rational powers".into(),
                    ));
                }
                for value in [power, gaussian_bounded(weight, limits, context)?] {
                    if crate::frobenius::exact::height(&value) > limits.max_coefficient_bits {
                        return Err(Error::Limit(
                            "exact asymptotic input height exceeds limit".into(),
                        ));
                    }
                }
            }
            if crate::frobenius::exact::height(&gaussian_bounded(&relation.value, limits, context)?)
                > limits.max_coefficient_bits
            {
                return Err(Error::Limit(
                    "exact asymptotic value height exceeds limit".into(),
                ));
            }
        }
        Ok(())
    }
}

impl ExactFrobeniusBasis {
    /// Exact coefficient row in integration-constant coordinates. Missing
    /// orders are unknown, even if normalization padded the stored arrays.
    pub fn coefficient_row(&self, selector: &AsymptoticSelector) -> Result<Vec<Gaussian>> {
        if selector.component >= self.dimension() {
            return Err(Error::InvalidInput(
                "exact asymptotic component out of range".into(),
            ));
        }
        let power = gaussian_bounded(&selector.power, &self.limits, &RunContext::default())?;
        if !power.im.is_zero() {
            return Err(Error::Unsupported("nonreal exact asymptotic power".into()));
        }
        self.columns
            .iter()
            .map(|column| {
                let exponent = gaussian(&column.exponent)?;
                let difference = &power.re - &exponent.re;
                if difference < Rational::zero() || !difference.is_integer() {
                    return Ok(zero());
                }
                let offset = difference
                    .to_string()
                    .parse::<usize>()
                    .map_err(|_| Error::Limit("exact asymptotic selector order overflow".into()))?;
                if offset > self.known_order {
                    return Err(Error::Accuracy(
                        "exact Frobenius prefix is too short for the requested coefficient".into(),
                    ));
                }
                Ok(column.coefficients[offset]
                    .get(selector.log_power)
                    .map_or_else(zero, |row| row[selector.component].clone()))
            })
            .collect()
    }

    /// Preserve all remaining degrees of freedom instead of setting omitted
    /// coefficients or unknown integration constants to numerical zero.
    pub fn constrain(
        &self,
        constraints: &ExactAsymptoticConstraints,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<ExactAsymptoticSpace> {
        self.constrain_with_rows(constraints, Vec::new(), limits, context)
    }

    /// Additional exact equations are derived by the registered-root owner;
    /// they are never inferred from numerical matching values.
    pub(crate) fn constrain_with_rows(
        &self,
        constraints: &ExactAsymptoticConstraints,
        mut equations: Vec<Vec<Gaussian>>,
        limits: &ExactFrobeniusLimits,
        context: &RunContext,
    ) -> Result<ExactAsymptoticSpace> {
        context.cancellation.check()?;
        let n = self.dimension();
        constraints.validate(n, limits, context)?;
        if equations.iter().any(|row| row.len() != n + 1)
            || equations
                .len()
                .checked_add(constraints.relations.len())
                .and_then(|rows| rows.checked_mul(n + 1))
                .is_none_or(|cells| cells > limits.max_scalar_cells)
        {
            return Err(Error::Limit(
                "exact sheet/asymptotic equation allocation exceeds limit".into(),
            ));
        }
        for relation in &constraints.relations {
            context.cancellation.check()?;
            let mut row = vec![zero(); n + 1];
            for (selector, weight) in &relation.terms {
                context.cancellation.check()?;
                let weight = gaussian(weight)?;
                if weight.is_zero() {
                    continue;
                }
                for (value, coefficient) in row.iter_mut().zip(self.coefficient_row(selector)?) {
                    *value = &*value + &(&weight * &coefficient);
                    if crate::frobenius::exact::height(value) > limits.max_coefficient_bits {
                        return Err(Error::Limit(
                            "exact asymptotic accumulated relation height exceeds limit".into(),
                        ));
                    }
                }
            }
            row[n] = gaussian(&relation.value)?;
            equations.push(row);
        }
        let reduced = row_reduce_with_mode(equations, n, limits, context, self.elimination_mode)?;
        let mut pivots = Vec::new();
        let mut offset = vec![zero(); n];
        for row in &reduced {
            context.cancellation.check()?;
            if let Some(pivot) = row[..n].iter().position(|a| !a.is_zero()) {
                pivots.push(pivot);
                offset[pivot] = row[n].clone();
            } else if !row[n].is_zero() {
                return Err(Error::InvalidInput(
                    "inconsistent exact asymptotic constraints".into(),
                ));
            }
        }
        let free = (0..n).filter(|j| !pivots.contains(j)).collect::<Vec<_>>();
        let mut directions = vec![vec![zero(); free.len()]; n];
        for (k, &j) in free.iter().enumerate() {
            directions[j][k] = one();
            for row in &reduced {
                if let Some(pivot) = row[..n].iter().position(|a| !a.is_zero()) {
                    directions[pivot][k] = -row[j].clone();
                }
            }
        }
        context.cancellation.check()?;
        Ok(ExactAsymptoticSpace {
            offset,
            directions,
            free_coordinates: free,
        })
    }

    pub(crate) fn finite_projection(
        &self,
        space: &ExactAsymptoticSpace,
        physical: usize,
        context: &RunContext,
    ) -> Result<(Vec<Gaussian>, Vec<Vec<Gaussian>>)> {
        if physical == 0 || physical > self.dimension() || space.dimension() != self.dimension() {
            return Err(Error::InvalidInput(
                "exact endpoint projection dimensions".into(),
            ));
        }
        let n = self.dimension();
        let mut rows = BTreeMap::<(Rational, usize, usize), Vec<Gaussian>>::new();
        let mut finite = vec![vec![zero(); n]; physical];
        for (j, column) in self.columns.iter().enumerate() {
            context.cancellation.check()?;
            let exponent = gaussian(&column.exponent)?;
            if &exponent.re + &Rational::from(self.known_order + 1) <= Rational::zero() {
                return Err(Error::Accuracy(
                    "exact prefix does not cover every nonpositive normalized endpoint power"
                        .into(),
                ));
            }
            for (k, logs) in column
                .coefficients
                .iter()
                .enumerate()
                .take(self.known_order + 1)
            {
                let power = &exponent.re + &Rational::from(k);
                if power > Rational::zero() {
                    break;
                }
                for (log, values) in logs.iter().enumerate() {
                    for (i, value) in values.iter().take(physical).enumerate() {
                        if power.is_zero() && log == 0 {
                            finite[i][j] = value.clone();
                        } else {
                            rows.entry((power.clone(), log, i))
                                .or_insert_with(|| vec![zero(); n])[j] = value.clone();
                        }
                    }
                }
            }
        }
        // One augmented affine map checks the central offset and every admitted
        // perturbation direction with native exact matrix multiplication.
        let affine = space
            .offset
            .iter()
            .zip(&space.directions)
            .map(|(offset, directions)| {
                std::iter::once(offset.clone())
                    .chain(directions.iter().cloned())
                    .collect()
            })
            .collect::<Vec<Vec<Gaussian>>>();
        if !rows.is_empty()
            && multiply(
                rows.into_values().collect(),
                affine.clone(),
                &self.limits,
                context,
            )?
            .iter()
            .flatten()
            .any(|a| !a.is_zero())
        {
            return Err(Error::Accuracy("asserted exact relations leave a divergent endpoint power or logarithm in the affine uncertainty space".into()));
        }
        let result = multiply(finite, affine, &self.limits, context)?;
        context.cancellation.check()?;
        Ok((
            result.iter().map(|row| row[0].clone()).collect(),
            result.iter().map(|row| row[1..].to_vec()).collect(),
        ))
    }
}
