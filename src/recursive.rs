//! Recursive AMF boundary evaluation with exact region and tensor algebra.
use crate::boundary::{BoundaryProvider, LeadingBoundary, OneLoopBoundary, RegionBoundary};
use crate::frobenius::FrobeniusBasis;
use crate::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};
use symbolica::prelude::*;

/// Optional user terminal integrals. Returning None delegates to the native
/// terminals and recursive AMF construction.
pub trait TerminalProvider: Send + Sync {
    fn evaluate(
        &self,
        family: &IntegralFamily,
        integral: &Integral,
        epsilon: &Rational,
        precision: Precision,
    ) -> Result<Option<ComplexFloat>>;
}

#[derive(Default)]
struct Memo {
    flows: BTreeMap<String, Arc<PreparedFlow>>,
    values: BTreeMap<String, ComplexFloat>,
}

/// Automatic boundary provider. All recursive children share symbolic and
/// numerical memo tables; an active ancestry detects recursive cycles.
pub struct RecursiveBoundary<'a> {
    backend: &'a dyn ReductionBackend,
    options: &'a FlowOptions,
    context: &'a RunContext,
    memo: Arc<Mutex<Memo>>,
    ancestry: Vec<BTreeSet<String>>,
    terminal: Option<&'a dyn TerminalProvider>,
    vacuum_ft: Arc<crate::ft::FtEvaluator<'a>>,
}
impl<'a> RecursiveBoundary<'a> {
    pub fn new(
        backend: &'a dyn ReductionBackend,
        options: &'a FlowOptions,
        context: &'a RunContext,
    ) -> Self {
        Self {
            backend,
            options,
            context,
            memo: Arc::default(),
            ancestry: vec![],
            terminal: None,
            vacuum_ft: Arc::new(crate::ft::FtEvaluator::new(backend, context)),
        }
    }
    pub fn with_terminal(mut self, terminal: &'a dyn TerminalProvider) -> Self {
        self.terminal = Some(terminal);
        self
    }
    fn numerical_options(&self, p: Precision) -> FlowOptions {
        let mut options = self.options.clone();
        let working_digits = ((u64::from(p.bits.saturating_sub(16)) * 1000) / 3322) as u32;
        let extra = working_digits.saturating_sub(options.digits + options.guard_digits);
        options.guard_digits = options
            .guard_digits
            .max(working_digits.saturating_sub(options.digits));
        options.series_order += extra as usize * 4 / 5;
        options.refine_rational_order(extra.div_ceil(20) as usize);
        options
    }
    fn child(&self, keys: BTreeSet<String>) -> Self {
        let mut ancestry = self.ancestry.clone();
        ancestry.push(keys);
        Self {
            backend: self.backend,
            options: self.options,
            context: self.context,
            memo: self.memo.clone(),
            ancestry,
            terminal: self.terminal,
            vacuum_ft: self.vacuum_ft.clone(),
        }
    }
    pub fn evaluate(
        &self,
        family: &IntegralFamily,
        target: &Integral,
        epsilon: &Rational,
        p: Precision,
    ) -> Result<ComplexFloat> {
        Ok(self
            .evaluate_many(family, std::slice::from_ref(target), epsilon, p)?
            .remove(0))
    }

    /// Evaluate related boundary integrals with one flow per mass placement.
    /// Target reductions are projected into the endpoint series before its
    /// physical limit is selected, including poles in reduction coefficients.
    pub fn evaluate_many(
        &self,
        family: &IntegralFamily,
        targets: &[Integral],
        epsilon: &Rational,
        p: Precision,
    ) -> Result<Vec<ComplexFloat>> {
        self.context.cancellation.check()?;
        if targets.is_empty() {
            return Ok(vec![]);
        }
        let family_key = boundary_family_key(family)?;
        let mut requests = BTreeMap::new();
        let mut output_keys = Vec::with_capacity(targets.len());
        for target in targets {
            family.validate_integral(target)?;
            let problem = family.integral_key(target)?;
            let value_key = format!("{problem}:{epsilon}:{}", p.bits);
            output_keys.push(value_key.clone());
            requests.insert(target.clone(), (problem, value_key));
        }
        // None selects all lines in nonvacuum families. Vacuum targets retain
        // their individual massive-line choice, so different choices never
        // share an auxiliary deformation accidentally.
        let mut groups = BTreeMap::<Option<usize>, Vec<(Integral, String, String)>>::new();
        for (target, (problem, value_key)) in requests {
            self.context.cancellation.check()?;
            if self
                .memo
                .lock()
                .map_err(memo_error)?
                .values
                .contains_key(&value_key)
            {
                continue;
            }
            if self.ancestry.iter().any(|frame| frame.contains(&problem)) {
                return Err(Error::Limit("recursive boundary cycle detected".into()));
            }
            if self.ancestry.len() >= 32 {
                return Err(Error::Limit("boundary recursion exceeds 32 levels".into()));
            }
            let custom = if let Some(provider) = self.terminal {
                provider.evaluate(family, &target, epsilon, p)?
            } else {
                None
            };
            let value = if let Some(value) = custom {
                if !p.finite(&value) {
                    return Err(Error::Numerical(
                        "nonfinite terminal provider result".into(),
                    ));
                }
                Some(value)
            } else if scaleless(family, &target)? {
                Some(p.zero())
            } else if let Some(value) =
                crate::massless_banana::evaluate(family, &target, epsilon, p)?
            {
                Some(value)
            } else if let Some(value) = vacuum_terminal(family, &target, epsilon, p)? {
                Some(value)
            } else if let Some(plan) = crate::vacuum_peel::SingleMassPlan::find(family, &target)? {
                // Homogeneity removes one massive radial integration. The child
                // has strictly fewer loops and uses this same boundary provider.
                let child = self.child(BTreeSet::from([problem.clone()]));
                let value = child.evaluate(&plan.child, &plan.target, epsilon, p)?;
                let value = p.mul(&plan.prefactor(epsilon, p)?, &value);
                if !p.finite(&value) {
                    return Err(Error::Numerical(
                        "nonfinite single-mass boundary value".into(),
                    ));
                }
                Some(value)
            } else if family.external.is_empty()
                && family
                    .propagators
                    .iter()
                    .zip(&target.0)
                    .filter(|(d, n)| **n > 0 && !d.constant.is_zero())
                    .count()
                    == 1
            {
                Some(self.vacuum_ft.evaluate(
                    family,
                    &target,
                    epsilon,
                    &self.numerical_options(p),
                )?)
            } else {
                None
            };
            if let Some(value) = value {
                self.memo
                    .lock()
                    .map_err(memo_error)?
                    .values
                    .insert(value_key, value);
                continue;
            }
            let placement = if family.external.is_empty() {
                Some(
                    family
                        .propagators
                        .iter()
                        .zip(&target.0)
                        .position(|(d, &n)| n > 0 && !d.constant.is_zero())
                        .or_else(|| target.0.iter().position(|&n| n > 0))
                        .ok_or_else(|| Error::InvalidInput("empty boundary sector".into()))?,
                )
            } else {
                None
            };
            groups
                .entry(placement)
                .or_default()
                .push((target, problem, value_key));
        }
        for (placement, requests) in groups {
            self.context.cancellation.check()?;
            let mut options = self.numerical_options(p);
            let sampled = options.sampled_reduction
                && !options.refine_basis
                && family.loops.len() > 1
                && !(1..=4 * family.loops.len()).any(|difference| {
                    (epsilon * &Rational::from(2 * difference as i64)).is_integer()
                });
            options.mass_mode = placement.map_or(MassMode::All, |i| MassMode::Propagators(vec![i]));
            let targets = requests
                .iter()
                .map(|(target, _, _)| target.clone())
                .collect::<Vec<_>>();
            // PreparedFlow stores a reduction for each target. A family-only
            // cache key would accidentally reuse another target's projection.
            let mut key = format!(
                "{family_key}:{placement:?}:{targets:?}:{}:{}",
                options.skip_reduction, options.refine_basis
            );
            if sampled {
                key.push_str(&format!(":epsilon={epsilon}"));
            }
            let cached = self
                .memo
                .lock()
                .map_err(memo_error)?
                .flows
                .get(&key)
                .cloned();
            let prepared = if let Some(v) = cached {
                v
            } else {
                let v = Arc::new(if sampled {
                    PreparedFlow::new_at_epsilon(
                        family,
                        &targets,
                        &KinematicPoint::default(),
                        self.backend,
                        &options,
                        self.context,
                        epsilon,
                    )?
                } else {
                    PreparedFlow::new(
                        family,
                        &targets,
                        &KinematicPoint::default(),
                        self.backend,
                        &options,
                        self.context,
                    )?
                });
                self.memo
                    .lock()
                    .map_err(memo_error)?
                    .flows
                    .insert(key, v.clone());
                v
            };
            // One ancestry frame per recursive solve, regardless of batch size.
            let child = self.child(
                requests
                    .iter()
                    .map(|(_, problem, _)| problem.clone())
                    .collect(),
            );
            let provider = RecursiveBoundary {
                options: &options,
                ..child
            };
            let values = prepared.evaluate(epsilon, &options, &provider, self.context)?;
            if values.len() != requests.len() {
                return Err(Error::Numerical("boundary batch result dimension".into()));
            }
            let mut memo = self.memo.lock().map_err(memo_error)?;
            for ((_, _, key), value) in requests.into_iter().zip(values) {
                memo.values.insert(key, value);
            }
        }
        let memo = self.memo.lock().map_err(memo_error)?;
        output_keys
            .iter()
            .map(|key| {
                memo.values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| Error::Numerical("missing boundary batch result".into()))
            })
            .collect()
    }
}

fn memo_error<T>(_: std::sync::PoisonError<T>) -> Error {
    Error::Numerical("boundary memo poisoned".into())
}

fn boundary_family_key(family: &IntegralFamily) -> Result<String> {
    Ok(format!(
        "{}:{family:?}",
        family.convert()?.family.fingerprint()
    ))
}

struct BoundaryRequestFamily {
    family: IntegralFamily,
    targets: BTreeSet<Integral>,
}

struct BoundaryProduct {
    coefficient: Atom,
    factors: Vec<String>,
}

struct PendingBoundaryCoefficient {
    component: usize,
    series: usize,
    order: usize,
    jacobian: ComplexFloat,
    products: Vec<BoundaryProduct>,
}

pub fn scaleless(family: &IntegralFamily, integral: &Integral) -> Result<bool> {
    use rustred::sector::{
        Mask,
        zero::{Analyzer, Decision},
    };
    let converted = family.convert()?;
    let analyzer = Analyzer::try_unrestricted(&converted.family)
        .map_err(|e| Error::Reduction(e.to_string()))?;
    let mask = Mask::try_new(integral.0.iter().map(|&n| n > 0))
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    Ok(matches!(
        analyzer
            .analyze(&mask)
            .map_err(|e| Error::Reduction(e.to_string()))?,
        Decision::ProvedZero(_)
    ))
}

fn vacuum_terminal(
    family: &IntegralFamily,
    target: &Integral,
    epsilon: &Rational,
    p: Precision,
) -> Result<Option<ComplexFloat>> {
    if !family.external.is_empty() {
        return Ok(None);
    }
    let dimension = Rational::from(family.dimension) - epsilon * &Rational::from(2);
    let params = ahash::HashMap::from_iter([(Atom::var(family.epsilon), p.rational(epsilon))]);
    if family.loops.len() == 1 && target.0.len() == 1 && target.0[0] > 0 {
        let d = &family.propagators[0];
        let scale = p.eval(&d.scalar_products[0], &params)?;
        let m = p.neg(&p.div(&p.eval(&d.constant, &params)?, &scale));
        return Ok(Some(p.mul(
            &p.powi(&scale, -i64::from(target.0[0])),
            &vacuum::tadpole(target.0[0] as u32, &m, &dimension, p)?,
        )));
    }
    if family.loops.len() != 2 || target.0.len() != 3 || target.0.iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    let masses = family
        .propagators
        .iter()
        .enumerate()
        .filter_map(|(i, d)| (!d.constant.is_zero()).then_some(i))
        .collect::<Vec<_>>();
    if masses.len() != 1 {
        return Ok(None);
    }
    let a = masses[0];
    let b = (0..3).find(|&j| j != a).unwrap();
    let c = (0..3).find(|&j| j != a && j != b).unwrap();
    let branches = family
        .propagators
        .iter()
        .map(|d| crate::regions::branch(d, 2))
        .collect::<Result<Vec<_>>>()?;
    let mut routing = crate::algebra::inverse(&[branches[a].clone(), branches[b].clone()])?;
    let third = crate::algebra::matmul(&[branches[c].clone()], &routing);
    if third[0].iter().any(|v| v.is_zero()) {
        return Ok(None);
    }
    let ratio = (&third[0][0] / &third[0][1]).together().cancel();
    for row in &mut routing {
        row[1] = (&row[1] * &ratio).together().cancel();
    }
    let transformed = family.transform_loops(&routing)?;
    let scales = [
        transformed.propagators[a].scalar_products[0].clone(),
        transformed.propagators[b].scalar_products[2].clone(),
        transformed.propagators[c].scalar_products[0].clone(),
    ];
    let mass = (-&transformed.propagators[a].constant / &scales[0])
        .together()
        .cancel();
    let indices = [target.0[a] as u32, target.0[b] as u32, target.0[c] as u32];
    let mut value = vacuum::single_mass_sunset(indices, &p.eval(&mass, &params)?, &dimension, p)?;
    for (s, n) in scales.iter().zip(indices) {
        value = p.mul(&value, &p.powi(&p.eval(s, &params)?, -i64::from(n)));
    }
    let determinant = p.eval(&crate::algebra::determinant(routing), &params)?;
    let jacobian = ComplexFloat::new(p.norm(&determinant), p.real(0));
    Ok(Some(
        p.mul(&value, &p.pow(&jacobian, &p.rational(&dimension))),
    ))
}

impl BoundaryProvider for RecursiveBoundary<'_> {
    fn leading(
        &self,
        family: &IntegralFamily,
        basis: &[Integral],
        epsilon: &Rational,
        p: Precision,
    ) -> Result<Vec<LeadingBoundary>> {
        OneLoopBoundary.leading(family, basis, epsilon, p)
    }
    fn constants(
        &self,
        family: &IntegralFamily,
        basis: &[Integral],
        epsilon: &Rational,
        p: Precision,
        solutions: &FrobeniusBasis,
    ) -> Result<Vec<ComplexFloat>> {
        let (_, shifted) =
            family.deform(symbol!("symbolica_amflow::eta"), &self.options.mass_mode)?;
        self.constants_with_deformation(family, basis, &shifted, epsilon, p, solutions)
    }

    fn constants_with_deformation(
        &self,
        family: &IntegralFamily,
        basis: &[Integral],
        shifted: &[bool],
        epsilon: &Rational,
        p: Precision,
        solutions: &FrobeniusBasis,
    ) -> Result<Vec<ComplexFloat>> {
        crate::boundary::validate_deformation_mask(family, shifted)?;
        if family.loops.len() == 1
            && crate::boundary::all_physical_lines_shifted(family, shifted)
            && family
                .propagators
                .iter()
                .all(|d| d.scalar_products.first() == Some(&Atom::num(1)))
        {
            return solutions.match_leading(&self.leading(family, basis, epsilon, p)?);
        }
        let regions = crate::regions::enumerate_regions(family, 10000)?;
        let parameters =
            ahash::HashMap::from_iter([(Atom::var(family.epsilon), p.rational(epsilon))]);
        let powers = basis
            .iter()
            .map(|integral| {
                regions
                    .iter()
                    .map(|region| {
                        crate::regions::expand_region(family, integral, shifted, region, 0)
                            .map(|leading| -leading.eta_power)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let orders = solutions.region_orders(&powers, 32)?;
        self.context.emit(Progress::BoundaryPlan {
            basis_size: basis.len(),
            region_series: orders
                .iter()
                .flatten()
                .filter(|order| order.is_some())
                .count(),
            coefficients: orders
                .iter()
                .flatten()
                .flatten()
                .map(|order| order + 1)
                .sum(),
            max_half_order: orders
                .iter()
                .flatten()
                .flatten()
                .copied()
                .max()
                .unwrap_or(0),
        })?;
        let mut data = vec![Vec::new(); basis.len()];
        let mut pending = Vec::new();
        let mut requests = BTreeMap::<String, BoundaryRequestFamily>::new();
        let mut zero = BTreeMap::new();
        for (i, integral) in basis.iter().enumerate() {
            for (r, region) in regions.iter().enumerate() {
                self.context.cancellation.check()?;
                // Empty arrays retain unknown regions/parity classes. Computed
                // coefficients are filled only after their factor values exist.
                for parity in 0..2 {
                    let count = orders[i][r].map_or(0, |order| (order + 2 - parity) / 2);
                    data[i].push(RegionBoundary {
                        exponent: (&powers[i][r] + Atom::num((parity as i64, 2)))
                            .together()
                            .cancel(),
                        coefficients: vec![p.zero(); count],
                    });
                }
                let Some(half_order) = orders[i][r] else {
                    continue;
                };
                let expansion =
                    crate::regions::expand_region(family, integral, shifted, region, half_order)?;
                let determinant = p.eval(&expansion.jacobian_determinant, &parameters)?;
                let jacobian = p.pow(
                    &ComplexFloat::new(p.norm(&determinant), p.real(0)),
                    &p.rational(&(Rational::from(family.dimension) - epsilon * &Rational::from(2))),
                );
                for (k, expression) in expansion.coefficients.iter().enumerate() {
                    let factors = crate::integrand::factor_region(
                        expression,
                        &expansion.coordinates,
                        family,
                        &region.hard,
                        10000,
                    )?;
                    let mut products = Vec::new();
                    for term in factors {
                        let mut keys = Vec::new();
                        let mut vanishes = false;
                        for factor in &term.factors {
                            let key = factor.family.integral_key(&factor.integral)?;
                            let is_zero = if let Some(value) = zero.get(&key) {
                                *value
                            } else {
                                let value = scaleless(&factor.family, &factor.integral)?;
                                zero.insert(key.clone(), value);
                                value
                            };
                            if is_zero {
                                vanishes = true;
                                break;
                            }
                            keys.push(key);
                        }
                        // Check every factor before requesting any recursive
                        // values: one scaleless factor annihilates the product.
                        if vanishes {
                            continue;
                        }
                        for factor in term.factors {
                            let key = boundary_family_key(&factor.family)?;
                            requests
                                .entry(key)
                                .or_insert_with(|| BoundaryRequestFamily {
                                    family: factor.family,
                                    targets: BTreeSet::new(),
                                })
                                .targets
                                .insert(factor.integral);
                        }
                        products.push(BoundaryProduct {
                            coefficient: term.coefficient,
                            factors: keys,
                        });
                    }
                    pending.push(PendingBoundaryCoefficient {
                        component: i,
                        series: 2 * r + k % 2,
                        order: k / 2,
                        jacobian: jacobian.clone(),
                        products,
                    });
                }
            }
        }
        let mut values = BTreeMap::new();
        for request in requests.into_values() {
            self.context.cancellation.check()?;
            let targets = request.targets.into_iter().collect::<Vec<_>>();
            let evaluated = self.evaluate_many(&request.family, &targets, epsilon, p)?;
            for (target, value) in targets.iter().zip(evaluated) {
                values.insert(request.family.integral_key(target)?, value);
            }
        }
        for coefficient in pending {
            self.context.cancellation.check()?;
            let mut value = p.zero();
            for product in coefficient.products {
                let mut term = p.eval(&product.coefficient, &parameters)?;
                for key in product.factors {
                    let factor = values.get(&key).ok_or_else(|| {
                        Error::Numerical("missing collected boundary factor".into())
                    })?;
                    term = p.mul(&term, factor);
                }
                value = p.add(&value, &term);
            }
            data[coefficient.component][coefficient.series].coefficients[coefficient.order] =
                p.mul(&value, &coefficient.jacobian);
        }
        solutions.match_regions(&data)
    }
}
