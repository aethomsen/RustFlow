//! Preparation, auxiliary-mass flow, and independent epsilon-fit refinement.
use crate::boundary::BoundaryProvider;
use crate::reduction::{ReducedSystem, differential_system};
use crate::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use symbolica::prelude::*;

mod euclidean;
mod frobenius_cache;
mod supplied;
pub use supplied::SuppliedAuxiliarySystem;

pub struct PreparedFlow {
    pub family: IntegralFamily,
    pub reduced: ReducedSystem,
    pub system: DifferentialSystem,
    frobenius_preparations: std::sync::Arc<frobenius_cache::FrobeniusPreparations>,
    epsilon_sample: Option<Rational>,
    deformation_mask: Vec<bool>,
    supplied: Option<supplied::SuppliedSeal>,
    // Retained independently of caller-mutable evaluation options and checked
    // on every evaluation, including systems loaded from the symbolic cache.
    principal_mass_constraints: Vec<Complex<Rational>>,
    // Public family fields can be replaced after preparation. A certificate
    // only applies to the exact family snapshot that was checked.
    positive_mass_contour: Option<String>,
    pub basis_refinement: Option<crate::refine::RefinementReport>,
}

impl PreparedFlow {
    pub fn new(
        family: &IntegralFamily,
        targets: &[Integral],
        point: &KinematicPoint,
        backend: &dyn ReductionBackend,
        options: &FlowOptions,
        context: &RunContext,
    ) -> Result<Self> {
        context.cancellation.check()?;
        options.validate()?;
        if options.recursion != RecursionMode::Amf {
            return Err(Error::Unsupported(
                "PreparedFlow prepares AMF systems; use solve_integrals for FT recursion".into(),
            ));
        }
        if targets.is_empty() {
            return Err(Error::InvalidInput("empty target list".into()));
        }
        let mut family = family.at(point);
        family.dimension = options.dimension;
        family.validate()?;
        for target in targets {
            family.validate_integral(target)?;
        }
        let principal_mass_constraints =
            crate::vacuum_peel::principal_mass_constraints(&family, targets)?;
        crate::vacuum_peel::validate_principal_mass_constraints(
            &principal_mass_constraints,
            options.prescription,
        )?;
        let eta = symbol!("symbolica_amflow::eta");
        let (auxiliary, mask) = family.deform(eta, &options.mass_mode)?;
        let positive_mass_contour =
            euclidean::positive_mass_contour(&auxiliary, eta)?.then(|| format!("{family:?}"));
        let cache = options
            .cache_directory
            .as_ref()
            .map(|directory| -> Result<_> {
                Ok((
                    directory,
                    crate::cache::system_key(&family, targets, backend, options)?,
                ))
            })
            .transpose()?;
        if let Some((directory, key)) = &cache
            && let Some((reduced, basis_refinement)) = crate::cache::read_system(directory, key)?
        {
            let system = DifferentialSystem {
                variable: eta,
                matrix: reduced.matrix.clone(),
            };
            return Ok(Self {
                family,
                reduced,
                system,
                basis_refinement,
                frobenius_preparations: Default::default(),
                epsilon_sample: None,
                deformation_mask: mask,
                supplied: None,
                principal_mass_constraints,
                positive_mass_contour,
            });
        }
        let session = crate::reduction::ReductionSession::new(backend);
        let backend = &session as &dyn ReductionBackend;
        let mut reduced = if options.skip_reduction {
            crate::reduction::differential_system_skip_initial(
                backend, &auxiliary, targets, &mask, 12, context,
            )?
        } else {
            differential_system(backend, &auxiliary, targets, &mask, 12, context)?
        };
        let basis_refinement = if options.refine_basis {
            Some(crate::refine::refine_basis(
                &mut reduced,
                eta,
                family.epsilon,
                32,
            )?)
        } else {
            None
        };
        if let Some((directory, key)) = cache {
            crate::cache::write_system(directory, &key, &reduced, &basis_refinement)?;
        }
        let system = DifferentialSystem {
            variable: eta,
            matrix: reduced.matrix.clone(),
        };
        context.emit(Progress::Prepared {
            basis_size: reduced.basis.len(),
            blocks: if reduced.basis.is_empty() {
                vec![]
            } else {
                system.blocks()?.iter().map(Vec::len).collect()
            },
        })?;
        Ok(Self {
            family,
            reduced,
            system,
            basis_refinement,
            frobenius_preparations: Default::default(),
            epsilon_sample: None,
            deformation_mask: mask,
            supplied: None,
            principal_mass_constraints,
            positive_mass_contour,
        })
    }
    /// Prepare at an exact regulator sample. Dimension-factorizing refinement
    /// retains symbolic epsilon instead, producing a reusable symbolic flow.
    pub fn new_at_epsilon(
        family: &IntegralFamily,
        targets: &[Integral],
        point: &KinematicPoint,
        backend: &dyn ReductionBackend,
        options: &FlowOptions,
        context: &RunContext,
        epsilon: &Rational,
    ) -> Result<Self> {
        if options.refine_basis {
            return Self::new(family, targets, point, backend, options, context);
        }
        let sampled = crate::reduction::SampledBackend {
            backend,
            epsilon: epsilon.clone(),
        };
        let mut flow = Self::new(family, targets, point, &sampled, options, context)?;
        flow.epsilon_sample = Some(epsilon.clone());
        Ok(flow)
    }

    fn lift_exponents(&self, basis: &mut frobenius::FrobeniusBasis) -> Result<()> {
        let Some(epsilon) = &self.epsilon_sample else {
            return Ok(());
        };
        let limit = 2 * self.family.loops.len() as i64;
        for column in &mut basis.columns {
            let value = if let AtomView::Num(n) = column.exponent.as_view() {
                if let symbolica::coefficient::Coefficient::Complex(c) =
                    n.get_coeff_view().to_owned()
                {
                    if !c.im.is_zero() {
                        return Err(Error::Unsupported("complex sampled indicial root".into()));
                    }
                    c.re
                } else {
                    return Err(Error::Unsupported(
                        "nonrational sampled indicial root".into(),
                    ));
                }
            } else {
                return Err(Error::Unsupported(
                    "nonnumeric sampled indicial root".into(),
                ));
            };
            let candidates = (-limit..=limit)
                .filter_map(|slope| {
                    let base = &value - &(epsilon * &Rational::from(slope));
                    (&base * &Rational::from(2))
                        .is_integer()
                        .then_some((base, slope))
                })
                .collect::<Vec<_>>();
            if candidates.len() != 1 {
                return Err(Error::Unsupported(format!(
                    "sample epsilon={epsilon} does not uniquely identify dimensional indicial sector {value} ({} candidates); a redundant reduction basis may require a deeper IBP search",
                    candidates.len()
                )));
            }
            let (base, slope) = &candidates[0];
            column.exponent =
                Atom::num(base.clone()) + Atom::num(*slope) * Atom::var(self.family.epsilon);
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        eps: &Rational,
        options: &FlowOptions,
        boundary: &dyn BoundaryProvider,
        context: &RunContext,
    ) -> Result<Vec<ComplexFloat>> {
        self.validate_supplied_contract(options)?;
        crate::vacuum_peel::validate_principal_mass_constraints(
            &self.principal_mass_constraints,
            options.prescription,
        )?;
        if self
            .epsilon_sample
            .as_ref()
            .is_some_and(|sample| sample != eps)
        {
            return Err(Error::InvalidInput(
                "prepared system belongs to a different epsilon sample".into(),
            ));
        }
        options.validate()?;
        context.cancellation.check()?;
        let digits = options
            .digits
            .checked_add(options.guard_digits)
            .ok_or_else(|| Error::InvalidInput("precision overflow".into()))?;
        let p = Precision::decimal(digits)?;
        let parameters =
            ahash::HashMap::from_iter([(Atom::var(self.family.epsilon), p.rational(eps))]);
        // Preserve original rational denominator domains before cancellation,
        // then specialize epsilon exactly. Floating substitution can miss an
        // identically zero eta-dependent condition at a rational sample.
        let epsilon_rule =
            BTreeMap::from([(Atom::var(self.family.epsilon), Atom::num(eps.clone()))]);
        let guards = crate::physical_conditions::canonical_conditions(
            &self.reduced.nonzero_conditions,
            &std::collections::BTreeSet::from([
                self.system.variable,
                self.family.epsilon,
                crate::family::imaginary_parameter(),
            ]),
        )?
        .iter()
        .map(|condition| {
            let specialized = crate::family::substitute(condition, &epsilon_rule)
                .together()
                .cancel();
            if specialized.is_zero() {
                return Err(Error::Reduction(format!(
                    "a reduction nonzero condition vanishes identically at epsilon={eps}"
                )));
            }
            Ok(specialized)
        })
        .collect::<Result<Vec<_>>>()?;
        if self.reduced.basis.is_empty() {
            return Ok(vec![p.zero(); self.reduced.targets.len()]);
        }
        let stage = |name: &str| {
            context.emit(Progress::Stage {
                name: format!(
                    "{name} ({} masters, {} loops)",
                    self.reduced.basis.len(),
                    self.family.loops.len()
                ),
            })
        };
        stage("compiling the auxiliary-mass differential equation")?;
        let mut compiled = self.system.compile(p, &parameters)?;
        compiled.exclude_polynomials(&guards)?;
        stage("constructing the expansion at auxiliary-mass infinity")?;
        let mut infinity = self
            .frobenius_preparations
            .get(&self.system, true, context)?
            .evaluate(p, &parameters, options.series_order, context)?;
        self.lift_exponents(&mut infinity)?;
        ensure_generic_indicial(&infinity, &parameters)?;
        stage("generating and matching native asymptotic boundary regions")?;
        let constants = boundary.constants_with_deformation(
            &self.family,
            &self.reduced.basis,
            &self.deformation_mask,
            eps,
            p,
            &infinity,
        )?;
        let maximum = compiled
            .poles
            .iter()
            .map(|v| p.norm(v))
            .fold(p.real(1), |a, b| if a > b { a } else { b });
        let direction = if options.prescription == Prescription::PlusI0 {
            -1
        } else {
            1
        };
        let positive_mass_contour = self
            .positive_mass_contour
            .as_ref()
            .is_some_and(|original| original == &format!("{:?}", self.family));
        let start = if positive_mass_contour {
            ComplexFloat::new(maximum * 8, p.real(0))
        } else {
            ComplexFloat::new(p.real(0), maximum * (8 * direction))
        };
        let at_start = infinity.evaluate(&p.div(&p.i(1), &start), &parameters)?;
        let initial = at_start
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&constants)
                    .fold(p.zero(), |s, (v, c)| p.add(&s, &p.mul(v, c)))
            })
            .collect();
        // A long infinity basis is no longer needed once its initial values
        // have been formed. Release it before allocating Taylor chart buffers.
        drop(infinity);
        let minimum = compiled
            .poles
            .iter()
            .map(|v| p.norm(v))
            .filter(|v| *v > p.real(0))
            .fold(p.real(1), |a, b| if a < b { a } else { b });
        let end = if positive_mass_contour {
            ComplexFloat::new(minimum / 8, p.real(0))
        } else {
            ComplexFloat::new(p.real(0), (minimum / 8) * direction)
        };
        let path = compiled.plan_path(&start, &end, -direction)?;
        stage("transporting the auxiliary-mass solution")?;
        let transported = compiled.transport(
            &BoundaryData {
                point: start,
                values: initial,
            },
            &path,
            options,
            context,
        )?;
        stage("constructing the physical endpoint expansion")?;
        let mut endpoint = self
            .frobenius_preparations
            .get(&self.system, false, context)?
            .evaluate(p, &parameters, options.series_order, context)?;
        self.lift_exponents(&mut endpoint)?;
        ensure_generic_indicial(&endpoint, &parameters)?;
        let constants = endpoint.match_values(&end, &transported.values, &parameters)?;
        stage("extracting the dimensionally regulated physical limit")?;
        // Reduction coefficients are multiplied into endpoint series before
        // selecting the physical constant; poles in these coefficients matter.
        self.reduced
            .targets
            .iter()
            .map(|target| {
                let weights = self
                    .reduced
                    .basis
                    .iter()
                    .map(|i| target.get(i).cloned().unwrap_or_default())
                    .collect::<Vec<_>>();
                project_limit(
                    &endpoint,
                    &constants,
                    &weights,
                    self.family.epsilon,
                    self.system.variable,
                    &parameters,
                )
            })
            .collect()
    }
}

pub(crate) fn ensure_generic_indicial(
    basis: &frobenius::FrobeniusBasis,
    parameters: &ahash::HashMap<Atom, ComplexFloat>,
) -> Result<()> {
    frobenius::ensure_generic_exponents(
        &basis
            .columns
            .iter()
            .map(|column| &column.exponent)
            .collect::<Vec<_>>(),
        basis.precision,
        parameters,
    )
}

pub(crate) use crate::frobenius::project_limit;

pub fn evaluate_samples(
    prepared: &PreparedFlow,
    samples: &[Rational],
    options: &FlowOptions,
    boundary: &dyn BoundaryProvider,
    context: &RunContext,
) -> Result<Vec<Vec<ComplexFloat>>> {
    options.validate()?;
    if samples.iter().any(|v| v.is_zero()) {
        return Err(Error::InvalidInput(
            "epsilon samples must be nonzero".into(),
        ));
    }
    if options.workers == 1 {
        return samples
            .iter()
            .enumerate()
            .map(|(index, eps)| {
                context.emit(Progress::Sample {
                    index,
                    total: samples.len(),
                })?;
                prepared.evaluate(eps, options, boundary, context)
            })
            .collect();
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    pool.install(|| {
        samples
            .par_iter()
            .enumerate()
            .map(|(index, eps)| {
                context.emit(Progress::Sample {
                    index,
                    total: samples.len(),
                })?;
                prepared.evaluate(eps, options, boundary, context)
            })
            .collect()
    })
}

fn has_linear_propagators(family: &IntegralFamily) -> bool {
    let quadratic = family.loops.len() * (family.loops.len() + 1) / 2;
    family
        .propagators
        .iter()
        .take(family.physical_propagators)
        .any(|d| {
            d.scalar_products.iter().take(quadratic).all(Atom::is_zero)
                && d.scalar_products
                    .iter()
                    .skip(quadratic)
                    .any(|a| !a.is_zero())
        })
}

pub fn solve_integrals(
    family: &IntegralFamily,
    targets: &[Integral],
    point: &KinematicPoint,
    last: i32,
    options: &FlowOptions,
    backend: &dyn ReductionBackend,
    context: &RunContext,
) -> Result<Vec<LaurentExpansion>> {
    options.validate()?;
    if targets.is_empty() {
        return Err(Error::InvalidInput("empty target list".into()));
    }
    // FT combines denominators directly, including linear ones. Its Schwinger
    // polynomial check below certifies the parameter domain before recursion;
    // it does not need the rank-one quadratic deformation used by AMF.
    if options.recursion == RecursionMode::Amf && has_linear_propagators(&family.at(point)) {
        let prepared = crate::linear::PreparedLinearFlow::new(
            family, targets, point, backend, options, context,
        )?;
        return fit_samples_refined(
            targets.len(),
            family.loops.len(),
            last,
            options,
            |samples, refined| {
                let evaluate = |(index, epsilon): (usize, &Rational)| {
                    context.emit(Progress::Sample {
                        index,
                        total: samples.len(),
                    })?;
                    Ok(prepared
                        .evaluate(epsilon, refined, backend, context)?
                        .values)
                };
                if refined.workers == 1 {
                    samples.iter().enumerate().map(evaluate).collect()
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(refined.workers)
                        .build()
                        .map_err(|e| Error::InvalidInput(e.to_string()))?
                        .install(|| samples.par_iter().enumerate().map(evaluate).collect())
                }
            },
        );
    }
    if options.recursion == RecursionMode::Ft {
        let mut family = family.at(point);
        family.dimension = options.dimension;
        let evaluator = crate::ft::FtEvaluator::new(backend, context);
        return fit_samples_refined(
            targets.len(),
            family.loops.len(),
            last,
            options,
            |samples, refined| {
                let evaluate = |(index, eps): (usize, &Rational)| {
                    context.emit(Progress::Sample {
                        index,
                        total: samples.len(),
                    })?;
                    targets
                        .iter()
                        .map(|target| evaluator.evaluate(&family, target, eps, refined))
                        .collect::<Result<Vec<_>>>()
                };
                if refined.workers == 1 {
                    samples.iter().enumerate().map(evaluate).collect()
                } else {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(refined.workers)
                        .build()
                        .map_err(|e| Error::InvalidInput(e.to_string()))?;
                    pool.install(|| samples.par_iter().enumerate().map(evaluate).collect())
                }
            },
        );
    }
    if options.sampled_reduction && !options.refine_basis && family.loops.len() > 1 {
        let boundary = crate::recursive::RecursiveBoundary::new(backend, options, context);
        return fit_samples_refined(
            targets.len(),
            family.loops.len(),
            last,
            options,
            |samples, refined| {
                let evaluate = |(index, epsilon): (usize, &Rational)| {
                    context.emit(Progress::Sample {
                        index,
                        total: samples.len(),
                    })?;
                    let prepared = PreparedFlow::new_at_epsilon(
                        family, targets, point, backend, refined, context, epsilon,
                    )?;
                    prepared.evaluate(epsilon, refined, &boundary, context)
                };
                if options.workers == 1 {
                    samples.iter().enumerate().map(evaluate).collect()
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(options.workers)
                        .build()
                        .map_err(|e| Error::InvalidInput(e.to_string()))?
                        .install(|| samples.par_iter().enumerate().map(evaluate).collect())
                }
            },
        );
    }
    let prepared = PreparedFlow::new(family, targets, point, backend, options, context)?;
    let boundary = crate::recursive::RecursiveBoundary::new(backend, options, context);
    solve_prepared(&prepared, last, options, &boundary, context)
}

pub fn solve_prepared(
    prepared: &PreparedFlow,
    last: i32,
    options: &FlowOptions,
    boundary: &dyn BoundaryProvider,
    context: &RunContext,
) -> Result<Vec<LaurentExpansion>> {
    fit_samples_refined(
        prepared.reduced.targets.len(),
        prepared.family.loops.len(),
        last,
        options,
        |samples, refined| evaluate_samples(prepared, samples, refined, boundary, context),
    )
}

/// Evaluate a sum of exact integral combinations, potentially from different
/// denominator completions or partial-fraction families. Coefficients are
/// applied at each exact nonzero epsilon sample before Laurent fitting. Their
/// epsilon poles extend the leading-power bound instead of being truncated.
pub fn solve_integral_combinations(
    groups: &[(IntegralFamily, crate::reduction::LinearCombination)],
    point: &KinematicPoint,
    last: i32,
    options: &FlowOptions,
    backend: &dyn ReductionBackend,
    context: &RunContext,
) -> Result<LaurentExpansion> {
    let groups = groups
        .iter()
        .map(|(family, terms)| (family.clone(), vec![terms.clone()]))
        .collect::<Vec<_>>();
    let mut values = solve_integral_projections(&groups, point, last, options, backend, context)?;
    Ok(values.remove(0))
}

/// Evaluate several exact linear projections together. Every family contributes
/// the same number of output rows, which are summed across families. Integral
/// targets shared by rows are evaluated once per epsilon sample, before any
/// projection is fitted. Empty rows contribute zero; weights remain exact
/// rational functions until numerical specialization.
pub fn solve_integral_projections(
    groups: &[(IntegralFamily, Vec<crate::reduction::LinearCombination>)],
    point: &KinematicPoint,
    last: i32,
    options: &FlowOptions,
    backend: &dyn ReductionBackend,
    context: &RunContext,
) -> Result<Vec<LaurentExpansion>> {
    solve_integral_projections_normalized(
        groups,
        point,
        last,
        options,
        backend,
        context,
        &Default::default(),
    )
}

/// Batched projections with explicitly registered algebraic roots and a common
/// analytic normalization. All factors are evaluated at each finite epsilon,
/// including during independent higher-precision refinement.
#[allow(clippy::too_many_arguments)]
pub fn solve_integral_projections_normalized(
    groups: &[(IntegralFamily, Vec<crate::reduction::LinearCombination>)],
    point: &KinematicPoint,
    last: i32,
    options: &FlowOptions,
    backend: &dyn ReductionBackend,
    context: &RunContext,
    factors: &crate::ProjectionFactors,
) -> Result<Vec<LaurentExpansion>> {
    solve_integral_projections_with_preparer(
        groups, point, last, options, backend, context, factors, None,
    )
}

/// Preparation is replaceable by a supplied closed connection; projection,
/// checkpointing, normalization and independent epsilon refinement stay shared.
pub(crate) type ProjectionPreparer<'a> =
    dyn Fn(&IntegralFamily, &[Integral], &FlowOptions, &RunContext) -> Result<PreparedFlow> + 'a;

#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_integral_projections_with_preparer(
    groups: &[(IntegralFamily, Vec<crate::reduction::LinearCombination>)],
    point: &KinematicPoint,
    last: i32,
    options: &FlowOptions,
    backend: &dyn ReductionBackend,
    context: &RunContext,
    factors: &crate::ProjectionFactors,
    supplied_preparer: Option<&ProjectionPreparer<'_>>,
) -> Result<Vec<LaurentExpansion>> {
    options.validate()?;
    context.cancellation.check()?;
    if supplied_preparer.is_some() && options.recursion != RecursionMode::Amf {
        return Err(Error::Unsupported(
            "a supplied auxiliary-mass connection requires AMF recursion".into(),
        ));
    }
    let factors = factors.at(point)?;
    let outputs = groups.first().map_or(0, |(_, rows)| rows.len());
    if outputs == 0 || groups.iter().any(|(_, rows)| rows.len() != outputs) {
        return Err(Error::InvalidInput(
            "integral projections require equal, nonzero output row counts".into(),
        ));
    }
    enum Preparation {
        Amf(PreparedFlow),
        Linear(crate::linear::PreparedLinearFlow),
        Sampled,
        Ft,
    }
    let mut prepared = Vec::new();
    let mut leading = 0i32;
    for (family, rows) in groups {
        if supplied_preparer.is_some() && family.dimension != options.dimension {
            return Err(Error::InvalidInput(
                "supplied projection family dimension differs from evaluation dimension".into(),
            ));
        }
        let mut family = family.at(point);
        family.dimension = options.dimension;
        family.validate()?;
        if factors.roots.iter().any(|r| r.symbol == family.epsilon) {
            return Err(Error::InvalidInput(
                "projection root cannot be epsilon".into(),
            ));
        }
        let allowed = factors
            .roots
            .iter()
            .map(|r| Atom::var(r.symbol))
            .chain([
                Atom::var(family.epsilon),
                Atom::var(crate::family::imaginary_parameter()),
            ])
            .collect::<std::collections::BTreeSet<_>>();
        let mut targets = std::collections::BTreeSet::new();
        let mut coefficients = Vec::with_capacity(outputs);
        let bound = i32::try_from(family.loops.len())
            .ok()
            .and_then(|n| n.checked_mul(-2))
            .ok_or_else(|| Error::Limit("Laurent pole bound overflow".into()))?;
        leading = leading.min(bound);
        for terms in rows {
            let mut row = BTreeMap::new();
            for (integral, coefficient) in terms {
                family.validate_integral(integral)?;
                let coefficient = point.apply(coefficient).together().cancel();
                if coefficient.is_zero() {
                    continue;
                }
                let valuation =
                    projection_weight_valuation(&coefficient, family.epsilon, &allowed)?;
                leading = leading.min(
                    bound
                        .checked_add(valuation)
                        .ok_or_else(|| Error::Limit("weighted Laurent bound overflow".into()))?,
                );
                targets.insert(integral.clone());
                row.insert(integral.clone(), coefficient);
            }
            coefficients.push(row);
        }
        let integrals = targets.into_iter().collect::<Vec<_>>();
        if integrals.is_empty() {
            continue;
        }
        let coefficients = coefficients
            .into_iter()
            .map(|row| {
                integrals
                    .iter()
                    .map(|integral| row.get(integral).cloned().unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let preparation = if let Some(prepare) = supplied_preparer {
            Preparation::Amf(prepare(&family, &integrals, options, context)?)
        } else if options.recursion == RecursionMode::Ft {
            Preparation::Ft
        } else if has_linear_propagators(&family) {
            Preparation::Linear(crate::linear::PreparedLinearFlow::new(
                &family,
                &integrals,
                &KinematicPoint::default(),
                backend,
                options,
                context,
            )?)
        } else if options.sampled_reduction && !options.refine_basis && family.loops.len() > 1 {
            Preparation::Sampled
        } else {
            Preparation::Amf(PreparedFlow::new(
                &family,
                &integrals,
                &KinematicPoint::default(),
                backend,
                options,
                context,
            )?)
        };
        if let Preparation::Amf(flow) = &preparation {
            // The immutable prepared flow is shared by every sample below.
            // Validate its sealed declaration even when all samples are hits.
            flow.validate_supplied_contract(options)?;
        }
        prepared.push((family, integrals, coefficients, preparation));
    }
    let boundary = crate::recursive::RecursiveBoundary::new(backend, options, context);
    let ft = crate::ft::FtEvaluator::new(backend, context);
    fit_samples_refined_leading(outputs, leading, last, options, |samples, refined| {
        let p = Precision::decimal(refined.digits + refined.guard_digits)?;
        let root_parameters = factors.parameters(p)?;
        let evaluate = |(index, epsilon): (usize, &Rational)| {
            context.emit(Progress::Sample {
                index,
                total: samples.len(),
            })?;
            let mut sums = vec![p.zero(); outputs];
            for (family, integrals, coefficients, preparation) in &prepared {
                let values = crate::sample_checkpoint::evaluate(
                    family,
                    integrals,
                    epsilon,
                    refined,
                    backend,
                    match preparation {
                        Preparation::Amf(flow) => flow.source_identity(),
                        _ => None,
                    },
                    || {
                        Ok(match preparation {
                            Preparation::Amf(flow) => {
                                flow.evaluate(epsilon, refined, &boundary, context)?
                            }
                            Preparation::Linear(flow) => {
                                flow.evaluate(epsilon, refined, backend, context)?.values
                            }
                            Preparation::Sampled => PreparedFlow::new_at_epsilon(
                                family,
                                integrals,
                                &KinematicPoint::default(),
                                backend,
                                refined,
                                context,
                                epsilon,
                            )?
                            .evaluate(epsilon, refined, &boundary, context)?,
                            Preparation::Ft => integrals
                                .iter()
                                .map(|i| ft.evaluate(family, i, epsilon, refined))
                                .collect::<Result<Vec<_>>>()?,
                        })
                    },
                )?;
                let mut parameters = root_parameters.clone();
                parameters.insert(Atom::var(family.epsilon), p.rational(epsilon));
                for (sum, row) in sums.iter_mut().zip(coefficients) {
                    for (coefficient, value) in row.iter().zip(&values) {
                        if !coefficient.is_zero() {
                            *sum = p.add(sum, &p.mul(&p.eval(coefficient, &parameters)?, value));
                        }
                    }
                }
            }
            let normalization = factors.evaluate(epsilon, p)?;
            Ok(sums
                .iter()
                .map(|value| p.mul(value, &normalization))
                .collect())
        };
        if refined.workers == 1 {
            samples.iter().enumerate().map(evaluate).collect()
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(refined.workers)
                .build()
                .map_err(|e| Error::InvalidInput(e.to_string()))?
                .install(|| samples.par_iter().enumerate().map(evaluate).collect())
        }
    })
}

/// Admit an exact specialized projection weight and retain its epsilon pole.
/// Shared by ordinary and cut projections before finite-sample multiplication.
pub(crate) fn projection_weight_valuation(
    coefficient: &Atom,
    epsilon: Symbol,
    allowed: &std::collections::BTreeSet<Atom>,
) -> Result<i32> {
    let mut symbols = Default::default();
    crate::family::scalar_symbols(coefficient.as_view(), &mut symbols)?;
    if !symbols.is_subset(allowed) {
        return Err(Error::InvalidInput(
            "projection coefficient has unsubstituted variables or unregistered roots".into(),
        ));
    }
    let encoded = crate::family::encode_complex(coefficient);
    let _: RationalPolynomial<IntegerRing, u16> = encoded
        .try_to_rational_polynomial(&Q, &Z, None)
        .map_err(|e| {
            Error::Unsupported(format!(
                "combination weights must be exact rational functions: {e}"
            ))
        })?;
    let expansion = encoded
        .series(epsilon, 0, 1)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let valuation = if expansion.is_zero() {
        0
    } else {
        expansion
            .get_trailing_exponent()
            .to_string()
            .parse::<i32>()
            .map_err(|_| {
                Error::Unsupported("noninteger epsilon valuation in combination weight".into())
            })?
            .min(0)
    };
    Ok(valuation)
}

fn fit_samples_refined(
    targets: usize,
    loops: usize,
    last: i32,
    options: &FlowOptions,
    evaluate: impl Fn(&[Rational], &FlowOptions) -> Result<Vec<Vec<ComplexFloat>>>,
) -> Result<Vec<LaurentExpansion>> {
    let leading = -2 * (loops as i32);
    fit_samples_refined_leading(targets, leading, last, options, evaluate)
}

pub(crate) fn fit_samples_refined_leading(
    targets: usize,
    leading: i32,
    last: i32,
    options: &FlowOptions,
    evaluate: impl Fn(&[Rational], &FlowOptions) -> Result<Vec<Vec<ComplexFloat>>>,
) -> Result<Vec<LaurentExpansion>> {
    options.validate()?;
    if last < leading {
        return Err(Error::InvalidInput(
            "last epsilon power precedes leading pole bound".into(),
        ));
    }
    let mut previous: Option<Vec<LaurentExpansion>> = None;
    let mut retry_digits = 0;
    for attempt in 0..=options.max_precision_attempts {
        let profile = epsilon::EpsilonFitProfile::new(
            options.digits,
            options.guard_digits,
            leading,
            last,
            attempt,
        )?;
        let current_digits = profile.working_digits.max(retry_digits);
        let mut refined = options.clone();
        refined.guard_digits = current_digits - options.digits;
        refined.series_order += attempt * 16;
        refined.refine_rational_order(attempt);
        refined.validate()?;
        let samples = profile.samples;
        let p = Precision::decimal(refined.digits + refined.guard_digits)?;
        // Numerical refinement is independent even after a precision hint has
        // overtaken the profile's baseline: increase the arithmetic precision,
        // shrink the exact radius and add nodes and transport terms.
        retry_digits = current_digits
            .checked_add(32)
            .ok_or_else(|| Error::Limit("epsilon fit precision overflow".into()))?;
        let values = match evaluate(&samples, &refined).and_then(|values| {
            if values.len() != samples.len() || values.iter().any(|row| row.len() != targets) {
                return Err(Error::InvalidInput(
                    "epsilon evaluator returned inconsistent sample dimensions".into(),
                ));
            }
            if values.iter().flatten().any(|value| !p.finite(value)) {
                return Err(Error::Numerical(
                    "nonfinite automatic epsilon sample".into(),
                ));
            }
            Ok(values)
        }) {
            Err(Error::InsufficientPrecision { minimum_bits, .. })
                if attempt < options.max_precision_attempts =>
            {
                retry_digits = Precision::refinement_digits(current_digits, minimum_bits)?;
                previous = None;
                continue;
            }
            Err(Error::Accuracy(_) | Error::Numerical(_))
                if attempt < options.max_precision_attempts =>
            {
                previous = None;
                continue;
            }
            result => result?,
        };
        let fits = (0..targets)
            .map(|i| {
                fit_epsilon(
                    &samples,
                    &values.iter().map(|row| row[i].clone()).collect::<Vec<_>>(),
                    leading,
                    last,
                    p,
                )
            })
            .collect::<Result<Vec<_>>>();
        let mut fits = match fits {
            Err(Error::InsufficientPrecision { minimum_bits, .. })
                if attempt < options.max_precision_attempts =>
            {
                retry_digits = Precision::refinement_digits(current_digits, minimum_bits)?;
                previous = None;
                continue;
            }
            Err(Error::Accuracy(_) | Error::Numerical(_))
                if attempt < options.max_precision_attempts =>
            {
                previous = None;
                continue;
            }
            result => result?,
        };
        if let Some(old) = &previous
            && old.iter().zip(&fits).all(|(a, b)| {
                a.coefficients.iter().all(|(k, v)| {
                    let next = &b.coefficients[k];
                    if !p.finite(v) || !p.finite(next) {
                        return false;
                    }
                    let old_magnitude = p.norm(v);
                    let new_magnitude = p.norm(next);
                    let magnitude = if old_magnitude > new_magnitude {
                        old_magnitude
                    } else {
                        new_magnitude
                    };
                    let scale = if magnitude <= p.tolerance(options.digits) {
                        p.real(1)
                    } else {
                        magnitude
                    };
                    p.norm(&p.sub(v, next)) <= p.tolerance(options.digits) * scale
                })
            })
        {
            for (f, previous) in fits.iter_mut().zip(old) {
                f.verified_digits = Some(options.digits);
                f.validation_samples = previous.samples;
                f.refinements = attempt;
                f.comparison_errors = f
                    .coefficients
                    .iter()
                    .map(|(k, c)| (*k, p.norm(&p.sub(c, &previous.coefficients[k]))))
                    .collect();
            }
            return Ok(fits);
        }
        previous = Some(fits);
    }
    Err(Error::Accuracy(
        "epsilon coefficients did not stabilize under independent sample/precision refinement"
            .into(),
    ))
}

#[cfg(test)]
mod projection_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balanced_laurent_reconstruction_verifies_poles_and_positive_orders() -> Result<()> {
        let calls = std::cell::RefCell::new(Vec::new());
        let options = FlowOptions {
            digits: 20,
            guard_digits: 24,
            ..Default::default()
        };
        let fits = fit_samples_refined_leading(1, -4, 2, &options, |samples, refined| {
            let p = Precision::decimal(refined.digits + refined.guard_digits)?;
            assert!(samples.iter().any(|value| value < &Rational::zero()));
            calls.borrow_mut().push(samples.len());
            // epsilon^-4/(1-epsilon) has every coefficient equal to one.
            Ok(samples
                .iter()
                .map(|sample| {
                    let epsilon = p.rational(sample);
                    vec![p.div(
                        &p.i(1),
                        &p.mul(&p.powi(&epsilon, 4), &p.sub(&p.i(1), &epsilon)),
                    )]
                })
                .collect())
        })?;
        assert_eq!(fits[0].verified_digits, Some(20));
        assert!(fits[0].validation_samples > 0);
        assert_eq!(*calls.borrow(), vec![32, 36]);
        let p = Precision::decimal(100)?;
        for power in -4..=2 {
            assert!(p.close(&fits[0].coefficients[&power], &p.i(1), 20));
        }
        Ok(())
    }

    #[test]
    fn balanced_laurent_fits_retain_zeros_and_tiny_relative_coefficients() -> Result<()> {
        let options = FlowOptions {
            digits: 20,
            guard_digits: 24,
            ..Default::default()
        };
        let fits = fit_samples_refined_leading(1, -2, 2, &options, |samples, refined| {
            let p = Precision::decimal(refined.digits + refined.guard_digits)?;
            let scale = p.parse("1e-30", "0")?;
            Ok(samples
                .iter()
                .map(|sample| {
                    let epsilon = p.rational(sample);
                    vec![p.mul(&scale, &p.add(&p.powi(&epsilon, -2), &p.powi(&epsilon, 2)))]
                })
                .collect())
        })?;
        let p = Precision::decimal(100)?;
        for power in -2_i32..=2 {
            let expected = if power.abs() == 2 {
                p.parse("1e-30", "0")?
            } else {
                p.zero()
            };
            assert!(
                p.norm(&p.sub(&fits[0].coefficients[&power], &expected))
                    < p.real(1) * p.parse("1e-45", "0")?.re
            );
        }
        Ok(())
    }

    #[test]
    fn balanced_fit_does_not_admit_a_regulator_pole_inside_the_sample_disk() -> Result<()> {
        let options = FlowOptions {
            digits: 20,
            guard_digits: 24,
            ..Default::default()
        };
        let outcome = fit_samples_refined_leading(1, 0, 0, &options, |samples, refined| {
            let p = Precision::decimal(refined.digits + refined.guard_digits)?;
            Ok(samples
                .iter()
                .map(|sample| {
                    let epsilon = p.rational(sample);
                    vec![p.div(&p.i(1), &p.sub(&p.i(1), &p.scale(&epsilon, 100_000, 1)))]
                })
                .collect())
        });
        assert!(matches!(outcome, Err(Error::Accuracy(_))));
        Ok(())
    }

    #[test]
    fn laurent_refinement_rebuilds_samples_after_a_precision_hint() -> Result<()> {
        let calls = std::cell::RefCell::new(Vec::new());
        let minimum = Precision::decimal(140)?.bits;
        let fits = fit_samples_refined_leading(1, 0, 0, &FlowOptions::default(), |samples, o| {
            let p = Precision::decimal(o.digits + o.guard_digits)?;
            calls.borrow_mut().push((p.bits, samples[0].clone()));
            if p.bits < minimum {
                return Err(Error::InsufficientPrecision {
                    minimum_bits: minimum,
                    context: "boundary reconstruction needs additional precision".into(),
                });
            }
            Ok(samples.iter().map(|_| vec![p.i(1)]).collect())
        })?;
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 3);
        assert!(calls[0].0 < minimum);
        assert!(calls[1].0 >= minimum && calls[2].0 > calls[1].0);
        assert_ne!(calls[0].1, calls[1].1);
        assert_ne!(calls[1].1, calls[2].1);
        assert_eq!(fits[0].verified_digits, Some(20));
        assert_eq!(fits[0].refinements, 2);
        assert!(Precision::decimal(160)?.close(
            &fits[0].coefficients[&0],
            &Precision::decimal(160)?.i(1),
            20
        ));
        Ok(())
    }

    #[test]
    fn fit_refinement_checks_relative_accuracy_for_small_coefficients() {
        for guard_digits in [40, 0] {
            let options = FlowOptions {
                guard_digits,
                ..Default::default()
            };
            let fits = fit_samples_refined(1, 1, 0, &options, |samples, options| {
                let p = Precision::decimal(options.digits + options.guard_digits)?;
                Ok(samples
                    .iter()
                    .map(|eps| {
                        let x = p.rational(eps);
                        vec![p.scale(
                            &p.add(&p.div(&p.i(1), &x), &p.div(&p.i(1), &p.sub(&p.i(1), &x))),
                            1,
                            1_000_000,
                        )]
                    })
                    .collect())
            })
            .unwrap();
            let p = Precision::decimal(80).unwrap();
            let expected = p.scale(&p.i(1), 1, 1_000_000);
            assert_eq!(fits[0].verified_digits, Some(20));
            for power in [-1, 0] {
                assert!(
                    p.norm(&p.sub(&fits[0].coefficients[&power], &expected))
                        < p.tolerance(20) * p.norm(&expected)
                );
                assert!(fits[0].comparison_errors[&power] < p.tolerance(20) * p.norm(&expected));
            }
        }
    }
}
