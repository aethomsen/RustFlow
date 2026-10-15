//! Runtime orchestration of RustRed's native exact sector solver with a
//! factorized coefficient field. Source derivation, zero certificates, seed
//! search and exact replay remain entirely in RustRed.
mod parametric;
mod symmetry;

use crate::family::{ConvertedFamily, substitute};
use crate::reduction::{LinearCombination, Reduction, RustRedBackend};
use crate::{Error, Integral, IntegralFamily, Progress, Result, RunContext};
use rayon::prelude::*;
use rustred::sector::{
    Mask,
    zero::{Analyzer, Decision},
};
use rustred::solver::{
    CoordinateCase, NumericalExactBackend, SearchOptions, SectorConfig, SectorSolver, SourceSystem,
};
use std::collections::{BTreeMap, BTreeSet};
use symbolica::prelude::*;

pub(crate) fn reduce(
    family: &ConvertedFamily,
    original: &IntegralFamily,
    dimension: &Atom,
    targets: &[Vec<i16>],
    options: &RustRedBackend,
    context: &RunContext,
    cuts: Option<&crate::cuts::CutFamily>,
    seed: Option<&crate::reduction::ResolvedIdentityBank>,
    provider: Option<&dyn crate::reduction::NativeIdentityProvider>,
) -> Result<Reduction> {
    let cut_options;
    let options = if cuts.is_some() {
        cut_options = RustRedBackend {
            bubble_subloops: false,
            parametric_rules: false,
            symmetry_rules: false,
            ..options.clone()
        };
        &cut_options
    } else {
        options
    };
    rustred::dispatch_arity!(
        family.family.denominator_count(),
        solve(family, original, dimension, targets, options, context, cuts, seed, provider),
        n => Err(Error::Unsupported(format!(
            "native runtime was compiled for {:?} scalar-product slots; received {n}",
            rustred::compiled_runtime_arities()
        )))
    )
}
fn cut_stage_key(key: String, cuts: Option<&str>) -> String {
    cuts.map_or_else(
        || key.clone(),
        |cuts| {
            blake3::hash(format!("native-cut-v1:{cuts}:{key}").as_bytes())
                .to_hex()
                .to_string()
        },
    )
}
fn native_error(error: impl std::fmt::Display) -> Error {
    Error::Reduction(error.to_string())
}
fn provider_key(
    key: String,
    provider: Option<&dyn crate::reduction::NativeIdentityProvider>,
) -> String {
    provider.map_or(key.clone(), |p| {
        format!(
            "provider-v1-{}",
            blake3::hash(format!("{}:{}:{key}", p.identity(), p.max_calls()).as_bytes()).to_hex()
        )
    })
}
fn identity_seed_key(key: String, seed: Option<&crate::reduction::ResolvedIdentityBank>) -> String {
    seed.map_or_else(
        || key.clone(),
        |seed| {
            format!(
                "identity-seed-v1-{}",
                blake3::hash(format!("{}:{key}", seed.identity).as_bytes()).to_hex()
            )
        },
    )
}

/// Reuse verified equations before counting solver searches. Supplied residuals
/// are only declarations for table admission; they never establish that a leaf
/// has been searched. The bank has its own independent, immutable budget.
fn validate_identity_order<const N: usize>(
    seed: &crate::reduction::ResolvedIdentityBank,
    original: &IntegralFamily,
    dimension: &Atom,
    options: &RustRedBackend,
    context: &RunContext,
) -> Result<BTreeSet<[i16; N]>> {
    let mut supplied = BTreeSet::new();
    let mut orders = BTreeMap::new();
    for (lhs, terms) in &seed.reduction.rules {
        context.cancellation.check()?;
        let powers: [i16; N] = lhs
            .0
            .as_slice()
            .try_into()
            .map_err(|_| Error::InvalidInput("supplied native identity arity".into()))?;
        let sector = powers.map(|p| p > 0);
        if let std::collections::btree_map::Entry::Vacant(entry) = orders.entry(sector) {
            let mut order = rustred::solver::IntegralOrder::new(sector, [false; N]);
            if options.bubble_subloops {
                if let Some(pattern) =
                    crate::bubble::Bubble::find(original, &sector, dimension.clone())?
                {
                    order = order
                        .with_permutation(pattern.permutation().try_into().map_err(|_| {
                            Error::InvalidInput("supplied identity bubble permutation arity".into())
                        })?)
                        .map_err(native_error)?;
                }
            }
            entry.insert(order);
        }
        let left = rustred::solver::Integral::numeric(powers).map_err(native_error)?;
        for rhs in terms.keys() {
            let right: [i16; N] = rhs
                .0
                .as_slice()
                .try_into()
                .map_err(|_| Error::InvalidInput("supplied identity RHS arity".into()))?;
            let right = rustred::solver::Integral::numeric(right).map_err(native_error)?;
            if orders[&sector].compare(&left, &right) != std::cmp::Ordering::Less {
                return Err(Error::InvalidInput(
                    "supplied native identity does not strictly descend the active native order"
                        .into(),
                ));
            }
        }
        supplied.insert(powers);
    }
    Ok(supplied)
}
fn admit_identity_seed<const N: usize>(
    seed: &crate::reduction::ResolvedIdentityBank,
    original: &IntegralFamily,
    dimension: &Atom,
    options: &RustRedBackend,
    context: &RunContext,
    result: &mut Reduction,
    cache: &mut FrontierCache,
    visited: &mut BTreeSet<[i16; N]>,
    pending: &mut BTreeSet<[i16; N]>,
    targets: &[Vec<i16>],
) -> Result<()> {
    let supplied = validate_identity_order::<N>(seed, original, dimension, options, context)?;
    result
        .nonzero_conditions
        .extend(seed.reduction.nonzero_conditions.iter().cloned());
    cache.insert_rules(result, seed.reduction.rules.clone());
    // A reusable family bank labels every stored LHS as done. Supplied LHSs
    // belong to the separately bounded equation bank, not the search budget.
    // No unrelated searched index is removed, and unknown leaves stay pending.
    visited.retain(|powers| !supplied.contains(powers));
    *pending = structural_frontier(result, targets, context)?
        .into_iter()
        .map(|i| {
            let powers: [i16; N] =
                i.0.as_slice()
                    .try_into()
                    .map_err(|_| Error::Reduction("seeded frontier arity".into()))?;
            Ok(powers)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    pending.retain(|powers| !visited.contains(powers));
    Ok(())
}
fn family_stage_key(
    family: &ConvertedFamily,
    physical_propagators: usize,
    options: &RustRedBackend,
) -> String {
    let digest = blake3::hash(
        format!(
            "native-family-v1/native-factorized-v5/bubble-subloops-v2/symmetry-v1:{}:{:?}:{}:{}:{}:{}:{}:{}:{}:{}",
            family.family.fingerprint(),
            family.reverse,
            physical_propagators,
            options.max_depth,
            options.include_lorentz,
            options.bubble_subloops,
            options.parametric_rules,
            options.symmetry_rules,
            env!("RUSTRED_SOURCE_DIGEST"),
            env!("DEPENDENCY_SOURCE_DIGEST"),
        )
        .as_bytes(),
    );
    // A recognizable key prefix distinguishes this reusable identity bank
    // from a target-specific search checkpoint, without changing the codec.
    format!("family-v1-{digest}")
}

/// Reuse equations, never a previous call's declaration that a leaf was
/// searched. All reachable residual leaves must be searched in the new call.
fn restrict_family_stage(
    mut reduction: Reduction,
    targets: &[Vec<i16>],
    context: &RunContext,
) -> Result<(Reduction, Vec<Integral>, Vec<Integral>)> {
    let frontier = structural_frontier(&reduction, targets, context)?;
    let mut reachable = BTreeSet::new();
    let mut stack = targets.iter().cloned().map(Integral).collect::<Vec<_>>();
    while let Some(node) = stack.pop() {
        context.cancellation.check()?;
        if reachable.insert(node.clone())
            && let Some(terms) = reduction.rules.get(&node)
        {
            stack.extend(terms.keys().cloned());
        }
    }
    reduction.rules.retain(|lhs, _| reachable.contains(lhs));
    let done = reduction.rules.keys().cloned().collect();
    reduction.residuals = frontier.iter().cloned().collect();
    Ok((reduction, done, frontier.into_iter().collect()))
}
fn save_stage<const N: usize>(
    options: &RustRedBackend,
    key: &str,
    reduction: &mut Reduction,
    visited: &BTreeSet<[i16; N]>,
    pending: &BTreeSet<[i16; N]>,
) -> Result<()> {
    if let Some(directory) = &options.checkpoints {
        reduction.residuals.sort();
        reduction.residuals.dedup();
        reduction.nonzero_conditions.sort();
        reduction.nonzero_conditions.dedup();
        crate::cache::write_native_stage(
            directory,
            key,
            reduction,
            &visited
                .iter()
                .map(|i| Integral(i.to_vec()))
                .collect::<Vec<_>>(),
            &pending
                .iter()
                .map(|i| Integral(i.to_vec()))
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}
fn save_if_due<const N: usize>(
    options: &RustRedBackend,
    key: &str,
    reduction: &mut Reduction,
    visited: &BTreeSet<[i16; N]>,
    pending: &BTreeSet<[i16; N]>,
    last_saved: &mut std::time::Instant,
) -> Result<()> {
    if options.checkpoints.is_some() && last_saved.elapsed() >= options.checkpoint_interval {
        save_stage(options, key, reduction, visited, pending)?;
        *last_saved = std::time::Instant::now();
    }
    Ok(())
}
fn convert_batch<const N: usize>(
    solved: rustred::solver::NumericResult<N>,
    family: &ConvertedFamily,
) -> Result<(Reduction, Progress)> {
    let mut result = Reduction::default();
    let completed = Progress::SectorReduced {
        integrals: solved.stats.cases,
        seeds: solved.stats.seeds,
        rows: solved.stats.rows,
        exact_trace_rows: solved.stats.exact_trace_rows,
        elapsed_ms: solved.stats.elapsed.as_millis(),
        exact_ms: solved.stats.exact_materialization.as_millis(),
    };
    result.residuals.extend(
        solved
            .residuals
            .into_iter()
            .map(|c| Integral(c.integral().powers().iter().map(|p| p.value()).collect())),
    );
    for rule in solved.rules {
        if rule
            .target
            .powers()
            .iter()
            .chain(rule.rhs.iter().flat_map(|t| t.integral.powers()))
            .any(|p| p.is_symbolic())
        {
            return Err(Error::Reduction(
                "concrete search returned symbolic indices".into(),
            ));
        }
        let target = Integral(rule.target.powers().iter().map(|p| p.value()).collect());
        let mut combination = LinearCombination::new();
        for term in rule.rhs {
            let powers: [i16; N] = std::array::from_fn(|i| term.integral.powers()[i].value());
            let integral = Integral(powers.to_vec());
            let coefficient = substitute(&term.coefficient.to_expression(), &family.reverse);
            // Native coefficients are already reduced rational
            // polynomials. Replacing their parameter names is
            // bijective and needs no second polynomial gcd.
            if let Some(previous) = combination.get_mut(&integral) {
                *previous = (&*previous + coefficient).together().cancel();
            } else {
                combination.insert(integral, coefficient);
            }
            if !term.coefficient.denominator.is_constant() {
                result.nonzero_conditions.push(substitute(
                    &term.coefficient.denominator.to_expression(),
                    &family.reverse,
                ));
            }
        }
        combination.retain(|_, c| !c.is_zero());
        result.rules.insert(target, combination);
    }
    Ok((result, completed))
}
fn solve<const N: usize>(
    family: &ConvertedFamily,
    original: &IntegralFamily,
    dimension: &Atom,
    targets: &[Vec<i16>],
    options: &RustRedBackend,
    context: &RunContext,
    cuts: Option<&crate::cuts::CutFamily>,
    seed: Option<&crate::reduction::ResolvedIdentityBank>,
    provider: Option<&dyn crate::reduction::NativeIdentityProvider>,
) -> Result<Reduction> {
    let mut pending = targets
        .iter()
        .map(|t| {
            t.as_slice()
                .try_into()
                .map_err(|_| Error::InvalidInput("native integral arity".into()))
        })
        .collect::<Result<BTreeSet<[i16; N]>>>()?;
    let initial = pending.iter().map(|p| p.map(|n| n > 0)).collect::<Vec<_>>();
    let sources =
        SourceSystem::<N>::from_family_with_lorentz(&family.family, options.include_lorentz)
            .map_err(native_error)?;
    let deltas: [bool; N] = std::array::from_fn(|slot| cuts.is_some_and(|c| c.cuts().is_cut(slot)));
    let cut_key = cuts.map(crate::cuts::CutFamily::fingerprint).transpose()?;
    let analyzer = if let Some(cuts) = cuts {
        Analyzer::try_new(&family.family, cuts.native_restrictions()?)
    } else {
        Analyzer::try_unrestricted(&family.family)
    }
    .map_err(native_error)?;
    let sector_count = u32::try_from(N)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
        .ok_or_else(|| {
            Error::Limit(format!(
                "native sector enumeration cannot represent {N} scalar-product slots on this host"
            ))
        })?;
    let mut zero = Vec::new();
    for bits in 0..sector_count {
        context.cancellation.check()?;
        let sector = std::array::from_fn(|i| bits & (1 << i) != 0);
        // A verified automorphism may route a target into another physical
        // sector, so that lane needs zero certificates outside the initial
        // top sectors too. Positive irreducible numerators remain excluded.
        let relevant = if options.symmetry_rules {
            sector
                .iter()
                .enumerate()
                .all(|(axis, &active)| !active || axis < original.physical_propagators)
        } else {
            initial
                .iter()
                .any(|parent| sector.iter().zip(parent).all(|(a, b)| !a || *b))
        };
        if !relevant {
            continue;
        }
        // A missing required cut vanishes by the authenticated distributional
        // definition, independently of the Analyzer's scaleless certificates.
        if deltas
            .iter()
            .zip(sector)
            .any(|(&cut, active)| cut && !active)
        {
            zero.push(sector);
            continue;
        }
        let mask = Mask::try_new(sector).map_err(native_error)?;
        if matches!(
            analyzer.analyze(&mask).map_err(native_error)?,
            Decision::ProvedZero(_)
        ) {
            zero.push(sector);
        }
    }
    let zero: std::sync::Arc<[[bool; N]]> = zero.into();
    let mut result = Reduction::default();
    result.nonzero_conditions.extend(
        sources
            .conditions()
            .iter()
            .map(|c| substitute(&c.to_expression(), &family.reverse)),
    );
    result.nonzero_conditions.extend(
        analyzer
            .domain()
            .conditions()
            .iter()
            .map(|c| substitute(&c.polynomial().to_expression(), &family.reverse)),
    );
    let mut visited = BTreeSet::new();
    let legacy_key_for_depth = |depth| {
        let raw = blake3::hash(
            format!(
                "native-factorized-v5:{}:{:?}:{targets:?}:{}:{}:{}:{}:{}",
                family.family.fingerprint(),
                family.reverse,
                depth,
                options.include_lorentz,
                env!("RUSTRED_SOURCE_DIGEST"),
                env!("DEPENDENCY_SOURCE_DIGEST"),
                options.parametric_rules
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        provider_key(
            identity_seed_key(cut_stage_key(raw, cut_key.as_deref()), seed),
            provider,
        )
    };
    let stage_key_for_depth = |depth| {
        let legacy_key = legacy_key_for_depth(depth);
        let key = if options.bubble_subloops {
            blake3::hash(format!("bubble-subloops-v2:{legacy_key}").as_bytes())
                .to_hex()
                .to_string()
        } else {
            legacy_key
        };
        if options.symmetry_rules {
            blake3::hash(format!("symmetry-v1:{key}").as_bytes())
                .to_hex()
                .to_string()
        } else {
            key
        }
    };
    let legacy_key = legacy_key_for_depth(options.max_depth);
    let stage_key = stage_key_for_depth(options.max_depth);
    let family_key = provider_key(
        identity_seed_key(
            cut_stage_key(
                family_stage_key(family, original.physical_propagators, options),
                cut_key.as_deref(),
            ),
            seed,
        ),
        provider,
    );
    let mut patterns = BTreeMap::<[bool; N], Option<crate::bubble::Bubble>>::new();
    let mut restart = if let Some(directory) = &options.checkpoints {
        crate::cache::read_native_stage(directory, &stage_key)?
    } else {
        None
    };
    // Exact identities survive a larger search radius, but a previous search
    // of a residual is not a search at the new depth. Reuse the raw DAG and
    // search all surviving leaves again, including formerly visited leaves.
    if restart.is_none()
        && let Some(depth) = options.max_depth.checked_sub(1)
        && let Some(directory) = &options.checkpoints
        && let Some((old, done, _)) =
            crate::cache::read_native_stage(directory, &stage_key_for_depth(depth))?
    {
        let frontier = structural_frontier(&old, targets, context)?;
        let done = done
            .into_iter()
            .filter(|i| old.rules.contains_key(i))
            .collect();
        restart = Some((old, done, frontier.into_iter().collect()));
    }
    // A legacy partial search can be reused only when none of its searched
    // sectors changes ordering under bubble elimination. In particular, the
    // expensive higher-sector searches need not be repeated on migration.
    if restart.is_none()
        && options.bubble_subloops
        && !options.symmetry_rules
        && let Some(directory) = &options.checkpoints
        && let Some(old) = crate::cache::read_native_stage(directory, &legacy_key)?
    {
        let mut compatible = true;
        for integral in old.0.rules.keys().chain(&old.1) {
            let sector: [bool; N] = integral
                .0
                .iter()
                .map(|&n| n > 0)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| Error::Cache("checkpoint index arity".into()))?;
            if let std::collections::btree_map::Entry::Vacant(entry) = patterns.entry(sector) {
                entry.insert(crate::bubble::Bubble::find(
                    original,
                    &sector,
                    dimension.clone(),
                )?);
            }
            if patterns[&sector].is_some() {
                compatible = false;
                break;
            }
        }
        if compatible {
            restart = Some(old);
        }
    }
    if restart.is_none()
        && let Some(directory) = &options.checkpoints
        && let Some((bank, _, _)) = crate::cache::read_native_stage(directory, &family_key)?
    {
        let (bank, mut done, remaining) = restrict_family_stage(bank, targets, context)?;
        if provider.is_some() {
            // Family banks store reusable equations, not search history. In the
            // dynamic-provider lane their LHSs must not consume search budget.
            // Actual target checkpoints above retain their genuine visited set.
            done.clear();
        }
        restart = Some((bank, done, remaining));
    }
    if let Some((mut restored, done, remaining)) = restart {
        let array = |i: Integral| {
            i.0.as_slice()
                .try_into()
                .map_err(|_| Error::Cache("checkpoint index arity".into()))
        };
        visited = done
            .into_iter()
            .map(array)
            .collect::<Result<BTreeSet<[i16; N]>>>()?;
        pending = remaining
            .into_iter()
            .map(array)
            .collect::<Result<BTreeSet<[i16; N]>>>()?;
        restored
            .nonzero_conditions
            .append(&mut result.nonzero_conditions);
        result = restored;
    }
    let mut solvers = BTreeMap::new();
    let mut parametric_banks = BTreeMap::new();
    let mut symmetry_bank = if options.symmetry_rules {
        context.cancellation.check()?;
        let started = std::time::Instant::now();
        let bank = symmetry::Bank::discover(
            &family.family,
            original.physical_propagators,
            symmetry::Limits::default(),
        );
        context.emit(Progress::SymmetryReduction {
            candidates: bank.statistics.candidates,
            automorphisms: bank.statistics.automorphisms,
            applied: 0,
            uncovered: 0,
            transport_failures: 0,
            search_limited: bank.statistics.search_limited,
            elapsed_ms: started.elapsed().as_millis(),
        })?;
        Some(bank)
    } else {
        None
    };
    let pool = if options.native_workers > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(options.native_workers)
                .build()
                .map_err(|e| Error::Reduction(format!("native worker pool: {e}")))?,
        )
    } else {
        None
    };
    let mut last_saved = std::time::Instant::now();
    let mut completed_expansions = None;
    let mut frontier_cache = FrontierCache::default();
    if let Some(seed) = seed {
        admit_identity_seed::<N>(
            seed,
            original,
            dimension,
            options,
            context,
            &mut result,
            &mut frontier_cache,
            &mut visited,
            &mut pending,
            targets,
        )?;
    }
    let mut provider_calls = 0usize;
    let mut provider_checked = BTreeSet::new();
    let mut saved_before_exact = BTreeSet::new();
    let search: Result<()> = (|| {
        while !pending.is_empty() {
            for target in pending.iter().copied().collect::<Vec<_>>() {
                context.cancellation.check()?;
                if zero.contains(&target.map(|n| n > 0)) {
                    continue;
                }
                // Existing proof providers may close newly discovered internal leaves.
                // Unsupported/budget-exhausted requests continue through ordinary reduction.
                if let Some(provider) = provider
                    && provider_checked.insert(target)
                    && provider_calls < provider.max_calls()
                {
                    provider_calls += 1;
                    if let Some(bank) =
                        provider.provide(original, &Integral(target.to_vec()), context)?
                    {
                        context.cancellation.check()?;
                        let supplied = bank.resolve(original, None, context)?;
                        if supplied.reduction.rules.len() != 1
                            || !supplied
                                .reduction
                                .rules
                                .contains_key(&Integral(target.to_vec()))
                        {
                            return Err(Error::InvalidInput(
                                "provider must return exactly the requested identity".into(),
                            ));
                        }
                        validate_identity_order::<N>(
                            &supplied, original, dimension, options, context,
                        )?;
                        for terms in supplied.reduction.rules.values() {
                            for rhs in terms.keys() {
                                let powers: [i16; N] =
                                    rhs.0.as_slice().try_into().map_err(|_| {
                                        Error::InvalidInput("provider RHS arity".into())
                                    })?;
                                if !visited.contains(&powers) {
                                    pending.insert(powers);
                                }
                            }
                        }
                        result
                            .nonzero_conditions
                            .extend(supplied.reduction.nonzero_conditions);
                        frontier_cache.insert_rules(&mut result, supplied.reduction.rules);
                        pending.remove(&target);
                        // Supplied identities are not expensive target searches.
                        // The equation itself survives restart in the native codec.
                        save_if_due(
                            options,
                            &stage_key,
                            &mut result,
                            &visited,
                            &pending,
                            &mut last_saved,
                        )?;
                        continue;
                    }
                }
            }
            if pending.is_empty() {
                break;
            }
            if visited.len() + pending.len() > options.max_targets {
                return Err(Error::Limit(format!(
                    "Laporta RHS closure exceeds max_targets={}",
                    options.max_targets
                )));
            }
            // Finish higher sectors before introducing their descendants into
            // lower-sector searches. Otherwise direct rules in a higher sector
            // can keep generating increasingly complicated lower-sector targets
            // after those lower sectors have already been expanded.
            let active_lines = pending
                .iter()
                .map(|i| i.iter().filter(|&&n| n > 0).count())
                .max()
                .unwrap();
            frontier_cache.enter_sector(active_lines);
            let selected = pending
                .iter()
                .filter(|i| i.iter().filter(|&&n| n > 0).count() == active_lines)
                .copied()
                .collect::<Vec<_>>();
            let mut sectors = BTreeMap::<[bool; N], Vec<CoordinateCase<N>>>::new();
            for target in selected {
                context.cancellation.check()?;
                let sector = target.map(|n| n > 0);
                if zero.contains(&sector) {
                    frontier_cache
                        .insert_rules(&mut result, [(Integral(target.to_vec()), BTreeMap::new())]);
                    pending.remove(&target);
                    visited.insert(target);
                    save_if_due(
                        options,
                        &stage_key,
                        &mut result,
                        &visited,
                        &pending,
                        &mut last_saved,
                    )?;
                    continue;
                }
                if options.bubble_subloops {
                    if let std::collections::btree_map::Entry::Vacant(entry) =
                        patterns.entry(sector)
                    {
                        entry.insert(crate::bubble::Bubble::find(
                            original,
                            &sector,
                            dimension.clone(),
                        )?);
                    }
                    if let Some(pattern) = &patterns[&sector]
                        && let Some(terms) = pattern.reduce(&Integral(target.to_vec()))?
                    {
                        let order = rustred::solver::IntegralOrder::new(sector, [false; N])
                            .with_permutation(
                                pattern
                                    .permutation()
                                    .try_into()
                                    .expect("bubble permutation arity"),
                            )
                            .map_err(native_error)?;
                        let lhs =
                            rustred::solver::Integral::numeric(target).map_err(native_error)?;
                        let decreasing = terms.keys().all(|i| {
                            let Ok(powers) = <[i16; N]>::try_from(i.0.as_slice()) else {
                                return false;
                            };
                            let Ok(rhs) = rustred::solver::Integral::numeric(powers) else {
                                return false;
                            };
                            order.compare(&lhs, &rhs) == std::cmp::Ordering::Less
                        });
                        if decreasing {
                            for (i, c) in &terms {
                                let powers: [i16; N] =
                                    i.0.as_slice().try_into().expect("bubble target arity");
                                if !visited.contains(&powers) {
                                    pending.insert(powers);
                                }
                                let rational: RationalPolynomial<IntegerRing, u16> = c
                                    .try_to_rational_polynomial(&Q, &Z, None)
                                    .map_err(native_error)?;
                                if !rational.denominator.is_constant() {
                                    result
                                        .nonzero_conditions
                                        .push(rational.denominator.to_expression());
                                }
                            }
                            frontier_cache
                                .insert_rules(&mut result, [(Integral(target.to_vec()), terms)]);
                            pending.remove(&target);
                            visited.insert(target);
                            save_if_due(
                                options,
                                &stage_key,
                                &mut result,
                                &visited,
                                &pending,
                                &mut last_saved,
                            )?;
                            continue;
                        }
                    }
                }
                sectors
                    .entry(sector)
                    .or_default()
                    .push(CoordinateCase::new(target.map(Some)).map_err(native_error)?);
            }
            let mut jobs = Vec::new();
            for (sector, mut cases) in sectors {
                if let std::collections::btree_map::Entry::Vacant(entry) = solvers.entry(sector) {
                    entry.insert(
                        SectorSolver::new(
                            &sources,
                            sector,
                            SectorConfig {
                                deltas,
                                zero_sectors: zero.clone(),
                                permutation: patterns.get(&sector).and_then(|p| p.as_ref()).map(
                                    |p| {
                                        p.permutation()
                                            .try_into()
                                            .expect("bubble permutation arity")
                                    },
                                ),
                                numerical_exact_backend: NumericalExactBackend::SparseFactorized,
                                ..Default::default()
                            },
                        )
                        .map_err(native_error)?,
                    );
                }
                if let Some(bank) = &mut symmetry_bank {
                    let started = std::time::Instant::now();
                    let before = bank.statistics;
                    let solver = &solvers[&sector];
                    let mut uncovered = Vec::new();
                    for case in cases {
                        context.cancellation.check()?;
                        let target = std::array::from_fn(|i| case.integral().powers()[i].value());
                        let Some(terms) = bank.apply(&family.family, target, solver.ordering())
                        else {
                            uncovered.push(case);
                            continue;
                        };
                        let (batch, _) = convert_batch(
                            rustred::solver::NumericResult {
                                rules: vec![rustred::solver::RuleCandidate {
                                    case: case.into(),
                                    target: rustred::solver::Integral::numeric(target)
                                        .map_err(native_error)?,
                                    rhs: terms,
                                    sources: Vec::new(),
                                    stats: Default::default(),
                                }],
                                residuals: Vec::new(),
                                stats: Default::default(),
                            },
                            family,
                        )?;
                        for terms in batch.rules.values() {
                            for integral in terms.keys() {
                                let powers: [i16; N] =
                                    integral.0.as_slice().try_into().map_err(|_| {
                                        Error::Reduction("symmetry RHS arity".into())
                                    })?;
                                if !visited.contains(&powers) {
                                    pending.insert(powers);
                                }
                            }
                        }
                        frontier_cache.insert_rules(&mut result, batch.rules);
                        result.nonzero_conditions.extend(batch.nonzero_conditions);
                        pending.remove(&target);
                        visited.insert(target);
                        save_if_due(
                            options,
                            &stage_key,
                            &mut result,
                            &visited,
                            &pending,
                            &mut last_saved,
                        )?;
                    }
                    context.emit(Progress::SymmetryReduction {
                        candidates: 0,
                        automorphisms: 0,
                        applied: bank.statistics.applied - before.applied,
                        uncovered: bank.statistics.missed - before.missed,
                        transport_failures: bank.statistics.transport_failures
                            - before.transport_failures,
                        search_limited: bank.statistics.search_limited,
                        elapsed_ms: started.elapsed().as_millis(),
                    })?;
                    cases = uncovered;
                }
                if options.parametric_rules {
                    let started = std::time::Instant::now();
                    let solver = &solvers[&sector];
                    let bank = parametric_banks
                        .entry(sector)
                        .or_insert_with(|| parametric::Bank::new(&sources, solver));
                    let before = (
                        bank.generated,
                        bank.hits,
                        bank.missed,
                        bank.domains,
                        bank.domain_hits,
                    );
                    let mut uncovered = Vec::new();
                    for case in cases {
                        context.cancellation.check()?;
                        let target = std::array::from_fn(|i| case.integral().powers()[i].value());
                        let Some(applied) = bank
                            .try_reduce(target, &sources, solver, options.max_depth)
                            .map_err(native_error)?
                        else {
                            uncovered.push(case);
                            continue;
                        };
                        let (batch, _) = convert_batch(
                            rustred::solver::NumericResult {
                                rules: vec![rustred::solver::RuleCandidate {
                                    case: case.into(),
                                    target: rustred::solver::Integral::numeric(target)
                                        .map_err(native_error)?,
                                    rhs: applied.terms,
                                    sources: Vec::new(),
                                    stats: Default::default(),
                                }],
                                residuals: Vec::new(),
                                stats: Default::default(),
                            },
                            family,
                        )?;
                        for terms in batch.rules.values() {
                            for integral in terms.keys() {
                                let powers: [i16; N] =
                                    integral.0.as_slice().try_into().map_err(|_| {
                                        Error::Reduction("parametric RHS arity".into())
                                    })?;
                                if !visited.contains(&powers) {
                                    pending.insert(powers);
                                }
                            }
                        }
                        frontier_cache.insert_rules(&mut result, batch.rules);
                        result.nonzero_conditions.extend(batch.nonzero_conditions);
                        result.nonzero_conditions.extend(
                            applied
                                .conditions
                                .iter()
                                .map(|p| substitute(&p.to_expression(), &family.reverse)),
                        );
                        pending.remove(&target);
                        visited.insert(target);
                        save_if_due(
                            options,
                            &stage_key,
                            &mut result,
                            &visited,
                            &pending,
                            &mut last_saved,
                        )?;
                    }
                    context.emit(Progress::ParametricReduction {
                        rays: bank.generated - before.0,
                        applied: bank.hits - before.1,
                        uncovered: bank.missed - before.2,
                        domains: bank.domains - before.3,
                        domain_applied: bank.domain_hits - before.4,
                        elapsed_ms: started.elapsed().as_millis(),
                    })?;
                    cases = uncovered;
                }
                // Group nearby complexities so a very easy target does not
                // enlarge the exact elimination needed by every difficult target.
                let order = rustred::solver::IntegralOrder::new(sector, deltas);
                let order = if let Some(pattern) = patterns.get(&sector).and_then(|p| p.as_ref()) {
                    order
                        .with_permutation(
                            pattern
                                .permutation()
                                .try_into()
                                .expect("bubble permutation arity"),
                        )
                        .map_err(native_error)?
                } else {
                    order
                };
                cases.sort_by(|a, b| order.compare(&a.integral(), &b.integral()).reverse());
                for cases in cases.chunks(options.max_sector_batch) {
                    jobs.push((sector, cases.to_vec()));
                }
            }
            // Bounded waves retain at most native_workers completed batches.
            // Merge in stable order, preserving deterministic exact rules and
            // the same restart semantics as the sequential implementation.
            for wave in jobs.chunks(options.native_workers) {
                let solve = |(sector, cases): &([bool; N], Vec<CoordinateCase<N>>)| {
                    context.emit(Progress::SectorReduction {
                        active_lines: sector.iter().filter(|&&v| v).count(),
                        integrals: cases.len(),
                        visited: visited.len(),
                    })?;
                    // A progress callback may request cancellation itself.
                    context.cancellation.check()?;
                    solvers[sector]
                        .solve_numeric_cases(
                            cases.to_vec(),
                            SearchOptions {
                                max_depth: Some(options.max_depth),
                                ..Default::default()
                            },
                        )
                        .map_err(native_error)
                        .and_then(|solved| convert_batch(solved, family))
                };
                let solutions = if let Some(pool) = &pool {
                    pool.install(|| wave.par_iter().map(solve).collect::<Vec<_>>())
                } else {
                    wave.iter().map(solve).collect::<Vec<_>>()
                };
                for ((_, cases), solved) in wave.iter().zip(solutions) {
                    let solved = solved?;
                    let (batch, completed) = solved;
                    for terms in batch.rules.values() {
                        for integral in terms.keys() {
                            let powers: [i16; N] = integral
                                .0
                                .as_slice()
                                .try_into()
                                .map_err(|_| Error::Reduction("native batch arity".into()))?;
                            if !visited.contains(&powers) {
                                pending.insert(powers);
                            }
                        }
                    }
                    frontier_cache.insert_rules(&mut result, batch.rules);
                    result.residuals.extend(batch.residuals);
                    result.nonzero_conditions.extend(batch.nonzero_conditions);
                    // Only completed searches enter the visited set. A checkpoint
                    // taken mid-round must leave every other case pending.
                    for case in cases {
                        let powers = std::array::from_fn(|i| case.integral().powers()[i].value());
                        pending.remove(&powers);
                        visited.insert(powers);
                    }
                    context.emit(completed)?;
                    save_if_due(
                        options,
                        &stage_key,
                        &mut result,
                        &visited,
                        &pending,
                        &mut last_saved,
                    )?;
                }
            }
            // Back-substitute into the requested targets before searching the
            // next frontier. Exact cancellations can remove intermediate RHS
            // integrals; searching those cancelled terms is unnecessary.
            // Validate the full original graph before any zero pruning, so a
            // cancelled or scaleless branch cannot conceal a cycle.
            let mut structural = structural_frontier(&result, targets, context)?;
            prune_zero_leaves(
                &mut result,
                &mut frontier_cache,
                &mut structural,
                &zero,
                &mut visited,
                &mut pending,
                context,
            )?;
            context.emit(Progress::Substitution {
                rules: result.rules.len(),
            })?;
            let unsearched = |i: &Integral| {
                <[i16; N]>::try_from(i.0.as_slice()).is_ok_and(|v| !visited.contains(&v))
            };
            let current_count = structural
                .iter()
                .filter(|i| line_count(i) == active_lines && unsearched(i))
                .count();
            let lower_count = structural
                .iter()
                .filter(|i| line_count(i) < active_lines && unsearched(i))
                .count();
            let mut frontier;
            if current_count > options.max_exact_frontier {
                // Extra searches are safe; unsearched leaves are never returned as
                // masters. Avoid constructing thousands of coefficients of lower
                // integrals that will subsequently be eliminated anyway.
                frontier = structural
                    .into_iter()
                    .filter(|i| line_count(i) <= active_lines)
                    .collect::<BTreeSet<_>>();
            } else if current_count == 0 && lower_count > options.max_exact_frontier {
                frontier = structural
                    .into_iter()
                    .filter(|i| line_count(i) < active_lines)
                    .collect();
            } else {
                // Fresh derivative searches can reach a long replay before the
                // periodic save interval. Preserve this exact raw graph once
                // per active level before starting coefficient arithmetic.
                if options.checkpoints.is_some() && saved_before_exact.insert(active_lines) {
                    save_stage(options, &stage_key, &mut result, &visited, &pending)?;
                    last_saved = std::time::Instant::now();
                }
                let mut expander = Expander::new(&result, context)?
                    .with_max_backward_frontier(options.max_backward_frontier);
                frontier = BTreeSet::new();
                let roots = targets.iter().cloned().map(Integral).collect::<Vec<_>>();
                let filtered = if let Some(previous) = &frontier_cache.targets {
                    expander.expand_weighted(previous, active_lines)?
                } else {
                    expander.expand_many(&roots, active_lines)?
                };
                frontier.extend(filtered.values().flat_map(|terms| terms.keys()).cloned());
                let unfinished = frontier
                    .iter()
                    .any(|i| line_count(i) == active_lines && unsearched(i));
                if !unfinished {
                    // A filtered cache has discarded lower-sector coefficients.
                    // Recover those contributions from the original targets.
                    frontier_cache.targets = None;
                    if lower_count > options.max_exact_frontier {
                        frontier = structural
                            .into_iter()
                            .filter(|i| line_count(i) < active_lines)
                            .collect();
                    } else {
                        frontier.clear();
                        let full = if structural.iter().all(|i| line_count(i) >= active_lines) {
                            filtered
                        } else {
                            expander.expand_many(&roots, 0)?
                        };
                        frontier.extend(full.values().flat_map(|terms| terms.keys()).cloned());
                        if frontier.iter().all(|i| !unsearched(i)) {
                            completed_expansions = Some(full);
                        }
                    }
                } else {
                    // Preserve searched residuals too: their coefficients are
                    // needed even though they do not drive another search.
                    frontier_cache.targets = Some(filtered);
                }
            }
            pending = frontier
                .iter()
                .map(|i| {
                    i.0.as_slice()
                        .try_into()
                        .map_err(|_| Error::Reduction("native frontier arity".into()))
                })
                .collect::<Result<BTreeSet<[i16; N]>>>()?;
            pending.retain(|i| !visited.contains(i));
            context.emit(Progress::ReductionFrontier {
                searched: visited.len(),
                remaining: pending.len(),
            })?;
            result.residuals = frontier.into_iter().collect();
            if pending.is_empty() {
                save_stage(options, &stage_key, &mut result, &visited, &pending)?;
                last_saved = std::time::Instant::now();
            } else {
                save_if_due(
                    options,
                    &stage_key,
                    &mut result,
                    &visited,
                    &pending,
                    &mut last_saved,
                )?;
            }
        }
        Ok(())
    })();
    if let Err(error) = search {
        if !matches!(&error, Error::Io(_) | Error::Cache(_)) {
            save_stage(options, &stage_key, &mut result, &visited, &pending)?;
        }
        return Err(error);
    }
    result.residuals.sort();
    result.residuals.dedup();
    result.residuals.retain(|i| !result.rules.contains_key(i));
    result.nonzero_conditions.sort();
    result.nonzero_conditions.dedup();
    let mut flattened = BTreeMap::new();
    let mut residuals = BTreeSet::new();
    let roots = targets.iter().cloned().map(Integral).collect::<Vec<_>>();
    let expansions = if let Some(completed) = completed_expansions {
        completed
    } else {
        // A completed checkpoint can skip the search loop entirely. Apply the
        // same certificate-based pruning before its final fresh expansion.
        let mut structural = structural_frontier(&result, targets, context)?;
        prune_zero_leaves(
            &mut result,
            &mut frontier_cache,
            &mut structural,
            &zero,
            &mut visited,
            &mut pending,
            context,
        )?;
        Expander::new(&result, context)?
            .with_max_backward_frontier(options.max_backward_frontier)
            .expand_many(&roots, 0)?
    };
    for (integral, terms) in expansions {
        residuals.extend(terms.keys().cloned());
        if result.rules.contains_key(&integral) {
            flattened.insert(
                integral,
                terms
                    .iter()
                    .map(|(i, c)| (i.clone(), coefficient_atom(c)))
                    .collect(),
            );
        }
    }
    for i in &residuals {
        let powers: [i16; N] =
            i.0.as_slice()
                .try_into()
                .map_err(|_| Error::Reduction("native residual arity".into()))?;
        if !visited.contains(&powers) {
            return Err(Error::IncompleteReduction(
                "native reduction left an unsearched target contribution".into(),
            ));
        }
    }
    if let Some(directory) = &options.checkpoints {
        // The final exact check above authenticates this successful call. The
        // reusable bank stores raw equations; its leaf search history is not
        // reused. Atomic last-writer replacement by another valid subset is
        // safe, even when independent samples or requests share a directory.
        crate::cache::write_native_stage(
            directory,
            &family_key,
            &result,
            &result.rules.keys().cloned().collect::<Vec<_>>(),
            &[],
        )?;
    }
    result.rules = flattened;
    result.residuals = residuals.into_iter().collect();
    for target in targets {
        result.expand(&Integral(target.clone()))?;
    }
    Ok(result)
}
fn line_count(integral: &Integral) -> usize {
    integral.0.iter().filter(|&&n| n > 0).count()
}
fn structural_frontier(
    reduction: &Reduction,
    targets: &[Vec<i16>],
    context: &RunContext,
) -> Result<BTreeSet<Integral>> {
    let mut state = ahash::HashMap::default();
    let mut leaves = BTreeSet::new();
    let roots = targets.iter().cloned().map(Integral).collect::<Vec<_>>();
    let mut stack = roots.iter().map(|root| (root, false)).collect::<Vec<_>>();
    while let Some((node, finished)) = stack.pop() {
        context.cancellation.check()?;
        if finished {
            state.insert(node, 2);
            continue;
        }
        match state.get(node) {
            Some(2) => continue,
            Some(1) => {
                return Err(Error::IncompleteReduction("cyclic native reduction".into()));
            }
            _ => {}
        }
        // Validate the complete original DAG even when exact cancellation
        // removed a branch from a cached weighted frontier.
        state.insert(node, 1);
        stack.push((node, true));
        if let Some(terms) = reduction.rules.get(node) {
            let parent_lines = line_count(node);
            for child in terms.keys() {
                if line_count(child) > parent_lines {
                    return Err(Error::Reduction(
                        "native rewrite increases its sector; descending-sector search is invalid"
                            .into(),
                    ));
                }
                stack.push((child, false));
            }
        } else {
            leaves.insert(node.clone());
        }
    }
    Ok(leaves)
}

/// Remove structural leaves with an authenticated zero identity: an Analyzer
/// scaleless certificate or a nonpositive required cut. Family and certificate
/// domain conditions are already retained in the reduction.
/// Call only after validating the complete original dependency graph.
fn prune_zero_leaves<const N: usize>(
    reduction: &mut Reduction,
    cache: &mut FrontierCache,
    frontier: &mut BTreeSet<Integral>,
    zero: &[[bool; N]],
    visited: &mut BTreeSet<[i16; N]>,
    pending: &mut BTreeSet<[i16; N]>,
    context: &RunContext,
) -> Result<()> {
    let mut certified = Vec::new();
    for integral in frontier.iter() {
        context.cancellation.check()?;
        let powers: [i16; N] = integral
            .0
            .as_slice()
            .try_into()
            .map_err(|_| Error::Reduction("native zero frontier arity".into()))?;
        if zero.contains(&powers.map(|n| n > 0)) {
            certified.push((integral.clone(), powers));
        }
    }
    for (integral, powers) in certified {
        cache.insert_rules(reduction, [(integral.clone(), BTreeMap::new())]);
        visited.insert(powers);
        pending.remove(&powers);
        frontier.remove(&integral);
    }
    Ok(())
}
use std::sync::Arc;
use symbolica::domains::factorized_rational_polynomial::{
    FactorizedRationalPolynomial, FactorizedRationalPolynomialField,
    FromNumeratorAndFactorizedDenominator,
};
use symbolica::poly::PolyVariable;
type Coefficient = FactorizedRationalPolynomial<IntegerRing, u16>;
type Terms = BTreeMap<Integral, Coefficient>;

/// Exact expansions at one fixed sector threshold. A skipped exact-expansion
/// round may append rules without invalidating these weighted seeds. Replacing
/// an already used identity, however, can change the unresolved leaf basis.
#[derive(Default)]
struct FrontierCache {
    minimum_lines: usize,
    targets: Option<BTreeMap<Integral, Terms>>,
}
impl FrontierCache {
    fn enter_sector(&mut self, minimum_lines: usize) {
        if self.minimum_lines != minimum_lines {
            self.targets = None;
            self.minimum_lines = minimum_lines;
        }
    }
    fn insert_rules(
        &mut self,
        reduction: &mut Reduction,
        rules: impl IntoIterator<Item = (Integral, LinearCombination)>,
    ) {
        for (integral, terms) in rules {
            match reduction.rules.entry(integral) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(terms);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get() != &terms {
                        self.targets = None;
                        entry.insert(terms);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct CoefficientSum {
    bins: Vec<Option<Coefficient>>,
}
impl CoefficientSum {
    fn push(
        &mut self,
        mut value: Coefficient,
        field: &FactorizedRationalPolynomialField<IntegerRing, u16>,
    ) {
        // A binary carry tree adds comparable numbers of contributions. This
        // avoids repeatedly expanding a large running denominator for each
        // small contribution, and keeps only logarithmically many partial sums.
        for bin in &mut self.bins {
            if value.is_zero() {
                return;
            }
            if let Some(previous) = bin.take() {
                value = field.add(&previous, &value);
            } else {
                *bin = Some(value);
                return;
            }
        }
        if !value.is_zero() {
            self.bins.push(Some(value));
        }
    }
    fn finish(self, field: &FactorizedRationalPolynomialField<IntegerRing, u16>) -> Coefficient {
        self.bins
            .into_iter()
            .flatten()
            .fold(field.zero(), |sum, term| field.add(&sum, &term))
    }
}
struct Expander<'a> {
    reduction: &'a Reduction,
    context: &'a RunContext,
    variables: Arc<Vec<PolyVariable>>,
    field: FactorizedRationalPolynomialField<IntegerRing, u16>,
    coefficients: ahash::HashMap<Atom, Arc<Coefficient>>,
    max_backward_frontier: usize,
}
impl<'a> Expander<'a> {
    fn new(reduction: &'a Reduction, context: &'a RunContext) -> Result<Self> {
        let mut symbols = BTreeSet::new();
        for coefficient in reduction.rules.values().flat_map(|r| r.values()) {
            crate::family::scalar_symbols(coefficient.as_view(), &mut symbols)?;
        }
        let variables: Arc<Vec<PolyVariable>> = Arc::new(
            symbols
                .into_iter()
                .map(|a| match a.as_view() {
                    AtomView::Var(v) => PolyVariable::Symbol(v.get_symbol()),
                    _ => unreachable!(),
                })
                .collect(),
        );
        Ok(Self {
            reduction,
            context,
            field: FactorizedRationalPolynomialField::new(Z, variables.clone()),
            variables,
            coefficients: Default::default(),
            max_backward_frontier: 128,
        })
    }
    fn with_max_backward_frontier(mut self, maximum: usize) -> Self {
        self.max_backward_frontier = maximum;
        self
    }
    fn coefficient(&mut self, a: &Atom) -> Result<Arc<Coefficient>> {
        if let Some(c) = self.coefficients.get(a) {
            return Ok(c.clone());
        }
        let polynomial: RationalPolynomial<IntegerRing, u16> = a
            .try_to_rational_polynomial(&Q, &Z, Some(self.variables.clone()))
            .map_err(|e| Error::Reduction(e.to_string()))?;
        let c = Arc::new(Coefficient::from_num_den(
            polynomial.numerator,
            vec![(polynomial.denominator, 1)],
            &Z,
            true,
        ));
        self.coefficients.insert(a.clone(), c.clone());
        Ok(c)
    }
    #[cfg(test)]
    fn expand(&mut self, integral: &Integral) -> Result<Terms> {
        self.expand_filtered(integral, 0)
    }
    #[cfg(test)]
    fn expand_filtered(&mut self, integral: &Integral, minimum_lines: usize) -> Result<Terms> {
        Ok(self
            .expand_many(std::slice::from_ref(integral), minimum_lines)?
            .remove(integral)
            .unwrap_or_default())
    }
    fn expand_many(
        &mut self,
        integrals: &[Integral],
        minimum_lines: usize,
    ) -> Result<BTreeMap<Integral, Terms>> {
        let (order, retained) = self.plan(integrals, minimum_lines)?;
        let terminals = retained
            .iter()
            .filter(|node| !self.reduction.rules.contains_key(*node))
            .count();
        self.context.emit(Progress::SubstitutionPlan {
            roots: integrals.len(),
            retained_nodes: retained.len(),
            terminals,
            minimum_lines,
            backward: terminals <= self.max_backward_frontier,
            weighted: false,
        })?;
        if terminals <= self.max_backward_frontier {
            return self.expand_backward(integrals, &order, &retained);
        }
        integrals
            .iter()
            .map(|integral| {
                Ok((
                    integral.clone(),
                    self.expand_forward(
                        &BTreeMap::from([(integral.clone(), self.field.one())]),
                        &order,
                        &retained,
                    )?,
                ))
            })
            .collect()
    }
    fn plan(
        &self,
        integrals: &[Integral],
        minimum_lines: usize,
    ) -> Result<(Vec<Integral>, BTreeSet<Integral>)> {
        // Accumulate coefficients in dependency order. Expanding every child
        // into its own complete residual map stores a quadratic number of
        // large rational functions on realistic Laporta DAGs. Forward
        // substitution needs only one coefficient per live integral.
        let mut state = BTreeMap::<Integral, u8>::new();
        let mut stack = integrals
            .iter()
            .cloned()
            .map(|i| (i, false))
            .collect::<Vec<_>>();
        let mut order = Vec::new();
        while let Some((node, finish)) = stack.pop() {
            self.context.cancellation.check()?;
            if finish {
                state.insert(node.clone(), 2);
                order.push(node);
                continue;
            }
            match state.get(&node) {
                Some(2) => continue,
                Some(1) => {
                    return Err(Error::IncompleteReduction("cyclic native reduction".into()));
                }
                _ => {}
            }
            state.insert(node.clone(), 1);
            stack.push((node.clone(), true));
            if let Some(terms) = self.reduction.rules.get(&node) {
                stack.extend(terms.keys().cloned().map(|child| (child, false)));
            }
        }
        // Retain exactly the dependency paths reaching requested terminal
        // sectors. This graph test is valid even for a caller's nontriangular
        // DAG; it does not assume that every rewrite lowers the sector.
        let mut retained = BTreeSet::new();
        for node in &order {
            let keep = if let Some(terms) = self.reduction.rules.get(node) {
                terms.keys().any(|child| retained.contains(child))
            } else {
                node.0.iter().filter(|&&n| n > 0).count() >= minimum_lines
            };
            if keep {
                retained.insert(node.clone());
            }
        }
        Ok((order, retained))
    }

    /// Apply newly available identities to an exact weighted frontier. Native
    /// structural validation guarantees that a discarded lower sector cannot
    /// feed back into the selected sectors. Callers reset seeds when the sector
    /// threshold changes or an existing identity is replaced.
    fn expand_weighted(
        &mut self,
        inputs: &BTreeMap<Integral, Terms>,
        minimum_lines: usize,
    ) -> Result<BTreeMap<Integral, Terms>> {
        let roots = inputs
            .values()
            .flat_map(|terms| terms.keys().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let (order, retained) = self.plan(&roots, minimum_lines)?;
        let terminals = retained
            .iter()
            .filter(|node| !self.reduction.rules.contains_key(*node))
            .count();
        self.context.emit(Progress::SubstitutionPlan {
            roots: roots.len(),
            retained_nodes: retained.len(),
            terminals,
            minimum_lines,
            backward: terminals <= self.max_backward_frontier,
            weighted: true,
        })?;
        if terminals > self.max_backward_frontier {
            return inputs
                .iter()
                .map(|(target, seeds)| {
                    Ok((
                        target.clone(),
                        self.expand_forward(seeds, &order, &retained)?,
                    ))
                })
                .collect();
        }
        let expanded = self.expand_backward(&roots, &order, &retained)?;
        inputs
            .iter()
            .map(|(target, seeds)| {
                let mut sums = BTreeMap::<Integral, CoefficientSum>::new();
                for (seed, coefficient) in seeds {
                    self.context.cancellation.check()?;
                    for (terminal, value) in &expanded[seed] {
                        // Public field arithmetic unifies variable maps if a
                        // later rule introduces a previously absent parameter.
                        let product = self.field.mul(coefficient, value);
                        if !product.is_zero() {
                            sums.entry(terminal.clone())
                                .or_default()
                                .push(product, &self.field);
                        }
                    }
                }
                Ok((
                    target.clone(),
                    sums.into_iter()
                        .filter_map(|(terminal, sum)| {
                            let value = sum.finish(&self.field);
                            (!value.is_zero()).then_some((terminal, value))
                        })
                        .collect(),
                ))
            })
            .collect()
    }
    fn expand_backward(
        &mut self,
        integrals: &[Integral],
        order: &[Integral],
        retained: &BTreeSet<Integral>,
    ) -> Result<BTreeMap<Integral, Terms>> {
        // With a small terminal basis, reducing each identity before using it
        // in its parents cancels apparent poles locally. Release a child's
        // expansion after its last parent, bounding the live dependency maps.
        let mut uses = ahash::HashMap::<&Integral, usize>::default();
        for integral in integrals {
            // A target can also be a dependency of another target. Preserve
            // its expansion until all requested outputs have been collected.
            *uses.entry(integral).or_default() += 1;
        }
        for node in order.iter().filter(|node| retained.contains(*node)) {
            if let Some(terms) = self.reduction.rules.get(node) {
                for child in terms.keys().filter(|child| retained.contains(*child)) {
                    *uses.entry(child).or_default() += 1;
                }
            }
        }
        let mut expanded = ahash::HashMap::<Integral, Terms>::default();
        for node in order.iter().filter(|node| retained.contains(*node)) {
            self.context.cancellation.check()?;
            let value = if let Some(terms) = self.reduction.rules.get(node) {
                let mut sums = BTreeMap::<Integral, CoefficientSum>::new();
                for (child, coefficient) in terms {
                    if !retained.contains(child) {
                        continue;
                    }
                    let coefficient = self.coefficient(coefficient)?;
                    for (terminal, value) in &expanded[child] {
                        let product = self.field.mul(coefficient.as_ref(), value);
                        if !product.is_zero() {
                            sums.entry(terminal.clone())
                                .or_default()
                                .push(product, &self.field);
                        }
                    }
                    let remaining = uses.get_mut(child).expect("retained dependency use");
                    *remaining -= 1;
                    if *remaining == 0 {
                        expanded.remove(child);
                    }
                }
                sums.into_iter()
                    .filter_map(|(terminal, sum)| {
                        let value = sum.finish(&self.field);
                        (!value.is_zero()).then_some((terminal, value))
                    })
                    .collect()
            } else {
                BTreeMap::from([(node.clone(), self.field.one())])
            };
            expanded.insert(node.clone(), value);
        }
        Ok(integrals
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|integral| {
                (
                    integral.clone(),
                    expanded.remove(integral).unwrap_or_default(),
                )
            })
            .collect())
    }
    fn expand_forward(
        &mut self,
        seeds: &Terms,
        order: &[Integral],
        retained: &BTreeSet<Integral>,
    ) -> Result<Terms> {
        let mut out = BTreeMap::<Integral, CoefficientSum>::new();
        for (integral, coefficient) in seeds {
            if retained.contains(integral) {
                out.entry(integral.clone())
                    .or_default()
                    .push(coefficient.clone(), &self.field);
            }
        }
        for node in order.iter().rev() {
            self.context.cancellation.check()?;
            let Some(terms) = self.reduction.rules.get(node) else {
                continue;
            };
            let Some(value) = out.remove(node) else {
                continue;
            };
            let value = value.finish(&self.field);
            if value.is_zero() {
                continue;
            }
            for (child, coefficient) in terms {
                if !retained.contains(child) {
                    continue;
                }
                let coefficient = self.coefficient(coefficient)?;
                let product = self.field.mul(&value, coefficient.as_ref());
                if !product.is_zero() {
                    out.entry(child.clone())
                        .or_default()
                        .push(product, &self.field);
                }
            }
        }
        Ok(out
            .into_iter()
            .filter_map(|(integral, terms)| {
                let coefficient = terms.finish(&self.field);
                (!coefficient.is_zero()).then_some((integral, coefficient))
            })
            .collect())
    }
}
fn coefficient_atom(c: &Coefficient) -> Atom {
    let mut a = c.numerator.to_expression() * Atom::num(c.numer_coeff.clone())
        / Atom::num(c.denom_coeff.clone());
    for (denominator, power) in &c.denominators {
        a *= denominator.to_expression().pow(-(*power as i64));
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_bank_prunes_unreachable_equations_preserves_guards_and_validates_the_dag() {
        let i = |n| Integral(vec![n, 0]);
        let condition = parse!("family_bank_guard::x");
        let reduction = Reduction {
            rules: BTreeMap::from([
                (i(5), BTreeMap::from([(i(4), condition.clone())])),
                (i(4), BTreeMap::from([(i(1), Atom::num(1))])),
                (i(6), BTreeMap::new()),
                (i(8), BTreeMap::from([(i(7), Atom::num(1))])),
            ]),
            residuals: vec![i(1), i(7)],
            nonzero_conditions: vec![condition.clone()],
        };
        let context = RunContext::default();
        let roots = [i(5).0, i(6).0];
        let (mut retained, done, pending) =
            restrict_family_stage(reduction, &roots, &context).unwrap();
        assert_eq!(
            retained.rules.keys().cloned().collect::<Vec<_>>(),
            vec![i(4), i(5), i(6)]
        );
        assert_eq!(done, vec![i(4), i(5), i(6)]);
        assert_eq!(pending, vec![i(1)]);
        assert_eq!(retained.residuals, pending);
        assert_eq!(retained.nonzero_conditions, vec![condition]);
        retained
            .rules
            .insert(i(1), BTreeMap::from([(i(5), Atom::num(1))]));
        assert!(matches!(
            restrict_family_stage(retained.clone(), &roots, &context),
            Err(Error::IncompleteReduction(_))
        ));
        retained
            .rules
            .insert(i(1), BTreeMap::from([(Integral(vec![1, 1]), Atom::num(1))]));
        assert!(matches!(
            restrict_family_stage(retained, &roots, &context),
            Err(Error::Reduction(_))
        ));
    }

    #[test]
    fn substitution_plans_report_actual_fresh_and_weighted_graphs() {
        let i = |n| Integral(vec![n]);
        let reduction = Reduction {
            rules: BTreeMap::from([(
                i(3),
                BTreeMap::from([(i(1), Atom::num(1)), (i(2), Atom::num(2))]),
            )]),
            ..Default::default()
        };
        let plans = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = Arc::clone(&plans);
        let context = RunContext {
            progress: Some(Arc::new(move |event| {
                if let Progress::SubstitutionPlan {
                    roots,
                    retained_nodes,
                    terminals,
                    minimum_lines,
                    backward,
                    weighted,
                } = event
                {
                    observed.lock().unwrap().push((
                        roots,
                        retained_nodes,
                        terminals,
                        minimum_lines,
                        backward,
                        weighted,
                    ));
                }
            })),
            ..Default::default()
        };
        for maximum in [0, 2] {
            let mut expander = Expander::new(&reduction, &context)
                .unwrap()
                .with_max_backward_frontier(maximum);
            let fresh = expander.expand_many(&[i(3)], 1).unwrap();
            let weighted = expander.expand_weighted(&fresh, 1).unwrap();
            assert_expansions_equal(&fresh, &weighted);
        }
        assert_eq!(
            *plans.lock().unwrap(),
            vec![
                (1, 3, 2, 1, false, false),
                (2, 2, 2, 1, false, true),
                (1, 3, 2, 1, true, false),
                (2, 2, 2, 1, true, true),
            ]
        );
    }

    fn assert_expansions_equal(
        left: &BTreeMap<Integral, Terms>,
        right: &BTreeMap<Integral, Terms>,
    ) {
        assert_eq!(
            left.keys().collect::<Vec<_>>(),
            right.keys().collect::<Vec<_>>()
        );
        for (root, terms) in left {
            let other = &right[root];
            assert_eq!(
                terms.keys().collect::<Vec<_>>(),
                other.keys().collect::<Vec<_>>()
            );
            for (integral, coefficient) in terms {
                assert!(
                    (coefficient - &other[integral]).is_zero(),
                    "different expansion of {root:?} into {integral:?}"
                );
            }
        }
    }

    #[test]
    fn eager_zero_leaves_use_analyzer_certificates_and_preserve_live_coefficients() {
        let s = parse!("certified_zero::s");
        let gram = vec![vec![s.clone()]];
        let family = IntegralFamily {
            name: "certified_zero_bubble".into(),
            loops: vec!["l".into()],
            external: vec!["p".into()],
            external_gram: gram.clone(),
            propagators: vec![
                crate::Propagator::quadratic(&[1], &[0], Atom::new(), &gram).unwrap(),
                crate::Propagator::quadratic(&[1], &[1], Atom::new(), &gram).unwrap(),
            ],
            physical_propagators: 2,
            epsilon: symbol!("certified_zero::eps"),
            dimension: 4,
        };
        let converted = family.convert().unwrap();
        let analyzer = Analyzer::try_unrestricted(&converted.family).unwrap();
        let zero = (0..4)
            .filter_map(|bits| {
                let sector = std::array::from_fn::<_, 2, _>(|i| bits & (1 << i) != 0);
                matches!(
                    analyzer.analyze(&Mask::try_new(sector).unwrap()).unwrap(),
                    Decision::ProvedZero(_)
                )
                .then_some(sector)
            })
            .collect::<Vec<_>>();
        assert!(zero.contains(&[false, false]));
        assert!(zero.contains(&[true, false]));
        assert!(zero.contains(&[false, true]));
        assert!(!zero.contains(&[true, true]));

        let root = Integral(vec![2, 2]);
        let zero_root = Integral(vec![2, 1]);
        let master = Integral(vec![1, 1]);
        let zero_powers = [[2, 0], [1, -2], [0, 3], [0, 0]];
        let coefficient = Atom::num(1) / (&s + Atom::num(1));
        let conditions = analyzer
            .domain()
            .conditions()
            .iter()
            .map(|condition| {
                substitute(&condition.polynomial().to_expression(), &converted.reverse)
            })
            .chain([&s + Atom::num(1)])
            .collect::<Vec<_>>();
        let mut terms = BTreeMap::from([(master.clone(), coefficient.clone())]);
        for (n, powers) in zero_powers.iter().enumerate() {
            terms.insert(
                Integral(powers.to_vec()),
                (&s + Atom::num(n + 2)) / (&s + Atom::num(n + 3)),
            );
        }
        let mut reduction = Reduction {
            rules: BTreeMap::from([
                (root.clone(), terms),
                (
                    zero_root.clone(),
                    BTreeMap::from([(Integral(zero_powers[0].to_vec()), Atom::num(1))]),
                ),
            ]),
            nonzero_conditions: conditions.clone(),
            ..Default::default()
        };
        let roots = [root.clone(), zero_root.clone()];
        let powers = roots.iter().map(|i| i.0.clone()).collect::<Vec<_>>();
        let context = RunContext::default();
        let mut structural = structural_frontier(&reduction, &powers, &context).unwrap();
        let mut pending = structural
            .iter()
            .map(|i| <[i16; 2]>::try_from(i.0.as_slice()).unwrap())
            .collect();
        let mut visited = BTreeSet::from([zero_powers[0]]);
        let mut cache = FrontierCache {
            targets: Some(
                Expander::new(&reduction, &context)
                    .unwrap()
                    .expand_many(&roots, 0)
                    .unwrap(),
            ),
            ..Default::default()
        };
        assert_eq!(structural.len(), 5);
        prune_zero_leaves(
            &mut reduction,
            &mut cache,
            &mut structural,
            &zero,
            &mut visited,
            &mut pending,
            &context,
        )
        .unwrap();
        assert_eq!(structural, BTreeSet::from([master.clone()]));
        assert_eq!(pending, BTreeSet::from([[1, 1]]));
        assert_eq!(visited, BTreeSet::from(zero_powers));
        assert_eq!(reduction.nonzero_conditions, conditions);
        for powers in zero_powers {
            assert!(reduction.rules[&Integral(powers.to_vec())].is_empty());
        }
        assert!(cache.targets.is_some());
        let mut expander = Expander::new(&reduction, &context).unwrap();
        let updated = expander
            .expand_weighted(cache.targets.as_ref().unwrap(), 0)
            .unwrap();
        let fresh = expander.expand_many(&roots, 0).unwrap();
        assert_expansions_equal(&updated, &fresh);
        assert!(updated[&zero_root].is_empty());
        assert_eq!(updated[&root].len(), 1);
        assert!(
            (coefficient_atom(&updated[&root][&master]) - coefficient)
                .together()
                .cancel()
                .is_zero()
        );
        // A certified-zero sector must not hide a malformed cyclic DAG.
        let zero_integral = Integral(zero_powers[0].to_vec());
        reduction.rules.insert(
            zero_integral.clone(),
            BTreeMap::from([(zero_integral, Atom::num(1))]),
        );
        assert!(matches!(
            structural_frontier(&reduction, &powers, &context),
            Err(Error::IncompleteReduction(_))
        ));
    }

    #[test]
    fn incremental_frontiers_match_fresh_expansion_across_partial_rounds() {
        let i = |n| Integral(vec![n, 1]);
        let low = Integral(vec![1, 0]);
        let context = RunContext::default();
        let roots = vec![i(9), i(8), i(7)];
        let mut reduction = Reduction {
            residuals: vec![low.clone()],
            ..Default::default()
        };
        let mut cache = FrontierCache::default();
        cache.enter_sector(2);
        cache.insert_rules(
            &mut reduction,
            [
                (
                    i(9),
                    BTreeMap::from([
                        (i(6), parse!("frontier::x")),
                        (i(5), Atom::num(1)),
                        (low.clone(), parse!("1/(frontier::x+1)")),
                    ]),
                ),
                (
                    i(8),
                    BTreeMap::from([(i(6), Atom::num(1)), (i(5), Atom::num(-1))]),
                ),
                (i(7), BTreeMap::new()),
            ],
        );
        cache.targets = Some(
            Expander::new(&reduction, &context)
                .unwrap()
                .expand_many(&roots, 2)
                .unwrap(),
        );
        // A structural-only round may skip exact expansion. Its earlier
        // frontier must remain valid when several later rules arrive at once.
        let skipped_round = cache.targets.clone().unwrap();
        for round in 1..=3 {
            let added = match round {
                1 => vec![
                    (
                        i(6),
                        BTreeMap::from([
                            (i(4), parse!("1/(frontier::x+1)")),
                            (low.clone(), Atom::num(3)),
                        ]),
                    ),
                    (
                        i(5),
                        BTreeMap::from([
                            (i(4), parse!("-frontier::x/(frontier::x+1)")),
                            (i(3), parse!("frontier::y")),
                        ]),
                    ),
                ],
                2 => vec![
                    (i(3), BTreeMap::new()),
                    (
                        i(4),
                        BTreeMap::from([
                            (i(2), parse!("frontier::z/(frontier::y+1)")),
                            (low.clone(), Atom::num(5)),
                        ]),
                    ),
                ],
                _ => vec![(i(2), BTreeMap::from([(low.clone(), Atom::num(7))]))],
            };
            cache.insert_rules(&mut reduction, added);
            let mut expander = Expander::new(&reduction, &context).unwrap();
            let fresh = expander.expand_many(&roots, 2).unwrap();
            let updated = expander
                .expand_weighted(cache.targets.as_ref().unwrap(), 2)
                .unwrap();
            assert_expansions_equal(&updated, &fresh);
            assert!(updated[&i(7)].is_empty());
            if round == 1 {
                // The x-dependent paths cancel, but a searched same-sector
                // residual still carries its coefficient into later rounds.
                assert!(!updated[&i(9)].contains_key(&i(4)));
                assert_eq!(
                    coefficient_atom(&updated[&i(9)][&i(3)]),
                    parse!("frontier::y")
                );
            }
            if round == 2 {
                assert!(updated[&i(9)].is_empty());
                let skipped = expander.expand_weighted(&skipped_round, 2).unwrap();
                assert_expansions_equal(&skipped, &fresh);
            }
            cache.targets = Some(updated);
        }
        assert!(
            cache
                .targets
                .as_ref()
                .unwrap()
                .values()
                .all(|r| r.is_empty())
        );
        cache.enter_sector(1);
        assert!(cache.targets.is_none());
        // The same original target has nonzero lower-sector coefficients;
        // entering that sector must recover them from the original roots.
        let full = Expander::new(&reduction, &context)
            .unwrap()
            .expand_many(&roots, 0)
            .unwrap();
        assert!(full[&i(9)].contains_key(&low));
        for root in roots {
            let independent = reduction.expand(&root).unwrap();
            assert_eq!(
                full[&root].keys().collect::<Vec<_>>(),
                independent.keys().collect::<Vec<_>>()
            );
            for (leaf, coefficient) in &full[&root] {
                assert!(
                    (coefficient_atom(coefficient) - &independent[leaf])
                        .together()
                        .cancel()
                        .is_zero()
                );
            }
        }
    }

    #[test]
    fn frontier_cache_invalidates_replaced_rules_and_sector_changes() {
        let root = Integral(vec![3, 1]);
        let leaf = Integral(vec![2, 1]);
        let context = RunContext::default();
        let mut reduction = Reduction::default();
        let mut cache = FrontierCache::default();
        cache.enter_sector(2);
        let rule = BTreeMap::from([(leaf.clone(), parse!("frontier_replace::x"))]);
        cache.insert_rules(&mut reduction, [(root.clone(), rule.clone())]);
        cache.targets = Some(
            Expander::new(&reduction, &context)
                .unwrap()
                .expand_many(std::slice::from_ref(&root), 2)
                .unwrap(),
        );
        cache.enter_sector(2);
        cache.insert_rules(&mut reduction, [(root.clone(), rule)]);
        assert!(cache.targets.is_some());
        cache.insert_rules(&mut reduction, [(Integral(vec![1, 0]), BTreeMap::new())]);
        assert!(cache.targets.is_some());
        cache.insert_rules(
            &mut reduction,
            [(
                root.clone(),
                BTreeMap::from([(leaf.clone(), parse!("2*frontier_replace::x"))]),
            )],
        );
        assert!(cache.targets.is_none());
        let fresh = Expander::new(&reduction, &context)
            .unwrap()
            .expand_many(std::slice::from_ref(&root), 2)
            .unwrap();
        assert_eq!(
            coefficient_atom(&fresh[&root][&leaf]),
            parse!("2*frontier_replace::x")
        );
        cache.targets = Some(fresh);
        cache.enter_sector(1);
        assert!(cache.targets.is_none());
    }

    #[test]
    fn structural_validation_rejects_cycles_in_previously_cancelled_branches() {
        let i = |n| Integral(vec![n, 1]);
        let roots = [i(6), i(5)];
        let context = RunContext::default();
        let mut reduction = Reduction {
            rules: BTreeMap::from([
                (i(6), BTreeMap::from([(i(1), Atom::num(1))])),
                (
                    i(5),
                    BTreeMap::from([(i(4), Atom::num(1)), (i(3), Atom::num(-1))]),
                ),
                (i(4), BTreeMap::from([(i(2), Atom::num(1))])),
                (i(3), BTreeMap::from([(i(2), Atom::num(1))])),
            ]),
            ..Default::default()
        };
        let powers = roots.iter().map(|i| i.0.clone()).collect::<Vec<_>>();
        assert_eq!(
            structural_frontier(&reduction, &powers, &context).unwrap(),
            BTreeSet::from([i(1), i(2)])
        );
        let cached = Expander::new(&reduction, &context)
            .unwrap()
            .expand_many(&roots, 2)
            .unwrap();
        assert!(cached[&i(5)].is_empty());
        assert!(cached[&i(6)].contains_key(&i(1)));
        // The surviving weighted seeds cannot see this new cycle. The full
        // structural validation performed before cache reuse must reject it.
        reduction
            .rules
            .insert(i(2), BTreeMap::from([(i(5), Atom::num(1))]));
        assert!(matches!(
            structural_frontier(&reduction, &powers, &context),
            Err(Error::IncompleteReduction(_))
        ));
        assert!(matches!(
            Expander::new(&reduction, &context)
                .unwrap()
                .expand_many(&roots, 2),
            Err(Error::IncompleteReduction(_))
        ));
    }

    #[test]
    fn weighted_wide_frontiers_cancel_exactly_and_keep_zero_targets() {
        let i = |n| Integral(vec![n, 1]);
        let x = parse!("weighted_wide::x");
        let context = RunContext::default();
        let mut reduction = Reduction::default();
        reduction.rules.insert(
            i(900),
            BTreeMap::from([(i(800), x.clone()), (i(700), Atom::num(-1))]),
        );
        reduction
            .rules
            .insert(i(901), BTreeMap::from([(i(800), Atom::num(1))]));
        reduction.rules.insert(i(902), BTreeMap::new());
        let roots = [i(900), i(901), i(902)];
        let seeds = Expander::new(&reduction, &context)
            .unwrap()
            .expand_many(&roots, 2)
            .unwrap();
        reduction.rules.insert(
            i(800),
            (1..=129)
                .map(|n| (i(n), Atom::num(n) / (&x + Atom::num(n))))
                .collect(),
        );
        reduction.rules.insert(
            i(700),
            (1..=129)
                .map(|n| {
                    (
                        i(n),
                        Atom::num(n) * &x / (&x + Atom::num(n)) - Atom::num(i64::from(n % 2 == 0)),
                    )
                })
                .collect(),
        );
        let mut reference = None;
        for maximum in [0, 128, 129] {
            // The same 129 leaves exercise forced forward, the unchanged
            // default, and backward substitution at the inclusive boundary.
            let mut expander = Expander::new(&reduction, &context)
                .unwrap()
                .with_max_backward_frontier(maximum);
            let weighted = expander.expand_weighted(&seeds, 2).unwrap();
            let fresh = expander.expand_many(&roots, 2).unwrap();
            assert_expansions_equal(&weighted, &fresh);
            if let Some(reference) = &reference {
                assert_expansions_equal(&weighted, reference);
            } else {
                reference = Some(weighted.clone());
            }
            assert_eq!(weighted[&i(900)].len(), 64);
            assert!(
                weighted[&i(900)]
                    .values()
                    .all(|v| coefficient_atom(v) == Atom::num(1))
            );
            assert_eq!(weighted[&i(901)].len(), 129);
            assert!(weighted[&i(902)].is_empty());
        }
        // Cached seeds use the same cycle-checked dependency planner as a
        // fresh expansion, including cycles introduced by subsequent rules.
        reduction
            .rules
            .insert(i(1), BTreeMap::from([(i(900), Atom::num(1))]));
        for maximum in [0, 129] {
            assert!(matches!(
                Expander::new(&reduction, &context)
                    .unwrap()
                    .with_max_backward_frontier(maximum)
                    .expand_weighted(&seeds, 2),
                Err(Error::IncompleteReduction(_))
            ));
        }
    }

    #[test]
    fn batched_substitution_retains_shared_targets_and_zero_rules() {
        let i = |n| Integral(vec![n]);
        let reduction = Reduction {
            rules: BTreeMap::from([
                (
                    i(1),
                    BTreeMap::from([(i(2), parse!("x")), (i(3), parse!("-x"))]),
                ),
                (
                    i(2),
                    BTreeMap::from([(i(3), Atom::num(1)), (i(4), parse!("1/(x+1)"))]),
                ),
                (i(3), BTreeMap::from([(i(5), parse!("x+1"))])),
                (i(4), BTreeMap::new()),
            ]),
            residuals: vec![i(5)],
            ..Default::default()
        };
        let context = RunContext::default();
        let targets = vec![i(1), i(2), i(3), i(4), i(5), i(2)];
        let actual = Expander::new(&reduction, &context)
            .unwrap()
            .expand_many(&targets, 0)
            .unwrap();
        assert_eq!(actual.len(), 5);
        for target in targets {
            let terms: BTreeMap<_, _> = actual[&target]
                .iter()
                .map(|(i, c)| (i.clone(), coefficient_atom(c)))
                .collect();
            assert_eq!(reduction.expand(&target).unwrap(), terms);
        }
        assert!(actual[&i(1)].is_empty());
    }

    #[test]
    fn substitution_directions_agree_across_a_wide_shared_frontier() {
        let root = Integral(vec![1]);
        let left = Integral(vec![2]);
        let right = Integral(vec![3]);
        let x = parse!("native_wide::x");
        let mut reduction = Reduction::default();
        let mut a = BTreeMap::new();
        let mut b = BTreeMap::new();
        let mut order = Vec::new();
        for k in 10..139 {
            let terminal = Integral(vec![k]);
            let denominator = &x + Atom::num(k);
            a.insert(terminal.clone(), Atom::num(1) / &denominator);
            b.insert(terminal.clone(), (Atom::num(k) - &x) / denominator);
            order.push(terminal);
        }
        reduction.rules.insert(left.clone(), a);
        reduction.rules.insert(right.clone(), b);
        reduction.rules.insert(
            root.clone(),
            BTreeMap::from([(left.clone(), x.clone()), (right.clone(), Atom::num(1))]),
        );
        order.extend([left, right, root.clone()]);
        let retained = order.iter().cloned().collect();
        let context = RunContext::default();
        let mut expander = Expander::new(&reduction, &context).unwrap();
        let forward = expander.expand(&root).unwrap();
        let backward = expander
            .expand_backward(std::slice::from_ref(&root), &order, &retained)
            .unwrap()
            .remove(&root)
            .unwrap();
        assert_eq!(forward.len(), 129);
        for (terminal, coefficient) in forward {
            assert_eq!(coefficient, backward[&terminal]);
            let k = terminal.0[0];
            assert!(
                (coefficient_atom(&coefficient) - Atom::num(k) / (&x + Atom::num(k)))
                    .together()
                    .cancel()
                    .is_zero()
            );
        }
    }

    #[test]
    fn many_rational_dependency_paths_cancel_exactly() {
        let root = Integral(vec![1]);
        let terminal = Integral(vec![1000]);
        let mut reduction = Reduction::default();
        let mut row = BTreeMap::from([(terminal.clone(), Atom::num((7, 3)))]);
        for k in 1..=32 {
            let denominator = parse!("native_sum::x") + Atom::num(k);
            for (offset, sign) in [(1, 1), (33, -1)] {
                let intermediate = Integral(vec![(offset + k) as i16]);
                row.insert(intermediate.clone(), Atom::num(sign) / &denominator);
                reduction.rules.insert(
                    intermediate,
                    BTreeMap::from([(terminal.clone(), Atom::num(1))]),
                );
            }
        }
        reduction.rules.insert(root.clone(), row);
        reduction.residuals.push(terminal.clone());
        let context = RunContext::default();
        let result = Expander::new(&reduction, &context)
            .unwrap()
            .expand(&root)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            (coefficient_atom(&result[&terminal]) - Atom::num((7, 3)))
                .together()
                .cancel()
                .is_zero()
        );
    }

    #[test]
    fn forward_substitution_cancels_shared_dependencies_and_rejects_cycles() {
        let i = |n| Integral(vec![n]);
        let mut reduction = Reduction {
            rules: BTreeMap::from([
                (
                    i(1),
                    BTreeMap::from([(i(2), parse!("x")), (i(3), Atom::num(1))]),
                ),
                (i(2), BTreeMap::from([(i(4), parse!("x+1"))])),
                (
                    i(3),
                    BTreeMap::from([(i(4), parse!("-x*(x+1)")), (i(5), Atom::num(1))]),
                ),
                (i(4), BTreeMap::from([(i(6), parse!("1/(x+1)"))])),
            ]),
            residuals: vec![i(5), i(6)],
            ..Default::default()
        };
        let context = RunContext::default();
        let actual = Expander::new(&reduction, &context)
            .unwrap()
            .expand(&i(1))
            .unwrap();
        let actual: LinearCombination = actual
            .iter()
            .map(|(i, c)| (i.clone(), coefficient_atom(c)))
            .collect();
        assert_eq!(actual, BTreeMap::from([(i(5), Atom::num(1))]));
        assert_eq!(actual, reduction.expand(&i(1)).unwrap());
        reduction
            .rules
            .insert(i(5), BTreeMap::from([(i(1), Atom::num(1))]));
        assert!(matches!(
            Expander::new(&reduction, &context).unwrap().expand(&i(1)),
            Err(Error::IncompleteReduction(_))
        ));
    }
    #[test]
    fn filtered_substitution_preserves_all_paths_to_selected_sectors() {
        let target = Integral(vec![1, 1]);
        let low = Integral(vec![1, 0]);
        let high = Integral(vec![2, 1]);
        let child = Integral(vec![1, 2]);
        let reduction = Reduction {
            rules: BTreeMap::from([
                (
                    target.clone(),
                    BTreeMap::from([
                        (child.clone(), parse!("x")),
                        (low.clone(), parse!("1/(1+x^2)")),
                    ]),
                ),
                (
                    child,
                    BTreeMap::from([(high.clone(), parse!("1/x")), (low.clone(), parse!("x"))]),
                ),
            ]),
            residuals: vec![low.clone(), high.clone()],
            ..Default::default()
        };
        let context = RunContext::default();
        let mut expander = Expander::new(&reduction, &context).unwrap();
        let full = expander.expand(&target).unwrap();
        let filtered = expander.expand_filtered(&target, 2).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(coefficient_atom(&filtered[&high]), Atom::num(1));
        assert_eq!(
            coefficient_atom(&filtered[&high]),
            coefficient_atom(&full[&high])
        );
        assert!(
            (coefficient_atom(&full[&low]) - parse!("x^2+1/(1+x^2)"))
                .together()
                .cancel()
                .is_zero()
        );
    }
}
