use crate::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecursionMode {
    Amf,
    Ft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prescription {
    PlusI0,
    MinusI0,
}

/// How supplied boundary uncertainty is propagated during physical transport.
/// This selects verification policy, not a different mathematical identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryErrorStrategy {
    /// Keep successful scalar admission; retry its accuracy failures with the
    /// bounded signed proof without repeating central transport/refinement.
    #[default]
    Automatic,
    /// Use only the original scalar uncertainty estimate.
    ScalarNorm,
    /// Bounded signed fundamental matrices for ordinary rational systems;
    /// unsupported or inconclusive proofs retain the scalar-norm fallback.
    FundamentalMatrix,
}
impl BoundaryErrorStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Automatic => "automatic",
            Self::ScalarNorm => "scalar_norm",
            Self::FundamentalMatrix => "fundamental_matrix",
        }
    }
}
impl std::str::FromStr for BoundaryErrorStrategy {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "automatic" => Ok(Self::Automatic),
            "scalar_norm" => Ok(Self::ScalarNorm),
            "fundamental_matrix" => Ok(Self::FundamentalMatrix),
            _ => Err(Error::InvalidInput(
                "boundary_error_strategy must be automatic, scalar_norm or fundamental_matrix"
                    .into(),
            )),
        }
    }
}

/// Arithmetic for source-defect enclosures in rational Taylor charts.
/// Registered-root charts and Padé candidates retain their own arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualArithmetic {
    #[default]
    Ball,
    /// Bounded native integer products, with conservative ball fallback.
    AdaptiveInteger,
}
impl ResidualArithmetic {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ball => "ball",
            Self::AdaptiveInteger => "adaptive_integer",
        }
    }
}
impl std::str::FromStr for ResidualArithmetic {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "ball" => Ok(Self::Ball),
            "adaptive_integer" => Ok(Self::AdaptiveInteger),
            _ => Err(Error::InvalidInput(
                "residual_arithmetic must be ball or adaptive_integer".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MassMode {
    /// All lines at one loop; one massive line (or the first line) at higher loops.
    Auto,
    All,
    /// Shift the smallest group of lines with an equal nonzero intrinsic mass.
    Mass,
    /// Shift a single line in a branch containing the most propagators.
    Propagator,
    /// Shift the branch containing the fewest propagators.
    Branch,
    /// Shift the smallest set of lines outside an independent (L-1)-branch span.
    Loop,
    /// Zero-based physical denominator slots in the evaluated family, excluding ISPs.
    /// Cut graph conversion preserves `GraphIntegral::propagator_edges()` order.
    Propagators(Vec<usize>),
}

impl MassMode {
    /// Decode native interface settings without silently overriding a named mode.
    /// Bounds and cut/ISP admission require the evaluated family and are checked
    /// by its deformation owner. `explicit` permits lossless option roundtrips.
    pub fn from_selection(name: Option<&str>, slots: Option<Vec<usize>>) -> Result<Self> {
        if let Some(slots) = slots {
            if name.is_some_and(|name| name != "explicit") {
                return Err(Error::InvalidInput(
                    "deformed_propagator_slots cannot be combined with a named mass_mode other than explicit".into(),
                ));
            }
            if slots.is_empty() {
                return Err(Error::InvalidInput(
                    "deformed_propagator_slots must be nonempty".into(),
                ));
            }
            return Ok(Self::Propagators(slots));
        }
        match name.unwrap_or("automatic") {
            "automatic" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            "mass" => Ok(Self::Mass),
            "propagator" => Ok(Self::Propagator),
            "branch" => Ok(Self::Branch),
            "loop" => Ok(Self::Loop),
            "explicit" => Err(Error::InvalidInput(
                "explicit mass_mode requires deformed_propagator_slots".into(),
            )),
            _ => Err(Error::InvalidInput(
                "unknown auxiliary mass placement".into(),
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "automatic",
            Self::All => "all",
            Self::Mass => "mass",
            Self::Propagator => "propagator",
            Self::Branch => "branch",
            Self::Loop => "loop",
            Self::Propagators(_) => "explicit",
        }
    }

    pub fn deformed_propagator_slots(&self) -> Option<&[usize]> {
        match self {
            Self::Propagators(slots) => Some(slots),
            _ => None,
        }
    }
}

/// Proposal policy for ordinary continuation. Both policies use the same
/// candidate-specific differential-defect, conditioning, domain and branch checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepSizeStrategy {
    /// Halve the geometric proposal until every acceptance check passes.
    #[default]
    Halving,
    /// After a successful halving, test up to two larger steps in the same
    /// Taylor chart. Each extra predicate evaluation consumes the step budget.
    Bracketed,
}

/// Resource bounds for optional native rational-approximant trials. Unsupported
/// charts and failed candidates fall back to Taylor at the same proposed point.
#[derive(Debug, Clone)]
pub struct PadeOptions {
    pub degree: usize,
    pub max_common_degree: usize,
    pub max_input_bits: u64,
    pub max_coefficient_bits: u64,
    pub max_work: u64,
}
impl Default for PadeOptions {
    fn default() -> Self {
        Self {
            degree: 16,
            max_common_degree: 128,
            max_input_bits: 4096,
            max_coefficient_bits: 1_000_000,
            max_work: 100_000_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowOptions {
    pub digits: u32,
    pub guard_digits: u32,
    pub series_order: usize,
    pub pade: Option<PadeOptions>,
    pub residual_arithmetic: ResidualArithmetic,
    pub boundary_error_strategy: BoundaryErrorStrategy,
    /// Regular boundary-centered Taylor coordinates; physical paths remain unchanged.
    pub local_coordinate: crate::local_coordinates::LocalCoordinate,
    /// Maximum physical proposals per continuation call. A rejected Padé
    /// candidate may also try Taylor at the same proposal. Includes committed,
    /// rejected and successful superseded proposals.
    pub max_steps: usize,
    pub step_size_strategy: StepSizeStrategy,
    pub max_precision_attempts: usize,
    /// Maximum compatible source boundaries tried after physical transport accuracy failures.
    pub max_boundary_attempts: usize,
    pub workers: usize,
    pub dimension: i64,
    pub recursion: RecursionMode,
    pub mass_mode: MassMode,
    pub prescription: Prescription,
    /// Factor dimension dependence before numerical specialization. This takes
    /// precedence over sampled_reduction and retains symbolic epsilon in IBPs.
    pub refine_basis: bool,
    pub skip_reduction: bool,
    pub sampled_reduction: bool,
    pub cache_directory: Option<std::path::PathBuf>,
    /// Completed finite-epsilon samples, separate from verified boundary data.
    /// Only automatic integral projections use this checkpoint directory.
    pub sample_cache_directory: Option<std::path::PathBuf>,
    /// False forces fresh numerical samples while retaining exact reductions.
    /// Successfully recomputed samples still replace their checkpoints.
    pub reuse_samples: bool,
}
impl Default for FlowOptions {
    fn default() -> Self {
        Self {
            digits: 20,
            guard_digits: 40,
            series_order: 80,
            pade: None,
            residual_arithmetic: ResidualArithmetic::Ball,
            boundary_error_strategy: BoundaryErrorStrategy::Automatic,
            local_coordinate: crate::local_coordinates::LocalCoordinate::Identity,
            max_steps: 1000,
            step_size_strategy: StepSizeStrategy::Halving,
            max_precision_attempts: 3,
            max_boundary_attempts: 8,
            workers: 1,
            dimension: 4,
            recursion: RecursionMode::Amf,
            mass_mode: MassMode::Auto,
            prescription: Prescription::PlusI0,
            refine_basis: false,
            skip_reduction: false,
            sampled_reduction: true,
            cache_directory: None,
            sample_cache_directory: None,
            reuse_samples: true,
        }
    }
}
impl FlowOptions {
    /// An independent transport profile must change rational order as well as
    /// precision. Beyond the native degree cap the reference uses Taylor.
    pub(crate) fn refine_rational_order(&mut self, increment: usize) {
        if let Some(pade) = &mut self.pade {
            match pade
                .degree
                .checked_add(increment)
                .filter(|degree| *degree <= 32)
            {
                Some(degree) => pade.degree = degree,
                None => self.pade = None,
            }
        }
    }
    pub fn validate(&self) -> Result<()> {
        if cfg!(feature = "wasm") && self.workers != 1 {
            return Err(Error::Unsupported(
                "the browser build supports exactly one worker".into(),
            ));
        }
        if let Some(pade) = &self.pade
            && (pade.degree == 0
                || pade.degree > 32
                || pade.max_common_degree == 0
                || pade.max_common_degree > 256
                || pade.max_input_bits == 0
                || pade.max_input_bits > 16384
                || pade.max_coefficient_bits < pade.max_input_bits
                || pade.max_coefficient_bits > 8_000_000
                || pade.max_work == 0)
        {
            return Err(Error::InvalidInput("Padé resource bounds require degree 1..32, common degree 1..256, input bits 1..16384, coefficient bits between input bits and 8000000, and positive work".into()));
        }
        if self.digits == 0
            || self.workers == 0
            || self.series_order < 8
            || self.max_steps == 0
            || self.max_precision_attempts == 0
            || self.max_boundary_attempts == 0
        {
            return Err(Error::InvalidInput(
                "positive digits, workers, limits and series_order >= 8 required".into(),
            ));
        }
        if self.digits.checked_add(self.guard_digits).is_none() {
            return Err(Error::InvalidInput("working precision overflow".into()));
        }
        if self.series_order > crate::series_limits::MAX_ORDER
            || self.max_precision_attempts > 16
            || self.workers > 256
        {
            return Err(Error::Limit(
                "limits are 200000 series terms, 16 refinement attempts, and 256 workers; numerical series allocations are admitted separately".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn check(&self) -> Result<()> {
        if self.0.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub enum Progress {
    /// A coarse native-owner operation, useful before a long contraction or preparation.
    Stage {
        name: String,
    },
    Reduction {
        integrals: usize,
    },
    Substitution {
        rules: usize,
    },
    /// The actual dependency graph and selected exact substitution direction.
    /// Terminal counts include searched residuals as well as unsearched leaves.
    SubstitutionPlan {
        roots: usize,
        retained_nodes: usize,
        terminals: usize,
        minimum_lines: usize,
        backward: bool,
        weighted: bool,
    },
    ReductionFrontier {
        searched: usize,
        remaining: usize,
    },
    DifferentialClosure {
        round: usize,
        requested: usize,
        basis_size: usize,
        new_derivatives: usize,
    },
    SectorReduction {
        active_lines: usize,
        integrals: usize,
        visited: usize,
    },
    ParametricReduction {
        rays: usize,
        applied: usize,
        uncovered: usize,
        domains: usize,
        domain_applied: usize,
        elapsed_ms: u128,
    },
    SymmetryReduction {
        candidates: usize,
        automorphisms: usize,
        applied: usize,
        uncovered: usize,
        transport_failures: usize,
        search_limited: bool,
        elapsed_ms: u128,
    },
    SectorReduced {
        integrals: usize,
        seeds: usize,
        rows: usize,
        exact_trace_rows: usize,
        elapsed_ms: u128,
        exact_ms: u128,
    },
    BoundaryPlan {
        basis_size: usize,
        region_series: usize,
        coefficients: usize,
        max_half_order: usize,
    },
    Prepared {
        basis_size: usize,
        blocks: Vec<usize>,
    },
    Step {
        index: usize,
    },
    Sample {
        index: usize,
        total: usize,
    },
}

#[derive(Clone, Default)]
pub struct RunContext {
    pub cancellation: CancellationToken,
    pub progress: Option<Arc<dyn Fn(Progress) + Send + Sync>>,
}
impl RunContext {
    pub fn emit(&self, event: Progress) -> Result<()> {
        self.cancellation.check()?;
        if let Some(f) = &self.progress {
            f(event);
        }
        Ok(())
    }
}
