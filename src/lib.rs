//! Symbolica-native auxiliary mass flow.
#![doc = include_str!("../docs/usage.md")]
#[cfg(all(feature = "wasm", feature = "native"))]
compile_error!(
    "`wasm` cannot be combined with `native` or the native `python` feature; use --no-default-features --features python_wasm"
);
pub use symbolica;
mod algebra;
pub mod algebraic;
pub mod asymptotic;
#[cfg(feature = "automatic")]
pub mod benchmarks;
#[cfg(feature = "automatic")]
pub mod boundary;
#[cfg(feature = "automatic")]
mod bubble;
#[cfg(feature = "automatic")]
pub mod cache;
mod coefficient;
#[cfg(feature = "automatic")]
pub mod common_mass;
pub mod contour;
#[cfg(feature = "automatic")]
pub mod cut_flow;
#[cfg(feature = "automatic")]
pub mod cut_projection;
#[cfg(feature = "automatic")]
mod cut_regions;
#[cfg(feature = "automatic")]
pub mod cuts;
pub mod diffexp;
#[cfg(feature = "automatic")]
pub mod engine;
pub mod epsilon;
pub mod epsilon_flow;
pub mod epsilon_shearing;
pub mod error;
pub mod family;
mod fixed_series;
pub mod frobenius;
#[cfg(feature = "automatic")]
pub mod ft;
#[cfg(feature = "automatic")]
pub mod gaussian;
pub mod gg_hg;
#[cfg(feature = "automatic")]
pub mod hepkit;
mod integer_shearing;
#[cfg(feature = "automatic")]
pub mod integrand;
#[cfg(feature = "automatic")]
pub mod kinematic_derivative;
pub mod kinematics;
#[cfg(feature = "automatic")]
pub mod linear;
pub mod local_coordinates;
#[cfg(feature = "automatic")]
mod massless_banana;
#[cfg(feature = "automatic")]
mod native;
pub mod normalize;
pub mod numeric;
pub mod ode;
pub mod options;
#[cfg(feature = "automatic")]
pub mod phase_space;
mod physical_conditions;
#[cfg(feature = "automatic")]
pub mod physical_family;
#[cfg(feature = "automatic")]
pub mod physical_targets;
pub mod physical_transport;
pub mod projections;
#[cfg(feature = "python_api")]
pub mod python;
#[cfg(feature = "automatic")]
pub mod recursive;
#[cfg(feature = "automatic")]
pub mod reduction;
#[cfg(feature = "automatic")]
pub mod refine;
#[cfg(feature = "automatic")]
pub mod regions;
mod root_path;
#[cfg(feature = "automatic")]
mod sample_checkpoint;
mod series_limits;
pub mod singular_endpoint;
#[cfg(feature = "automatic")]
pub mod tensor;
pub mod transport_cache;
#[cfg(feature = "automatic")]
pub mod vacuum;
#[cfg(feature = "automatic")]
mod vacuum_peel;
pub use asymptotic::AsymptoticConstraint;
#[cfg(feature = "automatic")]
pub use cut_flow::PreparedCutFlow;
#[cfg(feature = "automatic")]
pub use cut_projection::PreparedCutProjections;
#[cfg(feature = "automatic")]
pub use engine::{
    PreparedFlow, evaluate_samples, solve_integral_combinations, solve_integral_projections,
    solve_integral_projections_normalized, solve_integrals, solve_prepared,
};
pub use epsilon::{LaurentExpansion, fit_epsilon};
pub use epsilon_flow::EpsilonShearedFlow;
pub use epsilon_shearing::EpsilonShearing;
pub use error::{Error, Result};
pub use family::{Integral, IntegralFamily, KinematicPoint, Propagator};
pub use local_coordinates::LocalCoordinate;
pub use numeric::{ComplexFloat, Precision};
pub use ode::{BoundaryData, DifferentialSystem, FlowDiagnostics, FlowResult};
pub use options::{
    BoundaryErrorStrategy, CancellationToken, FlowOptions, MassMode, PadeOptions, Prescription,
    Progress, RecursionMode, ResidualArithmetic, RunContext, StepSizeStrategy,
};
#[cfg(feature = "automatic")]
pub use physical_family::PreparedPhysicalFamily;
#[cfg(feature = "automatic")]
pub use physical_targets::ProjectedLaurentExpansion;
pub use physical_transport::RustFlow;
pub use projections::{ProjectionFactors, SampleNormalization};
#[cfg(feature = "automatic")]
pub use reduction::{
    NativeIdentityBank, NativeIdentityLimits, NativeIdentityProvider, ProvidedRustRedBackend,
    ReductionBackend, RustRedBackend, ScopedTableBackend, SeededRustRedBackend, TableBackend,
};
pub use transport_cache::RustFlowCache;
