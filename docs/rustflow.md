# RustFlow physical transport and boundary cache

`RustFlow` is the native physical-transport object; `RustFlowCache` is a separate,
mutable bank shared across requests. Load a nonempty binary snapshot, pass a
mutable reference for each evaluation, and save the growing bank afterwards.
Neither object requires a Python runtime. The future Python API belongs in
HEPKit; this repository supplies the native core and the CLI.

The [physical-cache example](../examples/physical_cache.rs) seeds an exact
analytic boundary, evaluates one destination, evaluates a nearby destination
from the newly populated bank, and repeats the second request as a cache hit.

```rust,ignore
let mut cache = RustFlowCache::load(cache_directory)?;
// Insert an AMF, analytic, or supplied boundary with accuracy evidence if needed.
let result = flow.evaluate_to(
    &mut cache, &destination, epsilon_range, &options, &context, &cost_policy,
)?;
cache.save(cache_directory)?;
```

`KinematicSystem` holds exact partial-derivative matrices in a common basis.
`KinematicPath` performs simultaneous coordinate substitution with the full
chain-rule Jacobian. `EpsilonSystem` expands rational epsilon dependence exactly
and integrates its coefficient hierarchy directly. The recurrence clears
denominators once per physical row and shares those finite polynomials between
epsilon orders; it does not construct a dense augmented matrix.

`KinematicDerivative` derives invariant derivatives from an integral family,
including the change of the external Gram matrix. The older
`parameter_derivative` deliberately keeps scalar products fixed for auxiliary
and Feynman parameters. They describe different derivatives.

## Reusing computed physical points

Each cache record includes the full system/basis/normalization/branch identity,
physical coordinates, epsilon range, coefficient values, and accuracy evidence.
Exact input coordinates, numerical coordinates, and exact coordinate images of
numerically selected path parameters remain distinguishable. Retaining the exact
image of a path parameter prevents separately rounded invariants from leaving
the original kinematic line.

`RustFlow::evaluate_to` first selects a compatible starting boundary. An exact
coordinate hit returns the stored value; a nearby point supplies the starting
condition for a new transport to the actual requested destination. The default
provided distance policy uses arbitrary-precision scaled coordinate distances
and requires a caller-supplied path-admissibility check. Applications may provide
their own `TransportCost` policy, including singularities and expected step count.
Coordinate differences are formed exactly before numerical distance evaluation,
so a large common offset cannot erase a short separation. Equal rounded distance
costs are compared exactly for rational-complex coordinates; algebraic distances
use consistent higher-precision comparisons before accuracy breaks remaining ties.

Accepted intermediate points are compared at the same coordinates between
independent precision/order runs. Endpoint accuracy is not automatically assigned
to every intermediate segment. Boundary uncertainty is propagated with a
weighted matrix-norm estimate, using lower bounds for rational denominators on each local
disk and subdivisions when those bounds are inconclusive. A cache record's
working precision cannot replace its verified accuracy. Stronger cached accuracy
requires explicitly passing tighter independent comparisons and input-error
checks, with a conservative arithmetic ceiling and two-digit margin.
For physical transport and the point cache, `verified_digits = d` uses
`error <= 10^-d * max(1, |coefficient|)` for each coefficient. This is an absolute
criterion below unit magnitude and a relative criterion above it; it does not
claim `d` significant digits for every small nonzero coefficient. Returned
component error estimates allow applications to impose relative tolerances on
projected form factors or amplitudes after their exact normalization.
The weights retain separate physical-component and epsilon-order scales: a large
higher-order coefficient cannot inject its absolute uncertainty into an uncoupled
lower-order coefficient.

These are consistency/error estimates, not interval-arithmetic proofs. Repeated
transport from the same uncertain input cannot improve the input's accuracy.
If propagated uncertainty exceeds the request, use a closer or more accurate
boundary. Input reference precision and inherited errors must be supplied
honestly, even when a reference contains hundreds of displayed digits.

Insertions are indexed by content keys and committed in validated batches;
evaluation does not clone the entire existing bank. Binary persistence retains
Symbolica expressions and precision-bearing numbers, checks schema and dependency
compatibility plus an authenticated payload, and replaces snapshots atomically.
Concurrent processes should coordinate snapshot writes or explicitly merge
their banks; independent writers can otherwise replace one another's snapshots.

## Current scope

The cache-first orchestration currently handles supplied epsilon-regular rational
systems and regular straight paths. The standalone coefficient solver supports
explicit complex polygonal contours and retains logarithm monodromy. Both use
the shared AMF Taylor engine and its tail/defect checks. Matrix epsilon poles
require a basis change; common Laurent leading powers of the solution are allowed.

Standalone [square-root branch transport](algebraic.md) now accepts explicit
regular contours and initial sheets. Automatic threshold prescriptions and
algebraic cache reuse remain in the [parity plan](parity-plan.md), along with
the larger five-point and full-amplitude DiffExp examples. The live
polylogarithm oracle is checked through independent boundary construction, not
by using its target value as a boundary.

`tests/mg5_pointbank.rs` uses actual boundary data from the nonplanar Higgs+jet
application grid for its closed first two masters. Both new destinations agree
with live DiffExp to 20 absolute digits; the second starts from the newly cached
first point, and a repeated request after binary restart performs no transport.
This validates the cache on application data without claiming the complete
61-master system or the complete amplitude.

The HEPKit integration uses native DOT/model objects, `LoopMomentumBasis` routing,
Linnet graph algorithms, and Spenso/Idenso/HEPKit tensor facilities. One application
model is shared by `Arc<Model>`. `GraphIntegral::integral_groups` preserves exact
symbolic invariants when they are not specialized, so those families can be passed
to `PreparedPhysicalFamily`. It closes all requested physical derivatives on one
common basis; `seed_cache` computes verified AMF/FT master boundaries and appends
them to the caller's existing bank. Additional requested seed digits reserve
accuracy for later transports. `tests/native_physical.rs` exercises this complete
native DOT → symbolic family → AMF boundary → growing physical cache route.

Original family coefficients, denominator ordering, Gram matrix and nonzero
conditions enter physical basis identity. The restrictions are validated even if
they cancel from the final ODE, including before an exact cache hit. Candidate
straight paths that cross an excluded chart point are filtered before ranking.
Conditions are interpreted over formal epsilon, using Symbolica polynomial gcd
and root finding for their common physical zero sets. The binary cache schema
was incremented to retain these conditions; incompatible older snapshots are
rejected explicitly.

## Reducing transported masters to requested integrals

`PreparedPhysicalFamily::required_master_range(point, target_range)` determines
the epsilon orders needed by the retained exact target reductions. For example,
a coefficient proportional to `1/epsilon²` requires masters through epsilon
power two to obtain a target's finite coefficient. The common master leading
power is conservatively `-2*loops`; omitted pole coefficients are not assumed zero.
This bound requires epsilon-independent propagator coefficients and kinematics;
families with epsilon-dependent normalizations need a separate pole analysis and
are rejected by these projection methods.

Transport that master range, then call `project_targets(boundary, target_range,
digits)`. Coefficients are specialized exactly at the boundary coordinates and
expanded with Symbolica. Products use native Symbolica series with fixed MPFR
coefficient arithmetic. Unknown higher master orders produce an error. Target
maps remain on the prepared family, so requests sharing a master basis can reuse
the same cached values with different target reductions.

Projection propagates inherited master uncertainties through the exact weights
and repeats arithmetic with additional working bits. It does not promote input
accuracy or claim a new epsilon fit. Reserve extra verified digits when seeding
and transporting masters: cancellations or large reduction coefficients can
otherwise exhaust the target accuracy. An `Accuracy` error asks for more accurate
master data. The native bubble regression transports at 28 digits and projects
the scalar and squared-propagator targets at 20 digits.

### Verified state compression for finite-point transport

`DifferentialSystem::compress_states(Q, selected, conditions, context)` accepts an
exact rectangular embedding `old_y = Q * selected_y`. It checks `Q[selected]=I`
and the full differential identity, then derives the reduced connection from
selected rows of `A_old * Q`. Symbolica owns the exact matrix arithmetic. The
method does not discover an integral basis or validate an IBP reduction's origin.

The resulting opaque `StateCompression` compiles through the existing rational
ODE machinery. All declared nonzero conditions and original/map denominators
remain excluded even if they cancel in the reduced matrix. Supply assumptions
that were discarded before constructing the input Atoms explicitly.

Use `project_boundary` to select a full ordinary-point boundary after a numerical
subspace-consistency check, transport through `reduced_system()`, then call
`reconstruct` at the regular finite matching point. Reconstruction returns every
old component for the original singular-endpoint matching. It does not evaluate
singular limits or prove boundary/transport accuracy; independent numerical
refinement is still required. A map may introduce an excluded point even when
the original differential system is regular there.

The projection check uses the existing component-wise bound
`norm(full[i]-restored[i]) <= 10^-checked_digits * max(1,norm(full[i]))`.
It does not claim significant digits for small nonzero components or exact-zero
evidence. Callers can reconstruct the selected initial state to record residuals;
physical qualification still needs proper relative comparisons and source-proved
zeros.
