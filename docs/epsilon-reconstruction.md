# High-precision numerical Laurent reconstruction

`epsilon::EpsilonFitProfile::new(digits, guard_digits, leading, last, refinement)`
proposes exact balanced nonzero regulator samples, their radius, and the required
working decimal precision. `epsilon::symmetric_epsilon_samples(count, &radius)`
also accepts a caller-selected arbitrary positive `Rational` radius.

For a requested coefficient width `w`, the first profile uses
`N = max(32, 2*w)` nodes and `r = 10^-s`, where
`s = max(3, ceil((digits + 64)/(N-w)))`. Refinement `a` uses `N+4*a` nodes,
radius `r/10^a`, and at least
`digits + (w-1)*(s+a) + 64 + 32*a` working digits. Larger caller guards and
arithmetic precision hints are retained. The automatic solver also increases
transport order and compares complete, independently evaluated fits before
assigning verified digits.

The proposal accounts for dividing the requested regular coefficients by powers
of the regulator radius. It is not an error bound: singularities in epsilon,
large higher coefficients, cancellation in reductions and boundary matching,
and inaccurate samples can require further refinement. `fit_epsilon` alone
never assigns verified digits. Its existing coefficientwise refinement uses
relative comparisons for resolved nonzero coefficients and an absolute criterion
near zero; callers needing significant digits for extremely small coefficients
must inspect their individual errors.

`fit_epsilon` removes the declared Laurent pole, forms exact normalized
coordinates `t = epsilon/r`, and calls Symbolica's native Newton interpolation.
No dense Vandermonde matrix is allocated. Exact coordinates that collide after
conversion to the selected precision produce a precision-retry error.
Symbolica's complex float field tracks precision during interpolation; working
precision alone does not authenticate the input samples or output accuracy.

For repeated high-precision AMF evaluations, use symbolic-epsilon preparation
(`sampled_reduction: false`) when feasible, to share exact reductions across
profiles. Numerical sample and boundary caches must retain their precision and
profile identities. A supplied reference is comparison data, never a numerical
seed for the result being checked.

## Symbolica reuse audit

The implementation reuses the existing algebra owner after three checks:

1. Symbolica's public polynomial API documents `Polynomial.interpolate` as
   Newton interpolation (`src/api/python/polynomial.rs`).
2. `MultivariatePolynomial<F, E>::newton_interpolation` is implemented for a
   generic field in `src/poly/gcd.rs`; Numerica's `FloatField<Complex<Float>>`
   implements the required field and floating-point traits.
3. A standalone executable probe against the installed Symbolica library
   reconstructed `2 + 3*t + 5*t^2` using that complex field. The retained RustFlow
   unit tests additionally exercise sparse Laurent coefficients, rounded node
   collisions, arbitrary rational radii, and a nonpolynomial complex Laurent
   series at 20,000 digits on independent 32- and 36-node profiles.

No interpolation, polynomial, arbitrary-precision or integer-relation algorithm
was introduced by this change.

## Long numerical series

Ordinary rational and direct-epsilon Taylor transport, numerical Frobenius
expansions, and native fixed-series adapters admit orders up to 200,000. Each
allocation is checked against a conservative 32 GiB workspace estimate using
working precision, component/channel count, polynomial degree, and retained
segments. Frobenius admission includes cached block factorizations and basis
transformations. Exact prepared logarithmic widths sharpen that estimate when
the generic exponents have no integer resonance; resonant and strict endpoint
evaluation retain a dimension-based logarithm bound.

The consumed infinity basis is released before Taylor continuation. Algebraic
chart owners retain the 10,000-order limit until their additional root and
kernel storage has an explicit high-order estimate. These checks are admission
estimates, not process memory caps; campaign jobs still require the external
CPU/RSS supervisor and independent numerical refinements.

Repeated real-Gamma evaluations in recursive boundaries reuse the existing
per-thread LRU at all working precisions. Each thread retains at most 64 entries
and an estimated 8 MiB of arguments/results. Keys include working bits and the
rounded argument, so independent precision profiles cannot reuse lower-precision
values. This memoization changes neither the Symbolica Gamma evaluator nor the
accuracy evidence required for a completed master.
