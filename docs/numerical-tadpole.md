# Numerical tadpole and long Frobenius expansions

`examples/numerical_tadpole.rs` computes the convergent raised-power radial
integral from its differential equation. It does not call `vacuum::tadpole` or
use the target gamma formula as an input. The only special-function prefactors
are the angular measure and the specified loop-normalization conversion.

For the Minkowski measure `d^d k/(i pi^(d/2))`, with `d=4-2 eps` and unit mass,
the input integral is

```text
I3 = -1/Gamma(2-eps) integral_0^1 x^(1-eps) (1-x)^eps dx.
```

The integrand has the epsilon-independent value `1/2` at `x=1/2`.
The coupled system `f'=((1-eps)/x-eps/(1-x))f`, `J'=f` is prepared exactly
with symbolic epsilon. A second local system uses the coordinate `1-x` and
`J'=-f`. Both are matched to `(f,J)=(1/2,0)` at the midpoint. Their constant
endpoint values determine the numerical definite integral. The primitive has
finite endpoint limits for the admitted `abs(eps)<1`; no pointwise finite
limit is claimed for its integrand when epsilon is negative.

Integration by parts gives `(d-2a) Ia - 2a I(a+1)=0`. At `a=1,2` this implies
`I1=2 I3/(eps*(eps-1))`. Multiplication by `exp(EulerGamma*eps)` converts the
result to Vakint's `FMFTandMATAD` normalization. The native gamma target may be
used by a separate reference check, but does not enter these inputs.

The executable's sample JSON is deliberately marked `accuracy_verified=false`.
Independent epsilon profiles, endpoint truncations, arithmetic precision, and
references must establish coefficient accuracy outside the single-sample call.

## Reuse audit and recurrence

The existing `DifferentialSystem::prepare_frobenius`, `PreparedFrobenius::evaluate`,
`FrobeniusBasis::match_values`, and `physical_limit` APIs already implement
singular endpoint matching. The previous numerical recurrence expanded every
entry of `x M(x)` into an order-long series. Geometric rational entries then
required a quadratic convolution. The original ordinary Taylor solver already
has the needed finite-polynomial denominator clearing in
`ode::prepared::PreparedRows`; that owner uses native Symbolica rational
polynomials, gcd, exact division, and coefficient extraction.

The numerical Frobenius domain now reuses that owner and matches coefficients
of `x D_i(x) y'_i = sum_j P_ij(x) y_j`, with each constant `D_i(0)` normalized
to one. The same indicial, resonance, logarithm, and basis-transformation code
remains authoritative. No interpolation, polynomial, or matrix algebra is
reimplemented. The exact coefficient domain retains the expanded recurrence
and provides an independent numerical regression oracle.

Validation includes rational radial systems with both epsilon signs, a Jordan
system, an integer-resonant system, and a rational coefficient with nonreal
poles, compared coefficientwise with the exact expanded recurrence. Existing
Frobenius and endpoint regression suites also apply. The baseline compiled
endpoint example was exercised before this optimization on the same radial
equation at epsilon `1/100` and order 240.

Extended numerical orders are bounded by 200,000 terms; requests above 10,000
terms additionally use a 32 GiB peak-allocation estimate, reserving twice the
coefficient storage for transient buffers. This is an admission estimate,
not a process memory limit. The campaign's
external memory supervisor remains required for large jobs. Exact-coefficient
resource limits are unchanged.
