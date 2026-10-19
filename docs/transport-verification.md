# Taylor transport verification

Ordinary rational systems, their epsilon coefficient hierarchies and registered
square-root systems share the Taylor continuation controller. A trial must pass
truncation, differential-defect, domain and branch checks before its values or
intermediate boundary are committed. Independent precision/order evaluations
remain required by the checked interfaces. Working precision alone is not an
accuracy estimate.

## Whole-segment differential defects

For a Taylor polynomial `P(z)` centered at `c`, a rational connection row is
stored as `D_i(x) y'_i = sum_j N_ij(x) y_j`. The residual owner forms

```text
R_i(z) = D_i(c+z) P'_i(z) - sum_j N_ij(c+z) P_j(z).
```

Symbolica owns polynomial shifts, derivatives and products. Source numerator
and denominator degrees are retained in full, even when they exceed the Taylor
order. Epsilon hierarchies use the same construction with the appropriate
coefficient shifts. Cancellation occurs before taking coefficient magnitudes.
Coefficient collection uses Symbolica's polynomial owners with statistical zero
testing disabled. This matters when clearing rational rows creates exact
coefficients beyond the exponent range of a double. The
[coefficient-extraction reproducer](../tools/mre/symbolica-coefficient-list/README.md)
records why the convenience expression collector cannot be used here.
For a proposed step `h`, coefficient triangle inequalities estimate
`|h| sup_{|z| <= |h|} |R_i(z) / D_i(c+z)|`. If the denominator lower bound is
inconclusive, the controller rejects the trial and tries a shorter step.

This addresses the [recorded sparse-polynomial counterexample](../reports/validation/2026-10-05-residual-alias-counterexample.json):
its forcing vanishes at both old defect sample points and lies above both
retained expansion orders. Increasing the number of sample points alone does
not resolve that class of failure.

Registered roots satisfy scalar rational equations
`r' = R'/(2 R) r`, where `r^2 = R`. Their Taylor defects and scalar amplification
estimate the root approximation errors over the trial disk. Positive product
bounds propagate those errors into the connection. The main residual retains
full rational source polynomials and products of root Taylor polynomials;
root-sheet and original-domain checks remain separate requirements.

This new main-residual test applies to ordinary Taylor transport. Frobenius and
analytic-origin initialization have separate solution recurrences. An
analytic-origin root chart can check root errors without claiming a
whole-segment bound for that separate solution recurrence.

Ordinary rational and epsilon charts retain exact Gaussian-rational source rows
after denominator clearing. Symbolica's directed `ComplexBall` and polynomial
arithmetic enclose their residual against those rows, interpreting retained
Taylor coefficients and finite numerical parameters as their stored dyadics.
Directed coefficient bounds use a component one-norm disk containing the step.
The endpoint-conditioning diagnostic separately checks arithmetic cancellation.
Registered-root assembly retains its MPFR-estimate scope.

The whole-segment test bounds a local differential defect; it does not by itself
prove an accumulated solution-error bound. Propagation through nonnormal
systems, accumulation over many segments, arithmetic cancellation and final
output conditioning require separate accounting. Independent refinements and
reference comparisons establish evidence for their particular inputs, not a
general guarantee for every connection. The [large-cancellation reproducer](../reports/validation/2026-10-05-coordinate-roundoff-counterexample.json)
records the previous false-verification failure. The new
[arithmetic-conditioning guard](arithmetic-conditioning.md) requests fresh
higher-precision evaluation and caps cache evidence. An unresolved containing
saved segment is reported rather than silently discarded during path lookup.
Severe non-dyadic cancellation can still exhaust the bounded precision budget.

## Step proposals and retained evidence

`StepSizeStrategy::Halving` remains the default. The optional `Bracketed`
strategy first obtains a passing step by halving and then tests at most two
larger steps in the same chart. Each candidate passes the same checks; acceptance
is not assumed to be monotone in the step length. Only the selected candidate
updates the solution, root state, observer or saved trajectory.


With identity coordinates, Bracketed starts at up to three quarters of the
nearest-pole radius; Halving retains its one-third-radius proposal. Failed
proposals still backtrack through the same acceptance checks. Mapped-coordinate
and entire-system proposals are unchanged.

The public regression compares both strategies for `y′ = 2y/(1+x)`, with
`y(0)=1`, continued to `x=100`. It checks the exact result `10201` and requires
fewer committed charts with Bracketed:

```sh
cargo test --release --test bracketed_step_proposal -- --nocapture
```

`max_steps` counts every acceptance predicate, including rejected trials and
successful trials replaced by a larger one. Diagnostics distinguish committed
steps, rejected steps and superseded successes. On a successful continuation,
these sum to `predicate_evaluations`.

If all 32 local trial steps fail, the accuracy error includes the first and last
rejection reasons. Tail and differential-defect failures report the component,
measured error and admission budget; branch and domain failures remain distinct.
These diagnostics do not relax any acceptance threshold.

Persisted `RustFlowCache` snapshots and boundary identities contain source and
dependency fingerprints. A build that changes the numerical implementation
rejects older snapshots before exact-hit reuse. Re-importing values as supplied
boundaries requires independent accuracy evidence; changing a precision label
cannot repair evidence produced by an older incorrect evaluation.
