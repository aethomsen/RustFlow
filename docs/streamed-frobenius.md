# Evaluate a solution without retaining its full Frobenius basis

`PreparedFrobenius::evaluate_solution` evaluates one weighted solution at a
nonzero local point. `evaluate_solution_with_winding` also specifies the local
logarithm winding. Both take constants in the same column order as
`PreparedFrobenius::evaluate`.

Use the existing full evaluator at a short order to determine integration
constants with `match_constraints`, `match_regions`, or `match_values`. The
short order must include every coefficient needed by that matching problem.
Then pass those constants and the longer truncation order to the solution
evaluator. The same immutable exact preparation can be shared between these
calls. Neither matched constants nor the returned truncated values carry an
automatic accuracy certificate.

For matching at an ordinary nonzero point, `PreparedFrobenius::match_values`
streams each long column into an n-by-n matrix of evaluated values and calls
the existing numerical solver. `match_values_with_winding` supplies the local
branch explicitly. These methods use the generic parameter policy of
`PreparedFrobenius::evaluate`; they do not automatically replace the engine's
separate strict endpoint admission path.

The new path changes retention, not algebra:

- The existing coefficient-domain recurrence emits one column at a time.
- An exact identity basis transformation moves the native column directly;
  other transformations use the existing coefficient transformation.
- The existing power/log evaluator, including its real Horner specialization,
  evaluates the column on the requested branch.
- Its weighted contribution is accumulated and its coefficients are dropped.
- Order-long LU caching is disabled on this path. The full evaluator retains
  its existing caching policy and remains available as a reference.

For n components, order N and log width w, the long coefficient storage changes
from approximately n²Nw complex numbers to one nNw buffer for an exact
identity transformation, or two nNw buffers during other basis
transformations. The identity check uses exact symbolic zero/one tests.
One source connection and bounded transient solver workspace remain live.
Recomputing factorizations for repeated exponents can cost time; this API
makes no speedup claim.

The existing 200,000-order and 32-GiB allocation admission remain in force.
Admission counts the finite polynomial source, transformation buffers and
current column width. If resonance creates additional logarithms, the new
width is checked before retaining that coefficient. Resonant dense systems
are checked before allocation. No full-dimension logarithm bound is allocated
in advance. These are conservative estimates rather than allocator quotas;
external process memory limits are still required.

Original denominator restrictions, parameter-collision checks, cancellation,
and nonfinite-value rejection remain active. Winding changes the logarithm
used for both fractional powers and explicit logarithms. It is local branch
information, not a path-continuation proof.

Before this change, source/API review, existing documentation and tests, and a
native executable matching probe were checked. The existing exact coefficient
and asymptotic APIs supported short matching but did not consume long columns
incrementally. Symbolica's native algebra already supplied the required
series, matrix, field and exact-conversion operations. No algebra feature was
added or duplicated.

`tests/frobenius_streaming.rs` compares against full-basis evaluation for
one-loop radial systems, the two-loop sunset connection, Jordan logarithms,
integer resonance and sheared higher poles at real and complex points with
nonzero winding. It also checks an independently differentiated exact
logarithmic solution, short asymptotic matching, cancellation and domain/order
guards. Private admission tests exercise logarithm growth, connection storage
and resonant-workspace limits without allocating oversized arrays.
