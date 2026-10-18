# Massless banana boundaries

`RecursiveBoundary` recognizes unit-power massless two-point banana integrals directly, avoiding an auxiliary-mass solve for these boundary subintegrals. It evaluates the convolution factors with Symbolica's arbitrary-precision Gamma implementation.

The supported class has L loops and L+1 active propagators, all at power one. Each active propagator must be an exactly reconstructed positive multiple of a massless affine momentum square. The routing coefficients must be rational and fit the existing integer-routing reconstruction. There must be one real spacelike external invariant. Inactive family coordinates remain inactive; they are not counted as physical edges.

The recognizer checks the complete affine source, including the constant contributed by the external Gram matrix. Native exact matrix operations select independent momenta, determine the remaining transfer momentum, and retain both propagator scales and the loop-routing Jacobian. Consequently this is independent of topology names, denominator order, loop orientation and the initially chosen momentum basis.

For h=d/2, N=L+1 and a positive Euclidean invariant Q_E², the canonical convolution is

\[
E_L(Q_E^2)=(Q_E^2)^{Lh-N}
\frac{\Gamma(N-Lh)\Gamma(h-1)^N}{\Gamma(N(h-1))}.
\]

The Minkowski measure is \(\prod_i d^d k_i/(i\pi^{d/2})\). Its value is \((-1)^N E_L\), multiplied by the absolute routing Jacobian to power d and divided by the product of the extracted positive propagator scales. This convention contains no Euler-constant exponential factor.

The formula follows by repeated one-loop massless convolution in the common convergence strip \(2<\operatorname{Re}d<2+2/L\), followed by meromorphic continuation in d. At displayed Gamma poles, the terminal delegates to the existing route instead of assuming a removable limit or an exact zero.

Custom terminal providers and certified scaleless sectors retain precedence. Dotted propagators, numerators, additional active lines, unsupported external invariants or routing, and nonpositive quadratic scales retain the existing recursive behavior. The typed RustFlow family has integer indices; its native RustRed conversion supplies zero power shifts. This terminal does not accept a separate native family with fractional index shifts.

This optimization changes proper boundary evaluation. A massive parent integral outside the recognized class still follows its usual numerical differential equation, boundary checks and independent precision refinement. The residual-arithmetic default is unchanged.
