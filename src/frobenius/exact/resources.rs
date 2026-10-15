//! Metadata admission before native rational operations. This performs no
//! coefficient arithmetic, factorization, polynomial products or root finding.
use super::*;

fn degree_bounds(
    view: AtomView<'_>,
    variable: Symbol,
    context: &RunContext,
) -> Result<(usize, usize)> {
    context.cancellation.check()?;
    Ok(match view {
        AtomView::Num(_) => (0, 0),
        AtomView::Var(v) => (usize::from(v.get_symbol() == variable), 0),
        AtomView::Add(sum) => sum.iter().try_fold((0_usize, 0_usize), |(n, d), term| {
            let (a, b) = degree_bounds(term, variable, context)?;
            Ok::<_, Error>((
                n.saturating_add(b).max(a.saturating_add(d)),
                d.saturating_add(b),
            ))
        })?,
        AtomView::Mul(product) => product
            .iter()
            .try_fold((0_usize, 0_usize), |(n, d), term| {
                let (a, b) = degree_bounds(term, variable, context)?;
                Ok::<_, Error>((n.saturating_add(a), d.saturating_add(b)))
            })?,
        AtomView::Pow(power) => {
            let (base, exponent) = power.get_base_exp();
            let AtomView::Num(number) = exponent else {
                return Ok((usize::MAX, usize::MAX));
            };
            let symbolica::coefficient::Coefficient::Complex(value) =
                number.get_coeff_view().to_owned()
            else {
                return Ok((usize::MAX, usize::MAX));
            };
            let magnitude = value
                .re
                .clone()
                .abs()
                .to_string()
                .parse::<usize>()
                .unwrap_or(usize::MAX);
            let (n, d) = degree_bounds(base, variable, context)?;
            let (n, d) = (n.saturating_mul(magnitude), d.saturating_mul(magnitude));
            if value.re < Rational::zero() {
                (d, n)
            } else {
                (n, d)
            }
        }
        _ => (usize::MAX, usize::MAX),
    })
}

/// `factor` covers the known products/derivatives/common-denominator additions
/// in the upcoming operation. `copies` is its matrix-entry work multiplier.
/// Estimates deliberately ignore cancellations. They are work admission, not
/// allocator quotas or a claim about interrupting a native operation mid-call.
pub(crate) fn preflight_operation(
    expressions: &[&Atom],
    variable: Symbol,
    factor: usize,
    copies: usize,
    limits: &ExactFrobeniusLimits,
    context: &RunContext,
) -> Result<()> {
    limits.validate()?;
    let domain = ExactDomain::dense(limits, context);
    let mut degree = 0_usize;
    let mut height = 0_u64;
    let mut bytes = 0_usize;
    for expression in expressions {
        context.cancellation.check()?;
        // This rejects unsupported syntax and bounds recursion before the
        // metadata-only degree traversal, including every integer exponent.
        domain.preflight_series(expression, variable, 0)?;
        let mut nodes = 0;
        height = height
            .saturating_add(domain.series_bound(
                expression.as_view(),
                variable,
                0,
                0,
                &mut nodes,
            )?)
            .saturating_add(2);
        let (n, d) = degree_bounds(expression.as_view(), variable, context)?;
        degree = degree.saturating_add(n).saturating_add(d);
        bytes = bytes.saturating_add(expression.as_view().get_byte_size());
    }
    let predicted_degree = degree.saturating_mul(factor);
    let coefficient_bits = height
        .saturating_add(2)
        .saturating_mul(degree.saturating_add(1) as u64)
        .saturating_mul(factor as u64);
    let cells = predicted_degree.saturating_add(1).saturating_mul(copies);
    let work = cells.saturating_mul(predicted_degree.saturating_add(1));
    if factor == 0
        || copies == 0
        || predicted_degree > limits.max_order
        || coefficient_bits > limits.max_coefficient_bits
        || cells > limits.max_scalar_cells
        || work > limits.max_scalar_cells.saturating_mul(16)
        || bytes.saturating_mul(factor).saturating_mul(copies) > limits.max_scalar_cells
    {
        return Err(Error::Limit(
            "exact root rational-operation degree/height/work estimate exceeds limit".into(),
        ));
    }
    Ok(())
}
