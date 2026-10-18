//! Unit-power massless two-point bananas, recognized from full affine sources.
//!
//! All routing, reconstruction and Gamma operations use existing native owners.
//! Unsupported topology, sheet or Gamma-pole cases retain recursive AMF.
use crate::{ComplexFloat as C, Error, Integral, IntegralFamily, Precision, Propagator, Result};
use symbolica::prelude::*;

struct Line {
    index: usize,
    route: Vec<Atom>,
    external: Rational,
    scale: Rational,
}
#[derive(Debug)]
struct Plan {
    loops: usize,
    dimension: i64,
    // Product of extracted positive quadratic scales, each at unit power.
    scale_product: Rational,
    // Absolute Jacobian of the old loop variables with respect to the new ones.
    jacobian: Rational,
    // Positive Euclidean invariant -Q^2 on the admitted spacelike sheet.
    euclidean_scale: Rational,
}

fn rational(a: &Atom) -> Option<Rational> {
    Rational::try_from(a).ok()
}

fn line(family: &IntegralFamily, index: usize) -> Result<Option<Line>> {
    let l = family.loops.len();
    let source = &family.propagators[index];
    if source
        .scalar_products
        .iter()
        .chain([&source.constant])
        .any(|a| rational(a).is_none())
    {
        return Ok(None);
    }
    let route = match crate::regions::branch(source, l) {
        Ok(route) => route,
        Err(Error::Unsupported(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let pivot = route.iter().position(|a| !a.is_zero()).unwrap();
    let diagonal = pivot * l - pivot * pivot.saturating_sub(1) / 2;
    let scale = rational(&source.scalar_products[diagonal]).unwrap();
    if scale <= Rational::zero() {
        return Ok(None);
    }
    let offset = l * (l + 1) / 2;
    let external =
        rational(&source.scalar_products[offset + pivot]).unwrap() / (&scale * &Rational::from(2));
    let vectors = route
        .iter()
        .map(|a| rational(a).unwrap())
        .chain([external.clone()])
        .collect::<Vec<_>>();
    // Propagator::quadratic accepts integer routing vectors. Clear their rational
    // denominators natively, reconstruct the entire affine form, then restore
    // its positive scale. An oversized integer route simply retains AMF.
    let denominator = vectors
        .iter()
        .fold(Integer::one(), |a, b| a.lcm(b.denominator_ref()));
    let multiplier = Rational::from(denominator);
    let integers = vectors
        .iter()
        .map(|a| {
            let a = a * &multiplier;
            i64::try_from(a.numerator_ref().clone()).ok()
        })
        .collect::<Option<Vec<_>>>();
    let Some(integers) = integers else {
        return Ok(None);
    };
    let reconstructed = Propagator::quadratic(
        &integers[..l],
        &integers[l..],
        Atom::new(),
        &family.external_gram,
    )?;
    let factor = Atom::num(&scale / &(&multiplier * &multiplier));
    if source
        .scalar_products
        .iter()
        .chain([&source.constant])
        .zip(
            reconstructed
                .scalar_products
                .iter()
                .chain([&reconstructed.constant]),
        )
        .any(|(actual, expected)| (actual - &factor * expected).together().cancel() != Atom::new())
    {
        return Ok(None);
    }
    Ok(Some(Line {
        index,
        route,
        external,
        scale,
    }))
}

impl Plan {
    fn find(family: &IntegralFamily, target: &Integral) -> Result<Option<Self>> {
        family.validate_integral(target)?;
        let l = family.loops.len();
        // This typed API represents integer powers only. Its RustRed conversion
        // constructs zero power shifts; no shifted-index native family is read.
        if l == 0 || family.external.len() != 1 || target.0.iter().any(|&a| a != 0 && a != 1) {
            return Ok(None);
        }
        family.validate()?;
        let Some(gram) = rational(&family.external_gram[0][0]) else {
            return Ok(None);
        };
        if gram >= Rational::zero() {
            return Ok(None);
        }
        let active = target
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| (a == 1).then_some(i))
            .collect::<Vec<_>>();
        if active.len() != l + 1 {
            return Ok(None);
        }
        let mut lines = Vec::with_capacity(l + 1);
        for index in active {
            let Some(line) = line(family, index)? else {
                return Ok(None);
            };
            lines.push(line);
        }
        let mut selected = Vec::new();
        let mut basis = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if basis.len() == l {
                break;
            }
            let mut candidate = basis.clone();
            candidate.push(line.route.clone());
            if crate::algebra::rref(candidate.clone()).1.len() > basis.len() {
                basis = candidate;
                selected.push(i);
            }
        }
        if basis.len() != l {
            return Ok(None);
        }
        let last = (0..lines.len()).find(|i| !selected.contains(i)).unwrap();
        let inverse = crate::algebra::inverse(&basis)?;
        let coefficients = crate::algebra::matmul(&[lines[last].route.clone()], &inverse)[0]
            .iter()
            .map(rational)
            .collect::<Option<Vec<_>>>()
            .unwrap();
        if coefficients.iter().any(Rational::is_zero) {
            return Ok(None);
        }
        let mut scale_product = lines[last].scale.clone();
        let mut transfer = lines[last].external.clone();
        let mut transformation = inverse;
        for (column, (&i, coefficient)) in selected.iter().zip(&coefficients).enumerate() {
            scale_product *= &lines[i].scale / &(coefficient * coefficient);
            transfer -= coefficient * &lines[i].external;
            for row in &mut transformation {
                row[column] = (&row[column] / Atom::num(coefficient.clone()))
                    .together()
                    .cancel();
            }
        }
        let jacobian = rational(&crate::algebra::determinant(transformation))
            .unwrap()
            .abs();
        let euclidean_scale = -(&transfer * &transfer * &gram);
        if euclidean_scale <= Rational::zero() || jacobian.is_zero() {
            return Ok(None);
        }
        // All L+1 actual active sources participated; inactive ISP coordinates
        // are never mistaken for extra propagator edges.
        debug_assert!(lines.iter().all(|line| target.0[line.index] == 1));
        Ok(Some(Self {
            loops: l,
            dimension: family.dimension,
            scale_product,
            jacobian,
            euclidean_scale,
        }))
    }
    fn evaluate(&self, epsilon: &Rational, p: Precision) -> Result<Option<C>> {
        let dimension = Rational::from(self.dimension) - epsilon * &Rational::from(2);
        let half = &dimension / &Rational::from(2);
        let loops = Rational::from(self.loops as i64);
        let lines = Rational::from((self.loops + 1) as i64);
        let arguments = [
            &lines - &(&loops * &half),
            &half - &Rational::one(),
            &lines * &(&half - &Rational::one()),
        ];
        // Removable singularities require a limit calculation. Preserve the old
        // numerical route instead of inferring zero from a divergent Gamma.
        if arguments
            .iter()
            .any(|a| a <= &Rational::zero() && a.denominator_ref().is_one())
        {
            return Ok(None);
        }
        let gamma = |a: &Rational| p.gamma_real(&p.rational(a).re);
        let numerator = p.mul(
            &gamma(&arguments[0])?,
            &p.powi(&gamma(&arguments[1])?, (self.loops + 1) as i64),
        );
        let ratio = p.div(&numerator, &gamma(&arguments[2])?);
        let scale = p.pow(
            &p.rational(&self.euclidean_scale),
            &p.rational(&(&loops * &half - &lines)),
        );
        let jacobian = p.pow(&p.rational(&self.jacobian), &p.rational(&dimension));
        let value = p.div(
            &p.mul(&p.mul(&ratio, &scale), &jacobian),
            &p.rational(&self.scale_product),
        );
        let value = if (self.loops + 1).is_multiple_of(2) {
            value
        } else {
            p.neg(&value)
        };
        if !p.finite(&value) {
            return Err(Error::Numerical(
                "nonfinite massless banana terminal".into(),
            ));
        }
        Ok(Some(value))
    }
}

pub(crate) fn evaluate(
    family: &IntegralFamily,
    target: &Integral,
    epsilon: &Rational,
    p: Precision,
) -> Result<Option<C>> {
    match Plan::find(family, target)? {
        Some(plan) => plan.evaluate(epsilon, p),
        None => Ok(None),
    }
}

#[cfg(test)]
#[path = "massless_banana_tests.rs"]
mod tests;
