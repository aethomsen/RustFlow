use super::*;

fn family_from_active(active: Vec<Propagator>, loops: usize) -> (IntegralFamily, Integral) {
    let count = loops * (loops + 1) / 2 + loops;
    let physical = active.len();
    let mut propagators = active;
    let mut rows = propagators
        .iter()
        .map(|a| a.scalar_products.clone())
        .collect::<Vec<_>>();
    for j in 0..count {
        let mut row = vec![Atom::new(); count];
        row[j] = Atom::one();
        let mut trial = rows.clone();
        trial.push(row.clone());
        if crate::algebra::rref(trial).1.len() > rows.len() {
            rows.push(row.clone());
            propagators.push(Propagator {
                constant: Atom::new(),
                scalar_products: row,
            });
        }
    }
    let f = IntegralFamily {
        name: "structural_unit_massless_banana".into(),
        loops: (0..loops).map(|i| format!("k{i}")).collect(),
        external: vec!["p".into()],
        external_gram: vec![vec![Atom::num(-1)]],
        physical_propagators: physical,
        propagators,
        dimension: 4,
        epsilon: symbol!("massless_banana_test::eps"),
    };
    let mut powers = vec![0; f.propagators.len()];
    powers[..physical].fill(1);
    (f, Integral(powers))
}
fn canonical(loops: usize) -> (IntegralFamily, Integral) {
    let gram = vec![vec![Atom::num(-1)]];
    let mut lines = Vec::new();
    for i in 0..loops {
        let mut route = vec![0; loops];
        route[i] = 1;
        lines.push(Propagator::quadratic(&route, &[0], Atom::new(), &gram).unwrap());
    }
    lines.push(Propagator::quadratic(&vec![1; loops], &[1], Atom::new(), &gram).unwrap());
    family_from_active(lines, loops)
}
fn strict(p: Precision, a: &C, b: &C) {
    assert!(!p.norm(b).is_zero());
    assert!(
        p.norm(&p.sub(a, b)) / p.norm(b) < p.tolerance(70),
        "{a} != {b}"
    );
}
// Iterated one-loop beta factors, independently ordered rather than using the
// telescoped terminal formula. Physical ODE references are checked separately.
fn convolution(loops: usize, e: &Rational, p: Precision) -> C {
    let d = Rational::from(4) - e * &Rational::from(2);
    let h = &d / &Rational::from(2);
    let mut a = Rational::one();
    let mut value = p.i(1);
    let g = |x: &Rational| p.gamma_real(&p.rational(x).re).unwrap();
    for _ in 0..loops {
        let numerator = p.mul(
            &p.mul(&g(&(&a + &Rational::one() - &h)), &g(&(&h - &a))),
            &g(&(&h - &Rational::one())),
        );
        let denominator = p.mul(&g(&a), &g(&(&d - &a - Rational::one())));
        value = p.mul(&value, &p.div(&numerator, &denominator));
        a = a + Rational::one() - &h;
    }
    if (loops + 1).is_multiple_of(2) {
        value
    } else {
        p.neg(&value)
    }
}
#[test]
fn unit_bananas_match_iterated_convolution_even_and_odd_wick_signs() {
    let p = Precision::decimal(100).unwrap();
    for loops in 1..=4 {
        let (f, i) = canonical(loops);
        for e in [Rational::from((1, 10)), Rational::from((9, 10))] {
            let a = evaluate(&f, &i, &e, p).unwrap().unwrap();
            let b = convolution(loops, &e, p);
            strict(p, &a, &b);
        }
    }
}
#[test]
fn affine_routes_rescalings_and_nonunit_jacobian_are_retained() {
    let gram = vec![vec![Atom::num(-1)]];
    let routes = [([2, 0], [1]), ([0, 3], [-2]), ([2, 3], [0])];
    let scales = [2, 3, 5];
    let active = routes
        .iter()
        .zip(scales)
        .map(|((route, external), s)| {
            let mut a = Propagator::quadratic(route, external, Atom::new(), &gram).unwrap();
            a.constant *= Atom::num(s);
            for c in &mut a.scalar_products {
                *c *= Atom::num(s);
            }
            a
        })
        .collect();
    let (f, i) = family_from_active(active, 2);
    let p = Precision::decimal(100).unwrap();
    let e = Rational::from((1, 10));
    let d = Rational::from(4) - &e * &Rational::from(2);
    let expected = p.div(
        &p.mul(&convolution(2, &e, p), &p.pow(&p.i(6), &p.rational(&(-d)))),
        &p.i(30),
    );
    strict(p, &evaluate(&f, &i, &e, p).unwrap().unwrap(), &expected);
    let mut permuted = f.clone();
    permuted.propagators.swap(0, 2);
    strict(
        p,
        &evaluate(&permuted, &i, &e, p).unwrap().unwrap(),
        &expected,
    );
    let mut reflected = f.clone();
    let routing = vec![
        vec![Atom::num(-1), Atom::new()],
        vec![Atom::new(), Atom::one()],
    ];
    reflected = reflected.transform_loops(&routing).unwrap();
    strict(
        p,
        &evaluate(&reflected, &i, &e, p).unwrap().unwrap(),
        &expected,
    );
}
#[test]
fn full_affine_mass_and_mixed_external_terms_are_checked() {
    let (f, i) = canonical(3);
    let mut massive = f.clone();
    massive.propagators[3].constant -= Atom::one();
    assert!(Plan::find(&massive, &i).unwrap().is_none());
    let mut mixed = f.clone();
    mixed.propagators[3].scalar_products[7] += Atom::one();
    assert!(Plan::find(&mixed, &i).unwrap().is_none());
    let mut higher_rank = f.clone();
    higher_rank.propagators[0].scalar_products[3] += Atom::one();
    assert!(Plan::find(&higher_rank, &i).unwrap().is_none());
}
#[test]
fn unsupported_indices_scales_invariants_and_parameters_fall_back() {
    let (f, i) = canonical(3);
    for value in [-1, 2] {
        let mut dotted = i.clone();
        dotted.0[0] = value;
        assert!(Plan::find(&f, &dotted).unwrap().is_none());
    }
    let mut pinched = i.clone();
    pinched.0[0] = 0;
    assert!(Plan::find(&f, &pinched).unwrap().is_none());
    let mut extra = i.clone();
    extra.0[4] = 1;
    // A positive declared ISP is malformed input, rejected before recognition.
    assert!(matches!(
        Plan::find(&f, &extra),
        Err(Error::InvalidInput(_))
    ));
    // Separately test a valid fifth physical massless line: it is outside the
    // L+1-line terminal class and must retain the existing numerical fallback.
    let mut extra_physical = f.clone();
    extra_physical.physical_propagators = 5;
    extra_physical.propagators[4] =
        Propagator::quadratic(&[1, 1, 0], &[0], Atom::new(), &f.external_gram).unwrap();
    extra_physical.validate().unwrap();
    assert!(Plan::find(&extra_physical, &extra).unwrap().is_none());
    let mut negative = f.clone();
    negative.propagators[0].scalar_products[0] = Atom::num(-1);
    assert!(Plan::find(&negative, &i).unwrap().is_none());
    let mut symbolic = f.clone();
    symbolic.propagators[0].scalar_products[0] = Atom::var(symbol!("massless_banana_test::scale"));
    assert!(Plan::find(&symbolic, &i).unwrap().is_none());
    for gram in [0, 1] {
        let mut g = f.clone();
        g.external_gram[0][0] = Atom::num(gram);
        g.propagators[3].constant = Atom::num(gram);
        assert!(Plan::find(&g, &i).unwrap().is_none());
    }
    let p = Precision::decimal(80).unwrap();
    assert!(evaluate(&f, &i, &Rational::zero(), p).unwrap().is_none());
}
#[test]
fn typed_family_conversion_has_no_hidden_fractional_power_shifts() {
    let (f, _) = canonical(3);
    let native = f.convert().unwrap();
    assert!(native.family.power_shifts().iter().all(|s| s.is_zero()));
}

#[test]
fn zero_transfer_and_missing_loop_direction_fall_back() {
    let gram = vec![vec![Atom::num(-1)]];
    let routes = [
        ([1, 0, 0], 0),
        ([0, 1, 0], 0),
        ([0, 0, 1], 0),
        ([1, 1, 1], 0),
    ];
    let active = routes
        .iter()
        .map(|(r, e)| Propagator::quadratic(r, &[*e], Atom::new(), &gram).unwrap())
        .collect();
    let (f, i) = family_from_active(active, 3);
    assert!(Plan::find(&f, &i).unwrap().is_none());
    let routes = [
        ([1, 0, 0], 0),
        ([0, 1, 0], 0),
        ([0, 0, 1], 0),
        ([1, 1, 0], 1),
    ];
    let active = routes
        .iter()
        .map(|(r, e)| Propagator::quadratic(r, &[*e], Atom::new(), &gram).unwrap())
        .collect();
    let (f, i) = family_from_active(active, 3);
    assert!(Plan::find(&f, &i).unwrap().is_none());
}
