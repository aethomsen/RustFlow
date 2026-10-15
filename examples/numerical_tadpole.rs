//! Numerical radial integral for the vacuum-master campaign.
//!
//! The target tadpole gamma formula is deliberately not used. The only gamma
//! evaluation is the d-dimensional angular measure of the convergent a=3
//! radial integral. Two Frobenius solutions determine its endpoint integrals
//! from the independently known integrand value f(1/2)=1/2.
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
use symbolica::prelude::*;
use symbolica_amflow::*;

#[derive(Deserialize)]
struct Input {
    epsilon: String,
    working_digits: u32,
    series_order: usize,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input_path = args.next().ok_or_else(|| {
        Error::InvalidInput("usage: numerical_tadpole INPUT.json OUTPUT.json".into())
    })?;
    let output_path = PathBuf::from(args.next().ok_or_else(|| {
        Error::InvalidInput("usage: numerical_tadpole INPUT.json OUTPUT.json".into())
    })?);
    let input: Input = serde_json::from_reader(std::fs::File::open(input_path)?)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    if input.series_order < 8 || input.series_order > 200_000 {
        return Err(Error::InvalidInput(
            "endpoint order must be in 8..=200000".into(),
        ));
    }
    let started = Instant::now();
    let p = Precision::decimal(input.working_digits)?;
    let (x, eps) = symbol!("numerical_tadpole::x", "numerical_tadpole::eps");
    let parse = |s: &str| {
        Atom::parse(s, "numerical_tadpole", Default::default())
            .map_err(|e| Error::InvalidInput(e.to_string()))
    };
    let eps_atom = parse(&input.epsilon)?;
    let epsilon = p.eval(&eps_atom, &Default::default())?;
    if epsilon == p.zero() || p.norm(&epsilon) >= p.real(1) {
        return Err(Error::InvalidInput("require 0 < |epsilon| < 1".into()));
    }
    let parameters = ahash::HashMap::from_iter([(Atom::var(eps), epsilon.clone())]);
    let context = RunContext::default();
    let left = DifferentialSystem {
        variable: x,
        matrix: vec![
            vec![parse("(1-eps)/x-eps/(1-x)")?, Atom::new()],
            vec![Atom::num(1), Atom::new()],
        ],
    }
    .prepare_frobenius(&context)?;
    let right = DifferentialSystem {
        variable: x,
        matrix: vec![
            vec![parse("eps/x-(1-eps)/(1-x)")?, Atom::new()],
            vec![Atom::num(-1), Atom::new()],
        ],
    }
    .prepare_frobenius(&context)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let midpoint = p.rational(&Rational::from((1, 2)));
    let boundary = [midpoint.clone(), p.zero()];
    let mut endpoints = Vec::new();
    let mut endpoint_seconds = Vec::new();
    for (name, prepared) in [("left", &left), ("right", &right)] {
        let now = Instant::now();
        let basis = prepared.evaluate(p, &parameters, input.series_order, &context)?;
        let constants = basis.match_values(&midpoint, &boundary, &parameters)?;
        let endpoint = basis.physical_limit(&constants, eps)?;
        endpoints.push(endpoint[1].clone());
        endpoint_seconds.push(now.elapsed().as_secs_f64());
        eprintln!(
            "endpoint {name} completed in {:.3}s",
            now.elapsed().as_secs_f64()
        );
    }
    let radial = p.sub(&endpoints[1], &endpoints[0]);
    let angular = p.gamma_real(&p.sub(&p.i(2), &epsilon).re)?;
    let raised = p.neg(&p.div(&radial, &angular));
    // Exact one-loop IBP recurrence: I_1=2 I_3/[epsilon(epsilon-1)] at m^2=1.
    let target = p.div(
        &p.scale(&raised, 2, 1),
        &p.mul(&epsilon, &p.sub(&epsilon, &p.i(1))),
    );
    let euler = ComplexFloat::new(p.real(0).euler(), p.real(0));
    let normalized = p.mul(&target, &p.exp(&p.mul(&epsilon, &euler)));
    let strings = |v: &ComplexFloat| [v.re.as_raw().to_string(), v.im.as_raw().to_string()];
    let output = serde_json::json!({
        "method": "numerical_radial_integral_frobenius_midpoint",
        "target_gamma_formula_used": false,
        "angular_measure_gamma_used": true,
        "epsilon": input.epsilon,
        "working_digits": input.working_digits,
        "working_bits": p.bits,
        "series_order": input.series_order,
        "normalization": "FMFTandMATAD",
        "radial_integral": strings(&radial),
        "raised_power_integral": strings(&raised),
        "target": strings(&normalized),
        "preparation_seconds": preparation_seconds,
        "endpoint_seconds": endpoint_seconds,
        "total_seconds": started.elapsed().as_secs_f64(),
        "accuracy_verified": false,
    });
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = output_path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&output).unwrap())?;
    std::fs::rename(tmp, output_path)?;
    eprintln!(
        "sample completed in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
