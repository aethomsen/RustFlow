#[path = "support/native_cut_graph.rs"]
mod native_cut_graph;
#[path = "support/native_mixed_cut_graph.rs"]
mod native_mixed_cut_graph;
use feynkit_graph::EdgeId;
use idenso::tensor::SymbolicTensor;
use native_cut_graph::{diagram, kinematics};
use std::{collections::BTreeMap, sync::Arc};
use symbolica::prelude::*;
use symbolica_amflow::cuts::*;
use symbolica_amflow::hepkit::GraphIntegral;
use symbolica_amflow::*;

fn channel() -> FutureTimelikeChannel {
    FutureTimelikeChannel {
        external: vec![Rational::one()],
    }
}
fn epsilon() -> Symbol {
    symbol!("native_cut_test::eps")
}

#[test]
fn native_weighted_cut_dimension_contraction_and_independent_laurent_fit() -> Result<()> {
    let eps = Atom::var(epsilon());
    let dimension = Atom::var(symbol!("native_cut_test::D"));
    let graph = GraphIntegral::new(
        Arc::new(
            diagram()
                .with_numerator(SymbolicTensor::infer((dimension - 3) / eps).unwrap())
                .unwrap(),
        ),
        &kinematics(),
    )?;
    let backend = RustRedBackend::default();
    let options = FlowOptions {
        workers: 2,
        ..Default::default()
    };
    let context = RunContext::default();
    let prepared = graph.prepare_cut_projection(
        0,
        &KinematicPoint::default(),
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive],
        &backend,
        &options,
        &context,
    )?;
    assert_eq!(prepared.leading_power(), -3);
    let result = prepared.solve(0, &options, &context)?;
    assert_eq!(result[0].verified_digits, Some(20));
    let p = Precision::decimal(80)?;
    let pi = ComplexFloat::new(p.real(1).pi(), p.real(0));
    let volume = p.div(&p.i(1), &p.scale(&pi, 8, 1));
    let gamma = ComplexFloat::new(p.real(0).euler(), p.real(0));
    // (D-3)/eps = 1/eps-2; Phi_2'(0)/Phi_2(0)=2-gamma+log(4*pi/s).
    let finite = p.mul(&volume, &p.sub(&p.log(&p.scale(&pi, 4, 25)), &gamma));
    assert!(p.close(&result[0].coefficients[&-1], &volume, 20));
    assert!(p.close(&result[0].coefficients[&0], &finite, 20));
    assert!(p.close(&result[0].coefficients[&-3], &p.zero(), 20));
    assert!(p.close(&result[0].coefficients[&-2], &p.zero(), 20));
    assert!(result[0].validation_samples > 0);
    Ok(())
}

#[test]
fn exact_point_is_applied_once_and_native_raised_cut_sign_is_retained() -> Result<()> {
    let a = Atom::var(symbol!("native_cut_test::a"));
    let eps = Atom::var(epsilon());
    let graph = GraphIntegral::new(
        Arc::new(
            diagram()
                .with_numerator(SymbolicTensor::infer(a.clone()).unwrap())
                .unwrap(),
        ),
        &kinematics(),
    )?
    .with_powers(&BTreeMap::from([(EdgeId(1), 2)]))?;
    let point = KinematicPoint(BTreeMap::from([(a, eps.clone()), (eps, Atom::num(3))]));
    let backend = RustRedBackend::default();
    let options = FlowOptions::default();
    let context = RunContext::default();
    let prepared = graph.prepare_cut_projection(
        0,
        &point,
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive],
        &backend,
        &options,
        &context,
    )?;
    let unit_graph = GraphIntegral::new(Arc::new(diagram()), &kinematics())?;
    let unit = unit_graph.prepare_cut_projection(
        0,
        &KinematicPoint::default(),
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive],
        &backend,
        &options,
        &context,
    )?;
    for eps in [Rational::from((1, 13)), Rational::from((1, 17))] {
        let value = prepared.evaluate(&eps, &options, &context)?;
        let reference = unit.evaluate(&eps, &options, &context)?;
        let p = Precision::decimal(60)?;
        // d/dm1^2 Phi_2(s;0,0)/Phi_2 = -(1-2eps)/s, with s=25.
        let factor =
            -eps.clone() * (Rational::one() - Rational::from(2) * &eps) / Rational::from(25);
        assert!(p.close(&value[0], &p.mul(&reference[0], &p.rational(&factor)), 40));
    }
    Ok(())
}

#[test]
fn cut_zero_rows_still_require_valid_measure_and_cancellation_propagates() -> Result<()> {
    let graph = GraphIntegral::new(Arc::new(diagram()), &kinematics())?
        .with_powers(&BTreeMap::from([(EdgeId(1), 0)]))?;
    let backend = RustRedBackend::default();
    let options = FlowOptions::default();
    let context = RunContext::default();
    let prepare = |prescription, channel: &FutureTimelikeChannel, context: &RunContext| {
        graph.prepare_cut_projection(
            0,
            &KinematicPoint::default(),
            epsilon(),
            channel,
            vec![prescription],
            &backend,
            &options,
            context,
        )
    };
    assert!(matches!(
        prepare(LoopPrescription::PlusI0, &channel(), &context),
        Err(Error::Unsupported(_))
    ));
    assert!(
        prepare(
            LoopPrescription::Insensitive,
            &FutureTimelikeChannel { external: vec![] },
            &context
        )
        .is_err()
    );
    let prepared = prepare(LoopPrescription::Insensitive, &channel(), &context)?;
    let p = Precision::decimal(60)?;
    assert_eq!(
        prepared.evaluate(&Rational::from((1, 13)), &options, &context)?,
        vec![p.zero()]
    );
    assert!(matches!(
        prepared.evaluate_samples(&[], &options, &context),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        prepared.evaluate_samples(&[Rational::zero()], &options, &context),
        Err(Error::InvalidInput(_))
    ));
    let explicit_cut = FlowOptions {
        mass_mode: MassMode::Propagators(vec![0]),
        ..options.clone()
    };
    assert!(matches!(
        prepared.evaluate(&Rational::from((1, 13)), &explicit_cut, &context),
        Err(Error::InvalidInput(_))
    ));
    let cancelled = RunContext::default();
    cancelled.cancellation.cancel();
    assert!(matches!(
        prepare(LoopPrescription::Insensitive, &channel(), &cancelled),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        prepared.solve(0, &options, &cancelled),
        Err(Error::Cancelled)
    ));
    Ok(())
}

#[test]
fn native_cut_cli_roundtrip_uses_hepkit_dot_and_exact_samples() {
    let temporary = std::env::temp_dir().join(format!("native-cut-cli-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    let graph = diagram()
        .with_numerator(SymbolicTensor::infer(Atom::num(7)).unwrap())
        .unwrap();
    std::fs::write(temporary.join("cut.dot"), graph.to_dot().unwrap()).unwrap();
    let mut card = serde_json::json!({
        "schema_version": 1,
        "model": concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/hepkit/massless_phi3.json"),
        "diagram": "cut.dot", "scalar_products": [{"left":"gammalooprs::P(0)","right":"gammalooprs::P(0)","value":"25"}],
        "options": {"digits": 20},
        "cut": {"index":0,"future_channel":["1"],"loop_prescriptions":["insensitive"],"epsilon_samples":["1/13","1/17"]}
    });
    let run = |card: &serde_json::Value| {
        let path = temporary.join("input.json");
        std::fs::write(&path, serde_json::to_vec(card).unwrap()).unwrap();
        std::process::Command::new(env!("CARGO_BIN_EXE_rustflow"))
            .arg("graph")
            .arg(path)
            .output()
            .unwrap()
    };
    let output = run(&card);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(data["operation"], "cut_graph_samples");
    assert!(data["verified_digits"].is_null());
    assert_eq!(data["values"].as_array().unwrap().len(), 2);
    card["cut"]
        .as_object_mut()
        .unwrap()
        .remove("epsilon_samples");
    let output = run(&card);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(data["operation"], "cut_graph");
    assert_eq!(data["verified_digits"], 20);
    let p = Precision::decimal(60).unwrap();
    let expected = p.div(&p.i(7), &ComplexFloat::new(p.real(1).pi() * 8, p.real(0)));
    let value = p
        .parse(
            data["coefficients"]["0"]["real"].as_str().unwrap(),
            data["coefficients"]["0"]["imaginary"].as_str().unwrap(),
        )
        .unwrap();
    assert!(p.close(&value, &expected, 20));
    card["cut"]["loop_prescriptions"] = serde_json::json!(["ignored"]);
    let output = run(&card);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("loop prescription must be"));
    std::fs::remove_dir_all(temporary).unwrap();
}

#[test]
fn cut_projection_retains_backend_guards_even_for_zero_rows() -> Result<()> {
    struct Guarded;
    impl ReductionBackend for Guarded {
        fn identity(&self) -> String {
            "native-cut-projection-guard-test".into()
        }
        fn reduce(
            &self,
            _: &IntegralFamily,
            _: &[Integral],
            _: &RunContext,
        ) -> Result<reduction::Reduction> {
            panic!("a cut projection must not request ordinary reduction")
        }
        fn reduce_cut(
            &self,
            family: &CutFamily,
            targets: &[Integral],
            context: &RunContext,
        ) -> Result<reduction::Reduction> {
            let mut result = RustRedBackend::default().reduce_cut(family, targets, context)?;
            result
                .nonzero_conditions
                .push(Atom::var(family.family().epsilon) - Atom::num(Rational::from((1, 13))));
            Ok(result)
        }
    }
    let graph = GraphIntegral::new(Arc::new(diagram()), &kinematics())?;
    let context = RunContext::default();
    let options = FlowOptions::default();
    let (family, weights) = graph.cut_integral_group(
        0,
        &KinematicPoint::default(),
        epsilon(),
        4,
        vec![LoopPrescription::Insensitive],
        &context,
    )?;
    for row in [weights, BTreeMap::new()] {
        let prepared = PreparedCutProjections::new(
            &family,
            &channel(),
            &[row],
            &KinematicPoint::default(),
            &Guarded,
            &options,
            &context,
        )?;
        assert!(!prepared.nonzero_conditions().is_empty());
        assert!(matches!(
            prepared.evaluate(&Rational::from((1, 13)), &options, &context),
            Err(Error::Numerical(_))
        ));
        assert!(
            prepared
                .evaluate(&Rational::from((1, 17)), &options, &context)
                .is_ok()
        );
    }
    let bad = BTreeMap::from([(
        Integral(vec![1, 1]),
        Atom::var(symbol!("native_cut_test::unresolved")),
    )]);
    assert!(matches!(
        PreparedCutProjections::new(
            &family,
            &channel(),
            &[bad],
            &KinematicPoint::default(),
            &Guarded,
            &options,
            &context
        ),
        Err(Error::InvalidInput(_))
    ));
    Ok(())
}

#[test]
fn exact_complex_weights_survive_native_graph_encoding_and_direct_projection() -> Result<()> {
    let imaginary = Atom::num(Complex::new(Rational::zero(), Rational::one()));
    let weight = imaginary / Atom::var(epsilon());
    let plain = GraphIntegral::new(Arc::new(diagram()), &kinematics())?;
    let complex = GraphIntegral::new(
        Arc::new(
            diagram()
                .with_numerator(SymbolicTensor::infer(weight.clone()).unwrap())
                .unwrap(),
        ),
        &kinematics(),
    )?;
    let backend = RustRedBackend::default();
    let options = FlowOptions::default();
    let context = RunContext::default();
    let (family, _) = plain.cut_integral_group(
        0,
        &KinematicPoint::default(),
        epsilon(),
        4,
        vec![LoopPrescription::Insensitive],
        &context,
    )?;
    let rows = [BTreeMap::from([(Integral(vec![1, 1]), weight)])];
    let direct = PreparedCutProjections::new(
        &family,
        &channel(),
        &rows,
        &KinematicPoint::default(),
        &backend,
        &options,
        &context,
    )?;
    let native = complex.prepare_cut_projection(
        0,
        &KinematicPoint::default(),
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive],
        &backend,
        &options,
        &context,
    )?;
    let unit = plain.prepare_cut_projection(
        0,
        &KinematicPoint::default(),
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive],
        &backend,
        &options,
        &context,
    )?;
    let p = Precision::decimal(60)?;
    let sample = Rational::from((1, 17));
    let expected = p.mul(
        &unit.evaluate(&sample, &options, &context)?[0],
        &p.div(&p.complex(0, 1), &p.rational(&sample)),
    );
    for actual in [
        direct.evaluate(&sample, &options, &context)?,
        native.evaluate(&sample, &options, &context)?,
    ] {
        assert!(p.close(&actual[0], &expected, 40));
    }
    Ok(())
}

#[test]
fn connected_native_two_loop_cut_graph_generates_virtual_boundaries() -> Result<()> {
    let graph = native_mixed_cut_graph::connected_diagram();
    let graph = GraphIntegral::new(Arc::new(graph), &kinematics())?;
    let point = KinematicPoint(BTreeMap::from([(
        Atom::var(symbol!("UFO::M")),
        Atom::one(),
    )]));
    let backend = RustRedBackend::default();
    let options = FlowOptions {
        guard_digits: 50,
        ..Default::default()
    };
    let context = RunContext::default();
    let prepared = graph.prepare_cut_projection(
        0,
        &point,
        epsilon(),
        &channel(),
        vec![LoopPrescription::Insensitive, LoopPrescription::PlusI0],
        &backend,
        &options,
        &context,
    )?;
    assert_eq!(prepared.family().family().loops.len(), 2);
    let sample = Rational::from((1, 13));
    let value = prepared.evaluate(&sample, &options, &context)?;
    let p = Precision::decimal(70)?;
    let pi = ComplexFloat::new(p.real(1).pi(), p.real(0));
    let gamma = |value: Rational| p.gamma_real(&p.rational(&value).re).unwrap();
    let volume = p.mul(
        &p.div(
            &p.pow(&p.i(21), &p.rational(&Rational::from((1, 2)))),
            &p.scale(&pi, 40, 1),
        ),
        &p.mul(
            &p.pow(&p.scale(&pi, 16, 21), &p.rational(&sample)),
            &p.div(
                &gamma(Rational::from((3, 2))),
                &gamma(Rational::from((3, 2)) - &sample),
            ),
        ),
    );
    let bubble = p.div(
        &p.mul(
            &gamma(sample.clone()),
            &p.powi(&gamma(Rational::one() - &sample), 2),
        ),
        &gamma(Rational::from(2) - Rational::from(2) * &sample),
    );
    let expected = p.mul(
        &volume,
        &p.mul(
            &bubble,
            &p.exp(&p.mul(&p.complex(0, 1), &p.mul(&pi, &p.rational(&sample)))),
        ),
    );
    assert!(
        p.close(&value[0], &expected, 20),
        "{} != {expected}",
        value[0]
    );
    Ok(())
}

#[test]
fn cancellation_during_cut_reduction_cannot_return_a_prepared_projection() -> Result<()> {
    struct CancelOnReduce;
    impl ReductionBackend for CancelOnReduce {
        fn identity(&self) -> String {
            "cancelled-cut-projection-test".into()
        }
        fn reduce(
            &self,
            _: &IntegralFamily,
            _: &[Integral],
            _: &RunContext,
        ) -> Result<reduction::Reduction> {
            panic!("cut reduction required")
        }
        fn reduce_cut(
            &self,
            family: &CutFamily,
            targets: &[Integral],
            context: &RunContext,
        ) -> Result<reduction::Reduction> {
            let result = RustRedBackend::default().reduce_cut(family, targets, context)?;
            context.cancellation.cancel();
            Ok(result)
        }
    }
    let graph = GraphIntegral::new(Arc::new(diagram()), &kinematics())?;
    let options = FlowOptions::default();
    let context = RunContext::default();
    assert!(matches!(
        graph.prepare_cut_projection(
            0,
            &KinematicPoint::default(),
            epsilon(),
            &channel(),
            vec![LoopPrescription::Insensitive],
            &CancelOnReduce,
            &options,
            &context
        ),
        Err(Error::Cancelled)
    ));
    Ok(())
}

#[test]
fn native_cut_cli_explicit_slots_are_physical_positions_and_validate_conflicts() {
    let temporary =
        std::env::temp_dir().join(format!("native-partial-cut-cli-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    let diagram = native_mixed_cut_graph::connected_diagram();
    let graph = GraphIntegral::new(Arc::new(diagram.clone()), &kinematics()).unwrap();
    // Native EdgeId(3) is the first virtual line, at physical slot 2.
    assert_eq!(
        graph.propagator_edges(),
        &[EdgeId(1), EdgeId(2), EdgeId(3), EdgeId(4)]
    );
    std::fs::write(temporary.join("cut.dot"), diagram.to_dot().unwrap()).unwrap();
    std::fs::write(
        temporary.join("model.json"),
        diagram.model().to_json().unwrap(),
    )
    .unwrap();
    let mut card = serde_json::json!({
        "schema_version":1,"model":"model.json","diagram":"cut.dot",
        "scalar_products":[{"left":"gammalooprs::P(0)","right":"gammalooprs::P(0)","value":"25"}],
        "substitutions":{"UFO::M":"1"},
        "options":{"digits":20,"guard_digits":50,"deformed_propagator_slots":[2]},
        "cut":{"index":0,"future_channel":["1"],"loop_prescriptions":["insensitive","+i0"],"epsilon_samples":["1/13"]}
    });
    let run = |card: &serde_json::Value| {
        let path = temporary.join("input.json");
        std::fs::write(&path, serde_json::to_vec(card).unwrap()).unwrap();
        std::process::Command::new(env!("CARGO_BIN_EXE_rustflow"))
            .arg("graph")
            .arg(path)
            .output()
            .unwrap()
    };
    let partial = run(&card);
    assert!(
        partial.status.success(),
        "{}",
        String::from_utf8_lossy(&partial.stderr)
    );
    let partial: serde_json::Value = serde_json::from_slice(&partial.stdout).unwrap();
    card["options"]
        .as_object_mut()
        .unwrap()
        .remove("deformed_propagator_slots");
    card["options"]["mass_mode"] = "all".into();
    let all = run(&card);
    assert!(
        all.status.success(),
        "{}",
        String::from_utf8_lossy(&all.stderr)
    );
    let all: serde_json::Value = serde_json::from_slice(&all.stdout).unwrap();
    let p = Precision::decimal(70).unwrap();
    let value = |data: &serde_json::Value| {
        p.parse(
            data["values"][0]["real"].as_str().unwrap(),
            data["values"][0]["imaginary"].as_str().unwrap(),
        )
        .unwrap()
    };
    assert!(p.close(&value(&partial), &value(&all), 20));
    assert!(partial["verified_digits"].is_null());
    card["options"]["deformed_propagator_slots"] = serde_json::json!([2]);
    let conflict = run(&card);
    assert!(!conflict.status.success());
    assert!(String::from_utf8_lossy(&conflict.stderr).contains("cannot be combined"));
    card["options"]["mass_mode"] = "explicit".into();
    for slot in [0, 1, 4, 999] {
        card["options"]["deformed_propagator_slots"] = serde_json::json!([slot]);
        let invalid = run(&card);
        assert!(!invalid.status.success());
        assert!(String::from_utf8_lossy(&invalid.stderr).contains("uncut physical denominators"));
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
