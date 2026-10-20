#[path = "support/native_cut_graph.rs"]
mod native_cut_graph;
use feynkit_graph::EdgeId;
use idenso::tensor::SymbolicTensor;
use native_cut_graph::{diagram, kinematics};
use std::{collections::BTreeMap, sync::Arc};
use symbolica::prelude::*;
use symbolica_amflow::cuts::*;
use symbolica_amflow::hepkit::GraphIntegral;
use symbolica_amflow::phase_space::PreparedTwoBodyPhaseSpace;
use symbolica_amflow::*;

fn convert(graph: &GraphIntegral) -> (CutFamily, reduction::LinearCombination) {
    graph
        .cut_integral_group(
            0,
            &KinematicPoint::default(),
            symbol!("native_cut_test::eps"),
            4,
            vec![LoopPrescription::Insensitive],
            &RunContext::default(),
        )
        .unwrap()
}
fn prepared(family: &CutFamily, targets: &[Integral]) -> PreparedTwoBodyPhaseSpace {
    PreparedTwoBodyPhaseSpace::new(
        family,
        &FutureTimelikeChannel {
            external: vec![Rational::from(1)],
        },
        targets,
        &KinematicPoint::default(),
        &RustRedBackend::default(),
        &FlowOptions::default(),
        &RunContext::default(),
    )
    .unwrap()
}
#[test]
fn native_builder_cut_orientation_and_numerator_reach_phase_space() {
    let graph = GraphIntegral::new(
        Arc::new(
            diagram()
                .with_numerator(SymbolicTensor::infer(Atom::num(7)).unwrap())
                .unwrap(),
        ),
        &kinematics(),
    )
    .unwrap();
    let (family, terms) = convert(&graph);
    assert_eq!(family.family().physical_propagators, 2);
    assert_eq!(
        terms,
        BTreeMap::from([(Integral(vec![1, 1]), Atom::num(7))])
    );
    let mut loops = Rational::from(0);
    let mut external = Rational::from(0);
    for (&slot, definition) in family.cuts().lines() {
        let CutDefinition::PositiveEnergy { momentum } = definition else {
            panic!("physical cut lost energy orientation")
        };
        loops += &momentum.loops[0];
        external += &momentum.external[0];
        assert!(family.cut_mass_squared(slot).unwrap().is_zero());
    }
    assert!(loops.is_zero());
    assert_eq!(external, Rational::from(1));
    let values = prepared(&family, &[Integral(vec![1, 1])])
        .solve(0, &FlowOptions::default(), &RunContext::default())
        .unwrap();
    let p = Precision::decimal(80).unwrap();
    let expected = p.div(&p.i(7), &ComplexFloat::new(p.real(1).pi() * 8, p.real(0)));
    assert_eq!(values[0].verified_digits, Some(20));
    assert!(p.close(&p.scale(&values[0].coefficients[&0], 7, 1), &expected, 20));
    assert!(matches!(
        graph.integral_groups(
            &KinematicPoint::default(),
            symbol!("native_cut_test::eps"),
            4,
            100,
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
}
#[test]
fn native_reversed_partition_and_edge_reversal_preserve_physical_meaning() {
    let original = diagram();
    let source = original.cuts()[0].clone();
    let reversed = original
        .clone()
        .with_cut_partitions(vec![(source.right.half_edges, source.left.half_edges)])
        .unwrap();
    let (backward, _) = convert(&GraphIntegral::new(Arc::new(reversed), &kinematics()).unwrap());
    let epsilon = Rational::from((1, 13));
    let context = RunContext::default();
    let options = FlowOptions::default();
    assert!(
        prepared(&backward, &[Integral(vec![1, 1])])
            .evaluate(&epsilon, &options, &context)
            .unwrap()[0]
            == Precision::decimal(60).unwrap().zero()
    );
    let rerouted = original.clone().reverse_edge(EdgeId(1)).unwrap();
    let (ordinary, _) = convert(&GraphIntegral::new(Arc::new(original), &kinematics()).unwrap());
    let (rerouted, _) = convert(&GraphIntegral::new(Arc::new(rerouted), &kinematics()).unwrap());
    let a = prepared(&ordinary, &[Integral(vec![1, 1])])
        .evaluate(&epsilon, &options, &context)
        .unwrap();
    let b = prepared(&rerouted, &[Integral(vec![1, 1])])
        .evaluate(&epsilon, &options, &context)
        .unwrap();
    assert!(Precision::decimal(60).unwrap().close(&a[0], &b[0], 40));
}
#[test]
fn cut_cancellations_vanish_without_relabeling_the_native_basis() {
    let diagram = diagram();
    let kin = kinematics();
    let native = diagram.propagator_family(&kin).unwrap();
    let numerator = diagram
        .clone()
        .with_numerator(SymbolicTensor::infer(native.denominators()[0].clone()).unwrap())
        .unwrap();
    let (family, terms) = convert(&GraphIntegral::new(Arc::new(numerator), &kin).unwrap());
    assert_eq!(family.family().physical_propagators, 2);
    assert!(terms.is_empty());
    let graph = GraphIntegral::new(Arc::new(diagram), &kin)
        .unwrap()
        .with_powers(&BTreeMap::from([(EdgeId(1), 0)]))
        .unwrap();
    let (family, terms) = convert(&graph);
    assert_eq!(family.cuts().lines().len(), 2);
    assert!(terms.is_empty());
    assert!(matches!(
        graph.cut_integral_group(
            1,
            &KinematicPoint::default(),
            symbol!("native_cut_test::eps"),
            4,
            vec![LoopPrescription::Insensitive],
            &RunContext::default()
        ),
        Err(Error::InvalidInput(_))
    ));
}
