use feynkit_graph::{EdgeId, IntegralFamily as NativeFamily, symbols};
use feynkit_kinematics::Kinematics;
use feynkit_model::Model;
use idenso::tensor::SymbolicTensor;
use std::{collections::BTreeMap, sync::Arc};
use symbolica::prelude::*;
use symbolica_amflow::{hepkit::GraphIntegral, *};

const MODEL: &str = include_str!("../fixtures/hepkit/massless_phi3.json");
const BUBBLE: &str = include_str!("../fixtures/hepkit/massless_bubble.dot");

fn exact(a: &Atom, b: &Atom) {
    assert!((a - b).together().cancel().is_zero(), "{a} != {b}");
}
fn scalar(source: &str) -> Atom {
    Atom::parse(source, "feynkit_graph", ParseSettings::default()).unwrap()
}
fn model() -> Arc<Model> {
    Arc::new(Model::from_json(MODEL).unwrap())
}
fn kinematics(s: Atom) -> Kinematics {
    Kinematics::in_dimension(&scalar("D_"))
        .unwrap()
        .with_mass_squared(&symbols::external_momentum().call(1), s)
        .unwrap()
}
fn groups(
    graph: &GraphIntegral,
    point: &KinematicPoint,
) -> Vec<(IntegralFamily, reduction::LinearCombination)> {
    graph
        .integral_groups(
            point,
            symbol!("feynkit_graph::eps"),
            4,
            1000,
            &RunContext::default(),
        )
        .unwrap()
}

fn denominators(family: &IntegralFamily) -> Vec<Atom> {
    let coordinates = (0..(family.loops.len() * (family.loops.len() + 1) / 2
        + family.loops.len() * family.external.len()))
        .map(|i| symbol!("hepkit_test::z").call(i))
        .collect::<Vec<_>>();
    family
        .propagators
        .iter()
        .map(|d| {
            d.scalar_products
                .iter()
                .zip(&coordinates)
                .fold(d.constant.clone(), |sum, (a, x)| sum + a * x)
        })
        .collect()
}
fn integrand(groups: &[(IntegralFamily, reduction::LinearCombination)]) -> Atom {
    groups.iter().fold(Atom::new(), |sum, (family, terms)| {
        let ds = denominators(family);
        sum + terms.iter().fold(Atom::new(), |sum, (integral, weight)| {
            sum + integral
                .0
                .iter()
                .zip(&ds)
                .fold(weight.clone(), |term, (&power, d)| {
                    term * d.clone().pow(-i64::from(power))
                })
        })
    })
}

#[test]
fn native_graph_roundtrip_weights_and_edge_powers_are_preserved_once() {
    let dot = BUBBLE.replace("digraph massless_bubble {", "digraph massless_bubble {\n num=\"7\"; overall_factor=\"3/2\"; projector=\"5\"; a [num=\"11\"];")
        .replace("id=2, particle=\"phi\"", "id=2, num=\"13\", particle=\"phi\"");
    let graph = GraphIntegral::from_dot(model(), &dot, &kinematics(scalar("s_")))
        .unwrap()
        .with_powers(&BTreeMap::from([(EdgeId(2), 3)]))
        .unwrap();
    assert_eq!(graph.propagator_edges(), [EdgeId(2), EdgeId(3)]);
    assert_eq!(graph.powers(), [3, 1]);
    let exported = graph.diagram().to_dot().unwrap();
    let restored = GraphIntegral::from_dot(model(), &exported, graph.kinematics())
        .unwrap()
        .with_powers(&BTreeMap::from([(EdgeId(2), 3)]))
        .unwrap();
    assert_eq!(restored.diagram().to_dot().unwrap(), exported);
    let point = KinematicPoint(BTreeMap::from([(scalar("s_"), Atom::num(-1))]));
    let result = groups(&graph, &point);
    assert_eq!(result.len(), 1);
    exact(
        &result[0].1[&Integral(vec![3, 1])],
        &Atom::num(Rational::from((15015, 2))),
    );
    exact(&integrand(&result), &integrand(&groups(&restored, &point)));
    assert!(
        GraphIntegral::from_dot(model(), BUBBLE, &kinematics(Atom::num(-1)))
            .unwrap()
            .with_powers(&BTreeMap::from([(EdgeId(0), 2)]))
            .is_err()
    );
}

#[test]
fn scalar_product_identity_controls_the_native_to_rustred_permutation() {
    let k = [
        symbols::loop_momentum().call(0),
        symbols::loop_momentum().call(1),
    ];
    let p = symbols::external_momentum().call(1);
    let kin = kinematics(Atom::num(-1))
        .with_momenta(k.iter().cloned())
        .unwrap();
    let products = [
        kin.scalar_product(&k[0], &k[0]).unwrap(),
        kin.scalar_product(&k[0], &k[1]).unwrap(),
        kin.scalar_product(&k[1], &k[1]).unwrap(),
        kin.scalar_product(&k[0], &p).unwrap(),
        kin.scalar_product(&k[1], &p).unwrap(),
    ];
    let native = NativeFamily::new(k.to_vec(), vec![p], products.to_vec(), &kin).unwrap();
    assert_eq!(
        products
            .iter()
            .map(|a| native
                .scalar_products()
                .iter()
                .position(|b| a == b)
                .unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 3, 2, 4]
    );
    let family = IntegralFamily::from_hepkit(&native, 5, symbol!("feynkit_graph::eps"), 4).unwrap();
    for (i, d) in family.propagators.iter().enumerate() {
        for (j, c) in d.scalar_products.iter().enumerate() {
            exact(c, &Atom::num(i64::from(i == j)));
        }
    }
}

#[test]
fn signed_numerators_and_native_tensor_contractions_reconstruct_the_integrand() {
    let kin = kinematics(Atom::num(-1));
    let plain = GraphIntegral::from_dot(model(), BUBBLE, &kin).unwrap();
    let base = groups(&plain, &KinematicPoint::default());
    let ds = denominators(&base[0].0);
    let signed = GraphIntegral::from_dot(model(), BUBBLE, &kin)
        .unwrap()
        .with_powers(&BTreeMap::from([(EdgeId(2), -2), (EdgeId(3), 2)]))
        .unwrap();
    exact(
        &integrand(&groups(&signed, &KinematicPoint::default())),
        &(ds[0].clone().pow(2) / ds[1].clone().pow(2)),
    );
    let dot = BUBBLE.replace("digraph massless_bubble {", "digraph massless_bubble {\n num=\"gammalooprs::K(0,spenso::mink(D_,mu))*gammalooprs::K(0,spenso::mink(D_,mu))\";");
    let tensor = GraphIntegral::from_dot(model(), &dot, &kin).unwrap();
    exact(
        &integrand(&groups(&tensor, &KinematicPoint::default())),
        &(Atom::one() / &ds[1]),
    );
    assert!(matches!(
        plain.integral_groups(
            &KinematicPoint::default(),
            symbol!("feynkit_graph::eps"),
            4,
            0,
            &RunContext::default()
        ),
        Err(Error::Limit(_))
    ));
}

#[test]
fn structured_projector_routes_edge_momenta_and_applies_scalar_factors_once() {
    let kin = kinematics(Atom::num(-1));
    let plain = GraphIntegral::from_dot(model(), BUBBLE, &kin).unwrap();
    let base = groups(&plain, &KinematicPoint::default());
    let ds = denominators(&base[0].0);
    let vector = SymbolicTensor::infer(scalar("gammalooprs::Q(2,spenso::mink(D_,mu))")).unwrap();
    let diagram = plain
        .diagram()
        .as_ref()
        .clone()
        .with_numerator(vector.clone())
        .unwrap()
        .with_projector(vector)
        .with_numerator_prefactor(SymbolicTensor::infer(Atom::num(7)).unwrap())
        .with_overall_factor(Atom::num(3));
    let graph = GraphIntegral::new(Arc::new(diagram), &kin).unwrap();
    exact(
        &integrand(&groups(&graph, &KinematicPoint::default())),
        &(Atom::num(21) / &ds[1]),
    );
}

#[test]
fn exact_complex_native_bubble_reaches_the_automatic_solver_and_refines() {
    let s = Atom::num(symbolica::domains::float::Complex::new(
        Rational::from(-2),
        Rational::from(1),
    ));
    let graph = GraphIntegral::from_dot(model(), BUBBLE, &kinematics(scalar("s_"))).unwrap();
    let groups = groups(
        &graph,
        &KinematicPoint(BTreeMap::from([(scalar("s_"), s.clone())])),
    );
    assert_eq!(groups.len(), 1);
    exact(&groups[0].0.external_gram[0][0], &s);
    let epsilon = Rational::from((1, 10));
    let options = FlowOptions::default();
    let prepared = PreparedFlow::new_at_epsilon(
        &groups[0].0,
        &[Integral(vec![1, 1])],
        &KinematicPoint::default(),
        &RustRedBackend::default(),
        &options,
        &RunContext::default(),
        &epsilon,
    )
    .unwrap();
    let first = prepared
        .evaluate(
            &epsilon,
            &options,
            &boundary::OneLoopBoundary,
            &RunContext::default(),
        )
        .unwrap();
    let refined = FlowOptions {
        guard_digits: options.guard_digits + 20,
        series_order: options.series_order + 32,
        ..options
    };
    let second = prepared
        .evaluate(
            &epsilon,
            &refined,
            &boundary::OneLoopBoundary,
            &RunContext::default(),
        )
        .unwrap();
    let p = Precision::decimal(80).unwrap();
    let gamma = |x: Rational| p.gamma_real(&p.rational(&x).re).unwrap();
    let expected = p.mul(
        &p.div(
            &p.mul(
                &gamma(epsilon.clone()),
                &p.powi(&gamma(Rational::from(1) - &epsilon), 2),
            ),
            &gamma(Rational::from(2) - &epsilon * &Rational::from(2)),
        ),
        &p.pow(&p.complex(2, -1), &p.neg(&p.rational(&epsilon))),
    );
    assert!(p.close(&first[0], &expected, 20));
    assert!(p.close(&second[0], &expected, 20));
    assert!(p.close(&first[0], &second[0], 20));
}

#[test]
fn native_dimension_and_epsilon_weight_are_applied_before_fitting() {
    let dot = BUBBLE.replace(
        "digraph massless_bubble {",
        "digraph massless_bubble {\n num=\"eps*(spenso::g(spenso::mink(D_,mu),spenso::mink(D_,mu))-2)\";",
    );
    let graph = GraphIntegral::from_dot(model(), &dot, &kinematics(Atom::num(-1))).unwrap();
    let groups = groups(&graph, &KinematicPoint::default());
    exact(
        &groups[0].1[&Integral(vec![1, 1])],
        &scalar("eps*(2-2*eps)"),
    );
    let result = solve_integral_combinations(
        &groups,
        &KinematicPoint::default(),
        0,
        &FlowOptions {
            workers: 2,
            ..Default::default()
        },
        &RustRedBackend::default(),
        &RunContext::default(),
    )
    .unwrap();
    let p = Precision::decimal(80).unwrap();
    assert_eq!(result.verified_digits, Some(20));
    assert!(p.close(&result.coefficients[&0], &p.i(2), 20));
    for (&power, value) in &result.coefficients {
        if power < 0 {
            assert!(p.close(value, &p.zero(), 20));
        }
    }
}

#[test]
fn native_fixed_dimensional_traces_are_literal_and_surviving_mixed_dimensions_are_rejected() {
    let fixed = BUBBLE.replace(
        "digraph massless_bubble {",
        "digraph massless_bubble {\n num=\"spenso::g(spenso::mink(4,mu),spenso::mink(4,mu))\";",
    );
    let graph = GraphIntegral::from_dot(model(), &fixed, &kinematics(Atom::num(-1))).unwrap();
    exact(
        &groups(&graph, &KinematicPoint::default())[0].1[&Integral(vec![1, 1])],
        &Atom::num(4),
    );
    let surviving = BUBBLE.replace(
        "digraph massless_bubble {",
        "digraph massless_bubble {\n num=\"gammalooprs::K(0,spenso::mink(4,mu))*gammalooprs::K(0,spenso::mink(4,mu))\";",
    );
    let graph = GraphIntegral::from_dot(model(), &surviving, &kinematics(Atom::num(-1))).unwrap();
    assert!(matches!(
        graph.integral_groups(
            &KinematicPoint::default(),
            symbol!("feynkit_graph::eps"),
            4,
            1000,
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
    let separated = BUBBLE.replace(
        "digraph massless_bubble {",
        "digraph massless_bubble {\n num=\"gammalooprs::K(0,spenso::mink(4,mu))\"; projector=\"gammalooprs::K(0,spenso::mink(4,mu))\";",
    );
    let graph = GraphIntegral::from_dot(model(), &separated, &kinematics(Atom::num(-1))).unwrap();
    assert!(matches!(
        graph.integral_groups(
            &KinematicPoint::default(),
            symbol!("feynkit_graph::eps"),
            4,
            1000,
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn floating_model_defaults_are_not_used_as_exact_masses() {
    let mut json: serde_json::Value = serde_json::from_str(MODEL).unwrap();
    json["parameters"].as_array_mut().unwrap().push(serde_json::json!({"name":"M","lhablock":null,"lhacode":null,"nature":"external","parameter_type":"real","value":[7.0,0.0],"expression":null}));
    json["particles"][0]["mass"] = "M".into();
    json["propagators"][0]["denominator"] = "(UFO::P(UFO::idx(1,1)))^2-UFO::M^2".into();
    let model = Arc::new(Model::from_json(&json.to_string()).unwrap());
    let graph = GraphIntegral::from_dot(model, BUBBLE, &kinematics(Atom::num(-1))).unwrap();
    let symbolic = groups(&graph, &KinematicPoint::default());
    assert!(
        symbolic[0].0.propagators[0]
            .constant
            .contains(scalar("UFO::M").as_view())
    );
    let exact_groups = groups(
        &graph,
        &KinematicPoint(BTreeMap::from([(
            scalar("UFO::M"),
            Atom::num(Rational::from((3, 2))),
        )])),
    );
    exact(
        &exact_groups[0].0.propagators[0].constant,
        &Atom::num(Rational::from((-9, 4))),
    );
}

#[test]
fn loop_constraints_inexact_substitutions_and_unsupported_model_denominators_fail() {
    let k = symbols::loop_momentum().call(0);
    let kin = kinematics(Atom::num(-1))
        .with_mass_squared(&k, Atom::num(1))
        .unwrap();
    let graph = GraphIntegral::from_dot(model(), BUBBLE, &kin).unwrap();
    assert!(matches!(
        graph.integral_groups(
            &KinematicPoint::default(),
            symbol!("feynkit_graph::eps"),
            4,
            100,
            &RunContext::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    let graph = GraphIntegral::from_dot(model(), BUBBLE, &kinematics(Atom::num(-1))).unwrap();
    let point = KinematicPoint(BTreeMap::from([(
        scalar("s"),
        Atom::num(Float::with_val(80, 1)),
    )]));
    assert!(matches!(
        graph.integral_groups(
            &point,
            symbol!("feynkit_graph::eps"),
            4,
            100,
            &RunContext::default()
        ),
        Err(Error::InvalidInput(_))
    ));
    let mut json: serde_json::Value = serde_json::from_str(MODEL).unwrap();
    json["propagators"][0]["denominator"] = "(UFO::P(UFO::idx(1,1)))^4".into();
    let graph = GraphIntegral::from_dot(
        Arc::new(Model::from_json(&json.to_string()).unwrap()),
        BUBBLE,
        &kinematics(Atom::num(-1)),
    )
    .unwrap();
    assert!(matches!(
        graph.integral_groups(
            &KinematicPoint::default(),
            symbol!("feynkit_graph::eps"),
            4,
            100,
            &RunContext::default()
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn native_projector_rank_twenty_replaces_the_old_rank_fourteen_limit() {
    let dimension = scalar("D");
    let hard = (0..20)
        .map(|i| {
            (0..20)
                .map(|j| {
                    if i < 2 && j < 2 {
                        scalar("k2")
                    } else if i >= 2 && j >= 2 {
                        scalar("q2")
                    } else {
                        scalar("kq")
                    }
                })
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let external = vec![vec![scalar("p2"); 20]; 20];
    let count = (1..18)
        .step_by(2)
        .fold(Atom::one(), |a, n| a * Atom::num(n));
    let denominator = (0..10).fold(Atom::one(), |a, n| a * (&dimension + Atom::num(2 * n)));
    let expected = count * scalar("p2^10*(k2*q2^9+18*kq^2*q2^8)") / denominator;
    let mut projector = tensor::TensorProjector::new(dimension);
    exact(&projector.project(&hard, &external).unwrap(), &expected);
    let mixed = (0..22)
        .map(|i| {
            (0..22)
                .map(|j| {
                    if i < 2 && j < 2 {
                        scalar("k2")
                    } else if i >= 2 && j >= 2 {
                        scalar("q2")
                    } else {
                        scalar("kq")
                    }
                })
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    assert!(matches!(
        projector.project(&mixed, &vec![vec![scalar("p2"); 22]; 22]),
        Err(Error::Limit(_))
    ));
    let mut singular = tensor::TensorProjector::new(Atom::new());
    assert!(matches!(
        singular.project(
            &vec![vec![Atom::one(); 2]; 2],
            &vec![vec![Atom::one(); 2]; 2]
        ),
        Err(Error::Numerical(_))
    ));
}
