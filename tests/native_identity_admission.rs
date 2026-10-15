#![cfg(feature = "automatic")]
use serde_json::json;
use std::collections::BTreeMap;
use symbolica::prelude::*;
use symbolica_amflow::{reduction::Reduction, *};
fn i(n: i16) -> Integral {
    Integral(vec![n])
}
fn family() -> IntegralFamily {
    IntegralFamily {
        name: "shared_native_graph_admission".into(),
        loops: vec!["k".into()],
        external: vec![],
        external_gram: vec![],
        propagators: vec![Propagator {
            constant: Atom::num(-1),
            scalar_products: vec![Atom::one()],
        }],
        physical_propagators: 1,
        epsilon: symbol!("admission_regression::epsilon"),
        dimension: 4,
    }
}
fn row(children: &[(i16, i64)]) -> BTreeMap<Integral, Atom> {
    children
        .iter()
        .map(|&(n, c)| (i(n), Atom::num(c)))
        .collect()
}
#[test]
fn admission_matches_the_original_expansion_validation_path() {
    let cases = vec![
        (
            "zero_weight_cycle",
            Reduction {
                rules: BTreeMap::from([(i(1), row(&[(2, 0)])), (i(2), row(&[(1, 1)]))]),
                ..Default::default()
            },
        ),
        (
            "cancelled_paths_to_cycle",
            Reduction {
                rules: BTreeMap::from([
                    (i(1), row(&[(2, 1), (3, -1)])),
                    (i(2), row(&[(4, 1)])),
                    (i(3), row(&[(4, 1)])),
                    (i(4), row(&[(4, 1)])),
                ]),
                ..Default::default()
            },
        ),
        (
            "zero_weight_undeclared_leaf",
            Reduction {
                rules: BTreeMap::from([(i(1), row(&[(2, 0)]))]),
                ..Default::default()
            },
        ),
        (
            "cancelled_paths_to_undeclared_leaf",
            Reduction {
                rules: BTreeMap::from([
                    (i(1), row(&[(2, 1), (3, -1)])),
                    (i(2), row(&[(4, 1)])),
                    (i(3), row(&[(4, 1)])),
                ]),
                ..Default::default()
            },
        ),
        (
            "valid_multilevel",
            Reduction {
                rules: BTreeMap::from([
                    (i(1), row(&[(2, 1), (3, 1)])),
                    (i(2), row(&[(4, 2)])),
                    (i(3), row(&[(4, 3)])),
                ]),
                residuals: vec![i(4)],
                ..Default::default()
            },
        ),
        (
            "valid_cancellation",
            Reduction {
                rules: BTreeMap::from([
                    (i(1), row(&[(2, 1), (3, -1)])),
                    (i(2), row(&[(4, 1)])),
                    (i(3), row(&[(4, 1)])),
                ]),
                residuals: vec![i(4)],
                ..Default::default()
            },
        ),
        (
            "explicit_zero",
            Reduction {
                rules: BTreeMap::from([(i(1), BTreeMap::new())]),
                ..Default::default()
            },
        ),
    ];
    let mut reports = Vec::new();
    let family = family();
    for (name, table) in cases {
        let targets = table
            .rules
            .keys()
            .chain(&table.residuals)
            .cloned()
            .collect::<Vec<_>>();
        let original = table.expand_many(&targets);
        let admitted = ScopedTableBackend::from_table(name, &family, table.clone());
        assert_eq!(original.is_ok(), admitted.is_ok(), "{name}");
        if let Ok(backend) = admitted {
            let retained = backend
                .reduce(&family, &targets, &RunContext::default())
                .unwrap();
            assert_eq!(retained.rules, table.rules);
            let actual = retained.expand_many(&targets).unwrap();
            assert_eq!(actual, original.unwrap());
            reports.push(json!({"name":name,"accepted":true,"identity":backend.identity(),"values":actual.into_iter().map(|(lhs,rhs)|json!({"lhs":lhs.0,"rhs":rhs.into_iter().map(|(i,c)|json!([i.0,c.to_canonical_string()])).collect::<Vec<_>>() })).collect::<Vec<_>>()}));
        } else {
            reports.push(
                json!({"name":name,"accepted":false,"error":format!("{}",original.unwrap_err())}),
            );
        }
    }
    assert_eq!(reports.len(), 7);
}
