//! Per-call native graph planning for exact caller-supplied reduction tables.
use super::{LinearCombination, Reduction};
use crate::{Error, Integral, Result};
use linnet::{
    half_edge::{
        HedgeGraph, NoData, NodeIndex, algorithms::DirectionBasis, builder::HedgeGraphBuilder,
        involution::Flow,
    },
    tree::{Forest, child_vec::ChildVecStore},
};
use std::collections::{BTreeMap, BTreeSet};
use symbolica::prelude::*;

type Graph<'a> = HedgeGraph<(), &'a Integral, NoData, Forest<&'a Integral, ChildVecStore<()>>>;

/// Expand all requested roots through one current snapshot of the table.
/// Duplicate targets share an output. Mutable public rules are never cached
/// across calls. Native graph leaves still require declared residual identities.
pub(super) fn expand_many(
    reduction: &Reduction,
    targets: &[Integral],
) -> Result<BTreeMap<Integral, LinearCombination>> {
    if targets.is_empty() {
        return Ok(BTreeMap::new());
    }
    let residuals = reduction.residuals.iter().collect::<BTreeSet<_>>();
    // Native final tables are already flat. A direct row has no dependency
    // graph to traverse; retain its algebraic normalization and authenticate
    // every leaf before discarding zero coefficients.
    if targets.iter().all(|target| {
        reduction.rules.get(target).is_none_or(|terms| {
            terms
                .keys()
                .all(|child| !reduction.rules.contains_key(child))
        })
    }) {
        return targets
            .iter()
            .map(|target| {
                let terms = if let Some(terms) = reduction.rules.get(target) {
                    let mut result = LinearCombination::new();
                    for (child, coefficient) in terms {
                        if !residuals.contains(child) {
                            return Err(Error::IncompleteReduction(format!(
                                "uncovered integral {:?}",
                                child.0
                            )));
                        }
                        let coefficient = coefficient.together().cancel();
                        if !coefficient.is_zero() {
                            result.insert(child.clone(), coefficient);
                        }
                    }
                    result
                } else if residuals.contains(target) {
                    BTreeMap::from([(target.clone(), Atom::one())])
                } else {
                    return Err(Error::IncompleteReduction(format!(
                        "uncovered integral {:?}",
                        target.0
                    )));
                };
                Ok((target.clone(), terms))
            })
            .collect();
    }
    let Plan {
        graph,
        nodes,
        roots,
        order,
        mut uses,
    } = plan(reduction, targets)?;
    let mut expanded = BTreeMap::<NodeIndex, LinearCombination>::new();
    for &node in &order {
        let mut value = LinearCombination::new();
        if let Some(terms) = reduction.rules.get(graph[node]) {
            for (child, coefficient) in terms {
                let child_node = nodes[child];
                let child_value = expanded.get(&child_node).ok_or_else(|| {
                    Error::IncompleteReduction("native reduction graph lost a dependency".into())
                })?;
                for (terminal, weight) in child_value {
                    let old = value.remove(terminal).unwrap_or_default();
                    value.insert(
                        terminal.clone(),
                        (old + coefficient * weight).together().cancel(),
                    );
                }
                uses[child_node.0] -= 1;
                if uses[child_node.0] == 0 {
                    expanded.remove(&child_node);
                }
            }
            value.retain(|_, coefficient| !coefficient.is_zero());
        } else {
            value.insert(graph[node].clone(), Atom::one());
        }
        expanded.insert(node, value);
    }
    roots
        .into_iter()
        .map(|root| {
            expanded
                .remove(&root)
                .map(|value| (graph[root].clone(), value))
                .ok_or_else(|| {
                    Error::IncompleteReduction(
                        "native reduction graph lost a requested root".into(),
                    )
                })
        })
        .collect()
}

/// The existing sparse native dependency plan, separated from coefficient
/// arithmetic so table admission need not expand every declared root.
struct Plan<'a> {
    graph: Graph<'a>,
    nodes: BTreeMap<&'a Integral, NodeIndex>,
    roots: BTreeSet<NodeIndex>,
    order: Vec<NodeIndex>,
    uses: Vec<usize>,
}

pub(super) fn validate(reduction: &Reduction, targets: &[Integral]) -> Result<()> {
    if targets.is_empty() {
        return Ok(());
    }
    let _ = plan(reduction, targets)?;
    Ok(())
}

fn plan<'a>(reduction: &'a Reduction, targets: &'a [Integral]) -> Result<Plan<'a>> {
    let residuals = reduction.residuals.iter().collect::<BTreeSet<_>>();
    let keys = reduction
        .rules
        .iter()
        .flat_map(|(key, terms)| std::iter::once(key).chain(terms.keys()))
        .chain(targets)
        .collect::<BTreeSet<_>>();
    let mut builder = HedgeGraphBuilder::<(), &Integral>::new();
    let nodes = keys
        .into_iter()
        .map(|key| (key, builder.add_node(key)))
        .collect::<BTreeMap<_, _>>();
    for (key, terms) in &reduction.rules {
        for child in terms.keys() {
            builder.add_edge(nodes[key], nodes[child], (), true);
        }
    }
    // The default bitset-per-node store is quadratic in large dependency DAGs;
    // the owner's sparse forest store keeps incidence proportional to edges.
    let graph: Graph<'_> = builder.build();
    let components = graph.strongly_connected_components(DirectionBasis::Underlying);
    let mut component_of = vec![usize::MAX; nodes.len()];
    for (index, component) in components.iter().enumerate() {
        for &node in component {
            component_of[node.0] = index;
        }
    }
    let roots = targets
        .iter()
        .map(|target| nodes[target])
        .collect::<BTreeSet<_>>();
    let mut reached = vec![false; components.len()];
    for root in &roots {
        reached[component_of[root.0]] = true;
    }
    // Native SCCs are sink first. Walking their reverse order propagates root
    // membership without a second bespoke recursive traversal. Unrelated cyclic
    // components are ignored, as they were by the root-based table expander.
    for (index, component) in components.iter().enumerate().rev() {
        if !reached[index] {
            continue;
        }
        if component.len() != 1 {
            return Err(Error::IncompleteReduction("cyclic reduction table".into()));
        }
        let node = component[0];
        for hedge in graph
            .iter_crown(node)
            .filter(|&h| graph.flow(h) == Flow::Source)
        {
            let child = graph.node_id(graph.inv(hedge));
            if child == node {
                return Err(Error::IncompleteReduction("cyclic reduction table".into()));
            }
            reached[component_of[child.0]] = true;
        }
    }
    let order = components
        .iter()
        .enumerate()
        .filter(|(index, _)| reached[*index])
        .map(|(_, component)| component[0])
        .collect::<Vec<_>>();
    let mut uses = vec![0usize; nodes.len()];
    for root in &roots {
        uses[root.0] = 1;
    }
    // Check every reachable leaf before arithmetic, including branches whose
    // incoming coefficient is zero or may cancel against another branch.
    for &node in &order {
        if !reduction.rules.contains_key(graph[node]) && !residuals.contains(graph[node]) {
            return Err(Error::IncompleteReduction(format!(
                "uncovered integral {:?}",
                graph[node].0
            )));
        }
        for hedge in graph
            .iter_crown(node)
            .filter(|&h| graph.flow(h) == Flow::Source)
        {
            let child = graph.node_id(graph.inv(hedge));
            uses[child.0] = uses[child.0]
                .checked_add(1)
                .ok_or_else(|| Error::Limit("reduction dependency use count overflow".into()))?;
        }
    }
    Ok(Plan {
        graph,
        nodes,
        roots,
        order,
        uses,
    })
}
