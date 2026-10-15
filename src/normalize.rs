//! Bounded exact rational Fuchsian normalization by nilpotent balances.
use crate::{DifferentialSystem, Error, Result, algebra};
use symbolica::prelude::*;

#[cfg(test)]
#[path = "normalize_tests.rs"]
mod nilpotent_candidate_tests;

fn identity(n: usize) -> Vec<Vec<Atom>> {
    (0..n)
        .map(|i| (0..n).map(|j| Atom::num(i64::from(i == j))).collect())
        .collect()
}
fn leading(system: &DifferentialSystem) -> Result<(i64, Vec<Vec<Atom>>)> {
    let order = system
        .matrix
        .iter()
        .flatten()
        .map(|a| crate::frobenius::valuation(a, system.variable))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .unwrap();
    if order >= -1 {
        return Ok((order, identity(system.matrix.len())));
    }
    let matrix = system
        .matrix
        .iter()
        .map(|r| {
            r.iter()
                .map(|a| {
                    Ok(a.series(system.variable, 0, order + 1)
                        .map_err(|e| Error::Unsupported(e.to_string()))?
                        .coefficient(Rational::from(order))
                        .unwrap_or_default())
                })
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    Ok((order, matrix))
}

/// Jordan chains ordered by decreasing length, with each chain starting in ker N.
fn nilpotent_chains(a: &[Vec<Atom>]) -> Result<Vec<Vec<Vec<Atom>>>> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut powers = vec![identity(n)];
    let mut nilpotency_index = None;
    for k in 1..=n {
        powers.push(algebra::matmul(&powers[k - 1], a));
        if powers[k].iter().flatten().all(Atom::is_zero) {
            // This is an exact coefficient-field certificate, never a numerical
            // threshold. Every higher power is zero as well.
            nilpotency_index = Some(k);
            break;
        }
    }
    let Some(nilpotency_index) = nilpotency_index else {
        return Err(Error::Unsupported(
            "irreducible irregular singularity has nonnilpotent leading matrix".into(),
        ));
    };
    let mut basis = Vec::new();
    let mut chains = Vec::new();
    // A longer candidate starts with a zero vector, so the original rank test
    // always rejects it. Keep the order of every potentially accepted candidate.
    for length in (1..=nilpotency_index).rev() {
        for vector in algebra::nullspace(powers[length].clone()) {
            let chain = (0..length)
                .rev()
                .map(|k| {
                    (0..n)
                        .map(|i| {
                            (0..n)
                                .fold(Atom::new(), |s, j| s + &powers[k][i][j] * &vector[j])
                                .together()
                                .cancel()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut trial = basis.clone();
            trial.extend(chain.iter().cloned());
            if algebra::rref(trial.clone()).1.len() == basis.len() + length {
                basis = trial;
                chains.push(chain);
                if basis.len() == n {
                    // No later positive-length candidate can increase rank past n.
                    return Ok(chains);
                }
            }
        }
    }
    if basis.len() != n {
        return Err(Error::Numerical(
            "nilpotent Jordan chains do not span the system".into(),
        ));
    }
    Ok(chains)
}

/// Contract the selected inverse rows, subleading matrix and first chain vectors
/// with the existing native matrix owner. Their order is the Jordan-chain order.
fn projector_l0(
    chains: &[Vec<Vec<Atom>>],
    subleading: &[Vec<Atom>],
    inverse: &[Vec<Atom>],
    starts: &[usize],
) -> Vec<Vec<Atom>> {
    let left = chains
        .iter()
        .zip(starts)
        .map(|(chain, &start)| inverse[start + chain.len() - 1].clone())
        .collect::<Vec<_>>();
    let right = (0..subleading.len())
        .map(|i| chains.iter().map(|chain| chain[0][i].clone()).collect())
        .collect::<Vec<Vec<_>>>();
    algebra::matmul(&algebra::matmul(&left, subleading), &right)
}

/// AMFlow's ReduceL0/FindProjector algorithm: mix equal-position Jordan
/// vectors until a balance can remove one nilpotent chain. All ranks and
/// nullspaces are computed in the exact rational-function coefficient field.
fn refined_projector(
    chains: &[Vec<Vec<Atom>>],
    subleading: &[Vec<Atom>],
) -> Result<Vec<Vec<Atom>>> {
    let n = subleading.len();
    let m = chains.len();
    let lengths = chains.iter().map(Vec::len).collect::<Vec<_>>();
    let r = lengths.iter().filter(|&&length| length > 1).count();
    let vectors = chains.iter().flatten().collect::<Vec<_>>();
    let jordan = (0..n)
        .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
        .collect::<Vec<Vec<_>>>();
    let inverse = algebra::inverse(&jordan)?;
    let mut offset = 0;
    let starts = lengths
        .iter()
        .map(|&length| {
            let start = offset;
            offset += length;
            start
        })
        .collect::<Vec<_>>();
    let mut l0 = projector_l0(chains, subleading, &inverse, &starts);
    let mut delta = vec![vec![Atom::new(); m]; m];
    let mut selected = Vec::new();
    loop {
        let rows = (0..m).filter(|i| !selected.contains(i)).collect::<Vec<_>>();
        let mut choice = None;
        for &column in &rows {
            let prefix = rows
                .iter()
                .map(|&i| l0[i][..=column].to_vec())
                .collect::<Vec<_>>();
            let null = algebra::nullspace(prefix);
            if let Some(vector) = null.into_iter().find(|v| !v[column].is_zero()) {
                choice = Some((column, vector));
                break;
            }
        }
        let Some((column, vector)) = choice else {
            return Err(Error::Unsupported(
                "Moser projector has no dependent column; singularity may be irreducibly irregular"
                    .into(),
            ));
        };
        let mut right = identity(m);
        let mut left = identity(m);
        let mut d0 = vec![vec![Atom::new(); m]; m];
        for j in 0..column {
            d0[j][column] = (&vector[j] / &vector[column]).together().cancel();
            right[j][column] = d0[j][column].clone();
            if lengths[j] == lengths[column] {
                left[j][column] = -d0[j][column].clone();
            }
        }
        l0 = algebra::matmul(&algebra::matmul(&left, &l0), &right);
        let product = algebra::matmul(&delta, &d0);
        for i in 0..m {
            for j in 0..m {
                delta[i][j] = (&delta[i][j] + &d0[i][j] + &product[i][j])
                    .together()
                    .cancel();
            }
        }
        selected.push(column);
        if column < r {
            break;
        }
        if selected.len() == m {
            return Err(Error::Unsupported(
                "Moser projector contains no nontrivial Jordan chain".into(),
            ));
        }
    }
    let mut mixing = identity(n);
    for i in 0..m {
        for j in i + 1..m {
            for k in 0..lengths[j] {
                mixing[starts[i] + k][starts[j] + k] = delta[i][j].clone();
            }
        }
    }
    let basis = algebra::matmul(&jordan, &mixing);
    let inverse = algebra::inverse(&basis)?;
    Ok((0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    selected
                        .iter()
                        .fold(Atom::new(), |sum, &k| {
                            sum + &basis[i][starts[k]] * &inverse[starts[k]][j]
                        })
                        .together()
                        .cancel()
                })
                .collect()
        })
        .collect())
}

impl DifferentialSystem {
    /// Return (Fuchsian system, T) with old_y=T*new_y. First attempt diagonal
    /// shearing, then exact nilpotent rank-reducing balances. A search that does
    /// not find a decreasing balance reports an unsupported normalization.
    pub fn fuchsian_form(&self, max_balances: usize) -> Result<(Self, Vec<Vec<Atom>>)> {
        self.validate()?;
        let n = self.matrix.len();
        let x = Atom::var(self.variable);
        let mut system = self.clone();
        let mut transformation = identity(n);
        for iteration in 0..=max_balances {
            if let Ok((normal, shifts)) = system.diagonal_fuchsian_form() {
                let diagonal = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                if i == j {
                                    x.clone().pow(shifts[i])
                                } else {
                                    Atom::new()
                                }
                            })
                            .collect()
                    })
                    .collect::<Vec<_>>();
                return Ok((normal, algebra::matmul(&transformation, &diagonal)));
            }
            if iteration == max_balances {
                break;
            }
            let (order, coefficient) = leading(&system)?;
            let rank = algebra::rref(coefficient.clone()).1.len();
            let chains = nilpotent_chains(&coefficient)?;
            let vectors = chains.iter().flatten().collect::<Vec<_>>();
            let jordan = (0..n)
                .map(|i| vectors.iter().map(|v| v[i].clone()).collect())
                .collect::<Vec<Vec<_>>>();
            let inverse = algebra::inverse(&jordan)?;
            let mut offset = 0;
            let starts = chains
                .iter()
                .map(|c| {
                    let start = offset;
                    offset += c.len();
                    start
                })
                .collect::<Vec<_>>();
            // Large systems go directly to projector refinement instead of
            // enumerating exponentially many Jordan-chain subsets.
            let subsets = 1usize
                .checked_shl(chains.len() as u32)
                .filter(|&n| n <= 1024)
                .unwrap_or(1);
            let mut best = None;
            for mask in 1..subsets {
                let chosen = starts
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &j)| (mask & (1 << k) != 0).then_some(j))
                    .collect::<Vec<_>>();
                let projector = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                chosen
                                    .iter()
                                    .fold(Atom::new(), |s, &k| s + &jordan[i][k] * &inverse[k][j])
                                    .together()
                                    .cancel()
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                let balance = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                (Atom::num(i64::from(i == j)) - &projector[i][j]
                                    + &projector[i][j] / &x)
                                    .together()
                                    .cancel()
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                let candidate = system.change_basis(&balance)?;
                let (new_order, new_leading) = leading(&candidate)?;
                let new_rank = if new_order >= -1 {
                    0
                } else {
                    algebra::rref(new_leading).1.len()
                };
                let score = (-new_order, new_rank);
                if score < (-order, rank) && best.as_ref().is_none_or(|(old, _, _)| score < *old) {
                    best = Some((score, candidate, balance));
                }
            }
            if best.is_none() {
                let subleading = system
                    .matrix
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|a| {
                                Ok(a.series(system.variable, 0, order + 2)
                                    .map_err(|e| Error::Unsupported(e.to_string()))?
                                    .coefficient(Rational::from(order + 1))
                                    .unwrap_or_default())
                            })
                            .collect::<Result<_>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                let projector = refined_projector(&chains, &subleading)?;
                let balance = (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                (Atom::num(i64::from(i == j)) - &projector[i][j]
                                    + &projector[i][j] / &x)
                                    .together()
                                    .cancel()
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
                let candidate = system.change_basis(&balance)?;
                let (new_order, new_leading) = leading(&candidate)?;
                let new_rank = if new_order >= -1 {
                    0
                } else {
                    algebra::rref(new_leading).1.len()
                };
                if (-new_order, new_rank) < (-order, rank) {
                    best = Some(((-new_order, new_rank), candidate, balance));
                }
            }
            let Some((_, next, balance)) = best else {
                return Err(Error::Unsupported(
                    "Moser projector did not reduce the singular rank".into(),
                ));
            };
            transformation = algebra::matmul(&transformation, &balance);
            system = next;
        }
        Err(Error::Limit(
            "Fuchsian normalization balance budget exhausted".into(),
        ))
    }
}
