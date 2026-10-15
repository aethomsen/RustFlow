fn original_nilpotent_chains(a: &[Vec<Atom>]) -> Result<Vec<Vec<Vec<Atom>>>> {
    let n = a.len();
    let mut powers = vec![identity(n)];
    for k in 1..=n {
        powers.push(algebra::matmul(&powers[k - 1], a));
    }
    if powers[n].iter().flatten().any(|a| !a.is_zero()) {
        return Err(Error::Unsupported(
            "irreducible irregular singularity has nonnilpotent leading matrix".into(),
        ));
    }
    let mut basis = Vec::new();
    let mut chains = Vec::new();
    for length in (1..=n).rev() {
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
