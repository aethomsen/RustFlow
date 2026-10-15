use super::*;

// The reference is the unchanged pre-optimization owner body, renamed only.
include!("normalize_original_for_test.rs");

fn jordan(lengths: &[usize]) -> Vec<Vec<Atom>> {
    let n = lengths.iter().sum();
    let mut matrix = vec![vec![Atom::new(); n]; n];
    let mut offset = 0;
    for length in lengths {
        for column in offset + 1..offset + length {
            matrix[column - 1][column] = Atom::one();
        }
        offset += length;
    }
    matrix
}

fn verify_chains(matrix: &[Vec<Atom>], chains: &[Vec<Vec<Atom>>]) {
    let n = matrix.len();
    let lengths = chains.iter().map(Vec::len).collect::<Vec<_>>();
    assert_eq!(lengths.iter().sum::<usize>(), n);
    let columns = chains.iter().flatten().collect::<Vec<_>>();
    let basis = (0..n)
        .map(|row| columns.iter().map(|column| column[row].clone()).collect())
        .collect::<Vec<Vec<_>>>();
    assert_eq!(algebra::rref(basis.clone()).1.len(), n);
    let left = algebra::matmul(matrix, &basis);
    let right = algebra::matmul(&basis, &jordan(&lengths));
    assert!(
        left.iter()
            .flatten()
            .zip(right.iter().flatten())
            .all(|(a, b)| (a - b).together().cancel().is_zero())
    );
}

#[test]
fn exact_candidate_order_matches_original_for_dense_nilpotent_families() {
    let partitions = [
        vec![1],
        vec![2],
        vec![1, 1],
        vec![3],
        vec![2, 1],
        vec![1, 1, 1],
        vec![4],
        vec![3, 1],
        vec![2, 2],
        vec![2, 1, 1],
        vec![5],
        vec![4, 1],
        vec![3, 2],
        vec![2, 2, 1],
        vec![6],
        vec![4, 2],
        vec![3, 3],
        vec![2, 2, 2],
        vec![1, 1, 1, 1, 1, 1],
    ];
    for lengths in partitions {
        let n = lengths.iter().sum();
        let mut change = identity(n);
        for (i, row) in change.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate().take(i) {
                *value = Atom::num(((i + 2) * (j + 1)) as i64);
            }
        }
        let matrix = algebra::matmul(
            &algebra::matmul(&change, &jordan(&lengths)),
            &algebra::inverse(&change).unwrap(),
        );
        let old = original_nilpotent_chains(&matrix).unwrap();
        let optimized = nilpotent_chains(&matrix).unwrap();
        assert_eq!(old, optimized, "chain ordering changed for {lengths:?}");
        verify_chains(&matrix, &optimized);
    }
}

#[test]
fn symbolic_exact_zero_and_nonnilpotent_cases_keep_original_behavior() {
    let e = Atom::var(symbol!("normalization_test::epsilon"));
    let matrix = vec![vec![e.clone(), e.clone()], vec![-e.clone(), -e]];
    let old = original_nilpotent_chains(&matrix).unwrap();
    let optimized = nilpotent_chains(&matrix).unwrap();
    assert_eq!(old, optimized);
    verify_chains(&matrix, &optimized);
    for matrix in [
        identity(2),
        vec![
            vec![Atom::new(), Atom::one()],
            vec![Atom::new(), Atom::one()],
        ],
    ] {
        assert!(matches!(
            original_nilpotent_chains(&matrix),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            nilpotent_chains(&matrix),
            Err(Error::Unsupported(_))
        ));
    }
    let empty: Vec<Vec<Atom>> = vec![];
    assert_eq!(
        original_nilpotent_chains(&empty).unwrap(),
        nilpotent_chains(&empty).unwrap()
    );
    let zero = vec![vec![Atom::new(); 4]; 4];
    assert_eq!(
        original_nilpotent_chains(&zero).unwrap(),
        nilpotent_chains(&zero).unwrap()
    );
}

#[test]
fn large_square_zero_input_uses_an_exact_short_chain_certificate() {
    for n in [97, 127] {
        let mut matrix = vec![vec![Atom::new(); n]; n];
        matrix[0][0] = Atom::one();
        matrix[0][1] = -Atom::one();
        matrix[1][0] = Atom::one();
        matrix[1][1] = -Atom::one();
        assert!(
            algebra::matmul(&matrix, &matrix)
                .iter()
                .flatten()
                .all(Atom::is_zero)
        );
        let chains = nilpotent_chains(&matrix).unwrap();
        assert_eq!(chains.iter().filter(|c| c.len() == 2).count(), 1);
        assert!(chains.iter().all(|c| c.len() <= 2));
        verify_chains(&matrix, &chains);
    }
}

include!("normalize_l0_original_for_test.rs");
#[test]
fn native_projector_contraction_matches_original_quadratic_forms() {
    let t = Atom::var(symbol!("normalization_l0_test::t"));
    for lengths in [vec![2, 1], vec![2, 2], vec![3, 1, 1], vec![2, 2, 1, 1]] {
        let n = lengths.iter().sum();
        let mut change = identity(n);
        for (i, row) in change.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate().take(i) {
                *value = &t * Atom::num((i + j + 1) as i64);
            }
        }
        let matrix = algebra::matmul(
            &algebra::matmul(&change, &jordan(&lengths)),
            &algebra::inverse(&change).unwrap(),
        );
        let chains = nilpotent_chains(&matrix).unwrap();
        let columns = chains.iter().flatten().collect::<Vec<_>>();
        let basis = (0..n)
            .map(|i| columns.iter().map(|v| v[i].clone()).collect())
            .collect::<Vec<Vec<_>>>();
        let inverse = algebra::inverse(&basis).unwrap();
        let mut offset = 0;
        let starts = chains
            .iter()
            .map(|c| {
                let start = offset;
                offset += c.len();
                start
            })
            .collect::<Vec<_>>();
        let cancellation =
            (&t + Atom::one()).pow(2) - t.clone().pow(2) - Atom::num(2) * &t - Atom::one();
        let dense = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| Atom::num(((i + 1) as i64, (j + 2) as i64)))
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let symbolic = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (&t * Atom::num((i + 1) as i64) + Atom::num((j + 2) as i64))
                            / (&t + Atom::num((i + j + 3) as i64))
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        let cancelling = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        &cancellation * Atom::num(i as i64 - j as i64)
                            + Atom::num(i64::from(i == j))
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        for b in [vec![vec![Atom::new(); n]; n], dense, symbolic, cancelling] {
            let original = original_projector_l0(&chains, &b, &inverse, &starts);
            let native = projector_l0(&chains, &b, &inverse, &starts);
            assert_eq!(original.len(), native.len());
            for (a, b) in original.iter().flatten().zip(native.iter().flatten()) {
                assert!(
                    (a - b).together().cancel().is_zero(),
                    "Native contraction changed an exact entry"
                );
                assert_eq!(
                    a.is_zero(),
                    b.is_zero(),
                    "Moser exact-zero selection changed"
                );
            }
        }
    }
}
