use super::*;

fn dense_reference(a: &[Vec<Atom>], b: &[Vec<Atom>]) -> Vec<Vec<Atom>> {
    if a.is_empty() {
        return Vec::new();
    }
    let columns = b[0].len();
    if columns == 0 {
        return vec![Vec::new(); a.len()];
    }
    let left = Matrix::from_nested_vec(a.to_vec(), exact_atom_field()).unwrap();
    let right = Matrix::from_nested_vec(b.to_vec(), exact_atom_field()).unwrap();
    (&left * &right)
        .into_vec()
        .chunks(columns)
        .map(|row| row.iter().map(|a| a.together().cancel()).collect())
        .collect()
}

#[test]
fn sparse_representation_preserves_dense_rectangular_symbolic_and_zero_products() {
    assert_eq!(matmul(&[], &[]), Vec::<Vec<Atom>>::new());
    assert_eq!(
        matmul(&[vec![Atom::one()]], &[vec![]]),
        vec![Vec::<Atom>::new()]
    );
    let t = Atom::var(symbol!("sparse_representation_tests::t"));
    let identity = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| Atom::num(i64::from(i == j)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dense = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| {
                    (&t + Atom::num((i + j + 1) as i64)) / (&t + Atom::num((i + 2 * j + 2) as i64))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(structurally_sparse(&identity));
    assert!(!structurally_sparse(&dense));
    let mut rectangular = vec![vec![Atom::new(); 5]; 3];
    rectangular[0][1] = t.clone();
    rectangular[1][4] = Atom::num((1, 3));
    rectangular[2][0] = Atom::one();
    assert!(structurally_sparse(&rectangular));
    let mut cancellation_left = vec![vec![Atom::new(); 5]; 4];
    cancellation_left[0][0] = t.clone();
    cancellation_left[0][3] = Atom::one();
    let mut cancellation_right = vec![vec![Atom::new(); 3]; 5];
    cancellation_right[0][2] = Atom::one();
    cancellation_right[3][2] = -t.clone();
    let cancelled = matmul(&cancellation_left, &cancellation_right);
    assert!(cancelled.iter().flatten().all(Atom::is_zero));
    for (a, b) in [
        (identity.clone(), dense.clone()),
        (dense.clone(), identity.clone()),
        (dense.clone(), dense.clone()),
        (rectangular, dense),
        (cancellation_left, cancellation_right),
        (vec![vec![Atom::new(); 5]; 3], identity),
    ] {
        let expected = dense_reference(&a, &b);
        let actual = matmul(&a, &b);
        assert_eq!(expected.len(), actual.len());
        for (expected_row, actual_row) in expected.iter().zip(&actual) {
            assert_eq!(expected_row.len(), actual_row.len());
            for (expected, actual) in expected_row.iter().zip(actual_row) {
                assert!((expected - actual).together().cancel().is_zero());
                assert_eq!(expected.is_zero(), actual.is_zero());
            }
        }
    }
}
