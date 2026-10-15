fn original_projector_l0(
    chains: &[Vec<Vec<Atom>>],
    subleading: &[Vec<Atom>],
    inverse: &[Vec<Atom>],
    starts: &[usize],
) -> Vec<Vec<Atom>> {
    let n = subleading.len();
    let m = chains.len();
    let lengths = chains.iter().map(Vec::len).collect::<Vec<_>>();
    let l0 = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| {
                    let left = &inverse[starts[i] + lengths[i] - 1];
                    (0..n)
                        .fold(Atom::new(), |sum, a| {
                            sum + (0..n).fold(Atom::new(), |sum, b| {
                                sum + &left[a] * &subleading[a][b] * &chains[j][0][b]
                            })
                        })
                        .together()
                        .cancel()
                })
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    l0
}
