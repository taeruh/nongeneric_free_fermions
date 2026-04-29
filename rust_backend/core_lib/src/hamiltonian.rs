use crate::paulis::Pauli;

#[derive(Debug, Clone)]
pub struct Hamiltonian {
    pub weights: Vec<f64>,
    pub operators: Vec<Pauli>,
    pub pauli_l1_norm: f64,
    pub pauli_l2_norm: f64,
    pub num_ops: usize,
    pub n: usize,
}

impl Hamiltonian {
    pub fn new(weights: Vec<f64>, ops: Vec<Pauli>) -> Self {
        assert_eq!(weights.len(), ops.len());
        let num_ops = ops.len();
        let n = if num_ops == 0 { 1 } else { ops[0].n() };

        // note that the proportionality check also implicitly checks that all ops have the
        // same n, since otherwise they cannot be proportional (the method would raise an
        // error)
        let (has_prop_terms, _) = has_proportional_terms(&ops);
        if has_prop_terms {
            panic!("hamiltonian has proportional terms");
        }
        let (has_non_herm_terms, _) = has_non_hermitian_terms(&ops);
        if has_non_herm_terms {
            panic!("hamiltonian has non-hermitian terms");
        }

        let pauli_l1_norm = weights.iter().map(|w| w.abs()).sum();
        let pauli_l2_norm = weights.iter().map(|w| w.abs().powi(2)).sum::<f64>().sqrt();

        Self {
            weights,
            operators: ops,
            pauli_l1_norm,
            pauli_l2_norm,
            num_ops,
            n,
        }
    }
}

fn has_proportional_terms(ops: &[Pauli]) -> (bool, Option<Vec<(usize, usize)>>) {
    let mut proportional_pairs = Vec::new();
    for i in 0..ops.len() {
        for j in (i + 1)..ops.len() {
            if ops[i].is_proportional_to(&ops[j]) {
                proportional_pairs.push((i, j));
            }
        }
    }
    if !proportional_pairs.is_empty() {
        (true, Some(proportional_pairs))
    } else {
        (false, None)
    }
}
fn has_non_hermitian_terms(ops: &[Pauli]) -> (bool, Option<Vec<usize>>) {
    let mut non_hermitian_indices = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if !matches!(op.get_hermitian_phase(), 0 | 2) {
            non_hermitian_indices.push(i);
        }
    }
    if !non_hermitian_indices.is_empty() {
        (true, Some(non_hermitian_indices))
    } else {
        (false, None)
    }
}
