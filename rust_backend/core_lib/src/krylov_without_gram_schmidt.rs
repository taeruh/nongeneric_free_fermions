use std::{collections::HashMap, sync::Mutex};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    fendley::Fendley,
    paulis::{Pauli, PauliSum},
};

#[derive(Debug, Clone)]
pub struct GeneratorsWithoutGramSchmidt {
    pub n: usize,
    pub eta_vector_to_pauli_map: Vec<Pauli>,
    pub etas: Vec<PauliSum>,
    pub eta_vectors: Vec<Vec<f64>>,
    pub eta_renormalisation_factor: Vec<f64>,
    pub eta_normalisation_factors: Vec<f64>,
    pub num_generators: usize,
    pub eta_currents: Vec<PauliSum>,
}

impl GeneratorsWithoutGramSchmidt {
    pub fn new(
        simplicial_mode: (f64, Pauli),
        fendley: Fendley,
        max_eta: usize,
        renormalise: bool,
        eta_normalisation_factor: f64,
    ) -> Self {
        let n = simplicial_mode.1.n();
        let mut eta_vector_to_pauli_map = vec![simplicial_mode.1.clone()];
        let (mut etas, mut eta_vectors, mut eta_renormalisation_factor) = if renormalise {
            (
                vec![PauliSum(vec![(1.0, simplicial_mode.1)])],
                vec![vec![1.0]],
                vec![1.0 / simplicial_mode.0],
            )
        } else {
            (
                vec![PauliSum(vec![(simplicial_mode.0, simplicial_mode.1)])],
                vec![vec![simplicial_mode.0]],
                Vec::new(),
            )
        };
        let mut eta_normalisation_factors = vec![1.0];

        for index in 0..max_eta {
            let last_eta = &etas[index];
            let mut eta = PauliSum(vec![]);
            let mut vector = vec![0.0; eta_vector_to_pauli_map.len()];
            for (weight, op) in last_eta.0.iter() {
                for (ham_weight, ham_op) in fendley
                    .hamiltonian
                    .weights
                    .iter()
                    .zip(fendley.hamiltonian.operators.iter())
                {
                    if ham_op.symplectic_inner_product(op) {
                        let comm_weight = weight * ham_weight * eta_normalisation_factor;
                        let mut comm_op = ham_op.multiply_as_paulis(op);
                        comm_op.add_to_phase(1);

                        let mut already_in_vectors = false;
                        for (i, vec_op) in eta_vector_to_pauli_map.iter().enumerate() {
                            if comm_op.is_proportional_to(vec_op) {
                                let sign_phase = comm_op.phase_difference(vec_op);
                                assert!(sign_phase == 0 || sign_phase == 2);
                                vector[i] += comm_weight
                                    * (-1.0_f64).powi((sign_phase / 2) as i32);
                                already_in_vectors = true;
                                break;
                            }
                        }
                        if !already_in_vectors {
                            eta_vector_to_pauli_map.push(comm_op.clone());
                            vector.push(comm_weight);
                        }
                        eta.single_add(comm_weight, &comm_op);
                    }
                }
            }
            eta.remove_zero_weights();
            if renormalise {
                let norm = eta.0.iter().map(|(w, _)| w.powi(2)).sum::<f64>().sqrt()
                    / (eta.0.len() as f64);
                eta.multiply_with_float(1.0 / norm);
                vector.iter_mut().for_each(|v| *v /= norm);
                let last_eta_renorm_factor = eta_renormalisation_factor.last().unwrap();
                eta_renormalisation_factor.push(last_eta_renorm_factor / norm);
            }
            etas.push(eta);
            eta_vectors.push(vector);
            let last_eta_normalisation_factor = eta_normalisation_factors.last().unwrap();
            eta_normalisation_factors
                .push(last_eta_normalisation_factor * eta_normalisation_factor);
            println!("calculated eta {}", index + 1);
        }

        Self {
            num_generators: etas.len(),
            n,
            eta_vector_to_pauli_map,
            etas,
            eta_vectors,
            eta_renormalisation_factor,
            eta_normalisation_factors,
            eta_currents: Vec::new(),
        }
    }

    pub fn init_eta_currents(&mut self) {
        let currents = Mutex::new(HashMap::new());
        (0..self.num_generators).into_par_iter().for_each(|l| {
            if l % 2 == 0 {
                currents.lock().unwrap().insert(l, PauliSum(vec![]));
                return;
            }
            let mut current = PauliSum(vec![]);
            for k in 0..l {
                let l_k = l - k;
                if k == l_k {
                    continue;
                }
                let mut prod = self.etas[k].multiply(&self.etas[l_k]);
                let mut prod_inv = self.etas[l_k].multiply(&self.etas[k]);
                if !self.eta_renormalisation_factor.is_empty() {
                    let factor = 1.0
                        / (self.eta_renormalisation_factor[k]
                            * self.eta_renormalisation_factor[l_k]);
                    prod.multiply_with_float(factor);
                    prod_inv.multiply_with_float(factor);
                }
                let commutator = prod.subtract(&prod_inv);
                if k % 2 == 1 {
                    current = current.subtract(&commutator);
                } else {
                    current = current.add(&commutator);
                }
            }
            current.remove_zero_weights();
            current.multiply_with_one_imag_unit();
            for (_, op) in current.0.iter() {
                assert!(op.get_hermitian_phase() == 0 || op.get_hermitian_phase() == 2);
            }
            println!("calculated current for eta {l}");
            currents.lock().unwrap().insert(l, current);
        });
        let mut currents = currents.into_inner().unwrap();
        self.eta_currents = (0..self.num_generators)
            .map(|l| currents.remove(&l).unwrap())
            .collect();
    }
}
