use crate::{
    hamiltonian::Hamiltonian,
    paulis::{Pauli, PauliSum},
};

#[derive(Debug, Clone)]
pub struct Fendley {
    num_triangles: usize,
    cap_m: usize,
    n: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
    ops: Vec<Pauli>,
    weights: Vec<f64>,
    hamiltonian: Hamiltonian,
    example_simplicial_modes: Vec<(Pauli, usize)>,
    current_alpha: Vec<f64>,
}

impl Fendley {
    pub fn new(num_triangles: usize, alpha: f64, beta: f64, gamma: f64) -> Self {
        let cap_m = 3 * num_triangles;
        let n = cap_m + 2;

        let mut ops = Vec::new();
        let mut weights = Vec::new();
        for i in 0..num_triangles {
            for j in 0..3 {
                let parameter = match j {
                    0 => alpha,
                    1 => beta,
                    2 => gamma,
                    _ => unreachable!(),
                };
                if parameter != 0.0 {
                    let op = Pauli::from_indices(
                        n,
                        &[3 * i + j, 3 * i + (j + 1) % 3],
                        &[3 * i + (j + 2) % 3],
                        0,
                    );
                    ops.push(op);
                    weights.push(parameter);
                }
            }
        }

        let hamiltonian = Hamiltonian::new(weights.clone(), ops.clone());

        // not exhaustive
        let example_simplicial_modes = vec![
            // { connect to IIXZZ
            (Pauli::from_indices(n, &[], &[0], 0), 1),
            (Pauli::from_indices(n, &[0, 1, 2], &[], 0), 1),
            // } { connect to IIXZZ,
            //                IXZZI
            (Pauli::from_indices(n, &[0, 1], &[1], 3), 2),
            (Pauli::from_indices(n, &[0, 1, 2, 3], &[], 0), 2),
            // } { connect to IIXZZ, IXZZI, XZZII
            (Pauli::from_indices(n, &[2], &[2], 3), 3),
            (Pauli::from_indices(n, &[0, 1, 2, 3, 4], &[], 0), 3),
            // }
        ];

        Self {
            num_triangles,
            cap_m,
            n,
            alpha,
            beta,
            gamma,
            ops,
            weights,
            hamiltonian,
            example_simplicial_modes,
            current_alpha: Vec::new(),
        }
    }

    pub fn extend_with_currents(&mut self, currents: &[PauliSum], current_alpha: &[f64]) {
        assert_eq!(currents.len(), current_alpha.len());
        self.current_alpha = current_alpha.to_vec();

        for (alpha, current) in current_alpha.iter().zip(currents.iter()) {
            for (w, op) in &current.0 {
                let mut already_in = false;
                for (i, self_op) in self.ops.iter().enumerate() {
                    if op.is_proportional_to(self_op) {
                        let phase = op.phase_difference(self_op);
                        assert!(phase == 0 || phase == 2);
                        self.weights[i] += alpha * w * (-1.0f64).powi((phase / 2) as i32);
                        already_in = true;
                        break;
                    }
                }
                if !already_in {
                    self.ops.push(op.clone());
                    self.weights.push(alpha * w);
                }
            }
        }

        let mut to_remove = Vec::new();
        for (i, w) in self.weights.iter().enumerate() {
            if w.abs() < 1e-10 {
                to_remove.push(i);
            }
        }
        for i in to_remove.into_iter().rev() {
            self.weights.swap_remove(i);
            self.ops.swap_remove(i);
        }

        self.hamiltonian = Hamiltonian::new(self.weights.clone(), self.ops.clone());
    }
}
