use std::fmt::{Display, Formatter};

use bitvec::vec::BitVec;

#[derive(Clone, Debug)]
pub struct Pauli {
    n: usize,
    u: BitVec,
    l: BitVec,
    phase: u8,
}

impl Pauli {
    pub fn new(n: usize, u: &[bool], l: &[bool], phase: u8) -> Self {
        debug_assert_eq!(u.len(), n);
        debug_assert_eq!(l.len(), n);
        Self {
            n,
            u: BitVec::from_iter(u.iter()),
            l: BitVec::from_iter(l.iter()),
            phase,
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn phase(&self) -> u8 {
        self.phase
    }

    pub fn set_phase(&mut self, phase: u8) {
        self.phase = phase;
    }

    pub fn add_to_phase(&mut self, phase_diff: u8) {
        self.phase = (self.phase + phase_diff) % 4;
    }

    pub fn identity(n: usize) -> Self {
        Self {
            n,
            u: BitVec::repeat(false, n),
            l: BitVec::repeat(false, n),
            phase: 0,
        }
    }

    pub fn from_indices(
        n: usize,
        u_trues: &[usize],
        l_trues: &[usize],
        phase: u8,
    ) -> Self {
        let mut u = BitVec::repeat(false, n);
        let mut l = BitVec::repeat(false, n);
        for &i in u_trues {
            u.set(i, true);
        }
        for &i in l_trues {
            l.set(i, true);
        }
        Self { n, u, l, phase }
    }

    /// the phase when writing the pauli as X/Y/Z string
    pub fn get_hermitian_phase(&self) -> u8 {
        let mut phase = self.phase;
        // for i in 0..self.n {
        //     if self.u[i] && self.l[i] {
        //         // multiply by 1 = i * -i and take i into the phase and make ZX to Y
        //         phase = (phase + 1) % 4;
        //     }
        // }
        let ul = self.u.clone() & &self.l;
        phase = (phase + (ul.count_ones() % 4) as u8) % 4;
        phase
    }

    /// in the string we order from right to left, i.e., p_n-1, ... p_0 (as in a
    /// bitvector)
    pub fn to_pretty_string_opt(&self, flip: bool, with_phase: bool) -> String {
        let mut pauli_str = String::new();
        let iter: Box<dyn Iterator<Item = usize>> = if flip {
            Box::new(0..self.n)
        } else {
            Box::new((0..self.n).rev())
        };
        for i in iter {
            if self.u[i] && self.l[i] {
                pauli_str.push('Y');
            } else if self.u[i] {
                pauli_str.push('Z');
            } else if self.l[i] {
                pauli_str.push('X');
            } else {
                pauli_str.push('I');
            }
        }
        if with_phase {
            let phase = self.get_hermitian_phase();
            pauli_str.push_str(", ");
            match phase {
                0 => pauli_str.push_str("+1"),
                1 => pauli_str.push_str("+i"),
                2 => pauli_str.push_str("-1"),
                _ => pauli_str.push_str("-i"),
            }
        }
        pauli_str
    }

    pub fn to_pretty_string(&self) -> String {
        self.to_pretty_string_opt(false, true)
    }

    pub fn is_proportional_to(&self, other: &Self) -> bool {
        (self.n == other.n) && (self.u == other.u) && (self.l == other.l)
    }

    /// assuming self and other are proportional, returns the phase difference between
    /// self and other, more precisely: self = (i^phase_difference) * other
    pub fn phase_difference(&self, other: &Self) -> u8 {
        debug_assert!(self.is_proportional_to(other));
        (self.phase + 4 - other.phase) % 4
    }

    pub fn symplectic_inner_product(&self, other: &Self) -> bool {
        debug_assert!(self.n == other.n);
        // let mut ip = false;
        // for i in 0..self.n {
        //     ip ^= (self.l[i] && other.u[i]) ^ (self.u[i] && other.l[i]);
        // }
        // ip
        let lu = self.l.clone() & &other.u;
        let ul = self.u.clone() & &other.l;
        (ul.count_ones() + lu.count_ones()) % 2 == 1
    }

    pub fn multiply_as_paulis(&self, other: &Self) -> Self {
        debug_assert!(self.n == other.n);
        let new_l = self.l.clone() ^ &other.l;
        let new_u = self.u.clone() ^ &other.u;
        let mut new_phase = (self.phase + other.phase) % 4;
        // for i in 0..self.n {
        //     if self.l[i] && other.u[i] {
        //         new_phase = (new_phase + 2) % 4;
        //     }
        // }
        let lu = self.u.clone() & &other.l;
        if lu.count_ones() % 2 == 1 {
            new_phase = (new_phase + 2) % 4;
        }
        Self {
            n: self.n,
            u: new_u,
            l: new_l,
            phase: new_phase,
        }
    }

    pub fn representation(&self) -> String {
        format!(
            "({}, {}, {}; {})",
            self.u.iter().map(|b| if *b { '1' } else { '0' }).collect::<String>(),
            self.l.iter().map(|b| if *b { '1' } else { '0' }).collect::<String>(),
            self.phase,
            self.n
        )
    }
}

impl Display for Pauli {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.representation())
    }
}

#[derive(Clone, Debug)]
pub struct PauliSum(
    // we never have complex weights, as we ensure to make all the paulis always hermitian
    // (-> any phase differences are in {0, 2})
    pub Vec<(f64, Pauli)>,
);

impl PauliSum {
    pub fn new(ops: Vec<(f64, Pauli)>) -> Self {
        Self(ops)
    }

    pub fn multiply(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for (weight1, pauli1) in self.0.iter() {
            for (weight2, pauli2) in other.0.iter() {
                let prod = pauli1.multiply_as_paulis(pauli2);
                add_helper(&mut result, weight1 * weight2, &prod);
            }
        }
        removal_helper(&mut result);
        Self(result)
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.0.clone();
        for (weight, pauli) in other.0.iter() {
            add_helper(&mut result, *weight, pauli);
        }
        removal_helper(&mut result);
        Self(result)
    }

    pub fn subtract(&self, other: &Self) -> Self {
        let mut result = self.0.clone();
        for (weight, pauli) in other.0.iter() {
            add_helper(&mut result, -weight, pauli);
        }
        removal_helper(&mut result);
        Self(result)
    }

    pub fn single_add(&mut self, weight: f64, pauli: &Pauli) {
        add_helper(&mut self.0, weight, pauli);
        removal_helper(&mut self.0);
    }

    pub fn remove_zero_weights(&mut self) {
        removal_helper(&mut self.0);
    }

    pub fn multiply_with_one_imag_unit(&mut self) {
        for (_, pauli) in self.0.iter_mut() {
            pauli.add_to_phase(1);
        }
    }

    pub fn multiply_with_float(&mut self, scalar: f64) {
        for (weight, _) in self.0.iter_mut() {
            *weight *= scalar;
        }
    }

    /// given a list of (weight, pauli) pairs, return the hilbert schmidt inner product of
    /// the sum of these operators with another pauli; the paulis in the list should be
    /// hermition, (we define the inner product so that the conjugation is on the first
    /// argument into which we pass `ops` (and then don't conjugate it because it is
    /// hermitian))
    pub fn single_hilbert_schmidt_inner_product(&self, pauli: &Pauli) -> f64 {
        let mut total = 0.0;
        for (weight, op) in self.0.iter() {
            if op.is_proportional_to(pauli) {
                let phase = pauli.phase_difference(op);
                debug_assert!(phase.is_multiple_of(2));
                match phase {
                    0 => total += weight,
                    2 => total -= weight,
                    _ => unreachable!(),
                }
            }
        }
        total
    }
    // TODO: this method can be improved by assuming that the pauls in self are all
    // different (i.e., not proportional to each other), because then we can early break
    // the loop

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

fn add_helper(list: &mut Vec<(f64, Pauli)>, weight: f64, pauli: &Pauli) {
    for (ret_weight, ret_pauli) in list.iter_mut() {
        if pauli.is_proportional_to(ret_pauli) {
            let phase = pauli.phase_difference(ret_pauli);
            debug_assert!(phase.is_multiple_of(2));
            match phase {
                0 => *ret_weight += weight,
                2 => *ret_weight -= weight,
                _ => unreachable!(),
            }
            return; // EARLY RETURN here
        }
    }
    list.push((weight, pauli.clone())); // not done if EARLY RETURN hit
}

fn removal_helper(ops: &mut Vec<(f64, Pauli)>) {
    // result.retain(|(weight, _)| weight.abs() > 1e-10);
    // or alternatively:
    let to_remove: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(i, (weight, _))| {
            if weight.abs() < 1e-8 {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    for i in to_remove.into_iter().rev() {
        ops.swap_remove(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let p1 = Pauli::from_indices(3, &[0], &[1], 0);
        let p2 = Pauli::from_indices(3, &[1], &[1], 2);
        println!("{} {}", p1.to_pretty_string(), p1);
    }
}
