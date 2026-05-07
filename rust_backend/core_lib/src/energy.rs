// mus = []
// for l in range(self.num_currents):
//     l = 1 + 2 * l
//     mul = np.zeros(self.num_triangles * self.num_triangles, dtype=np.float128)
//     for m in range(self.num_triangles):
//         for n in range(self.num_triangles):
//             mu = (
//                 2
//                 * (-1) ** (((l - 1) // 2) % 2)
//                 * self.effective_norm[m]
//                 * self.effective_norm[n]
//             )
//             eps_sum = 0
//             for i in range(l):
//                 if i % 2 == 0:
//                     eps_sum += self.eps[m] ** i * self.eps[n] ** (l - i)
//                 else:
//                     eps_sum += self.eps[n] ** i * self.eps[m] ** (l - i)
//             mul[m * self.num_triangles + n] = mu * eps_sum
//     mus.append(mul)

pub type Float = f64;

pub fn calculate_mus(
    num_currents: usize,
    num_triangles: usize,
    effective_norm: &[Float],
    eps: &[Float],
) -> Vec<Vec<Float>> {
    let mut mus = Vec::new();
    for l in 0..num_currents {
        let l = 1 + 2 * l;
        let mut mul = vec![0.0; num_triangles * num_triangles];
        for m in 0..num_triangles {
            for n in 0..num_triangles {
                let mu = 2.0
                    * (-1.0 as Float).powi(((l - 1) / 2) as i32 % 2)
                    * effective_norm[m]
                    * effective_norm[n];
                let mut eps_sum = 0.0;
                for i in 0..l {
                    if i % 2 == 0 {
                        eps_sum += eps[m].powi(i as i32) * eps[n].powi((l - i) as i32);
                    } else {
                        eps_sum += eps[n].powi(i as i32) * eps[m].powi((l - i) as i32);
                    }
                }
                mul[m * num_triangles + n] = mu * eps_sum;
            }
        }
        mus.push(mul);
    }
    // println!("{:?}", mus[0][0]);
    mus
}
