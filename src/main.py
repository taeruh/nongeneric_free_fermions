#!/usr/bin/env python

import os
from run import run, num_claws, num_vertices, phase_diagram


def test():
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wl, wlexpr

    print("Starting Wolfram session...")
    session = WolframLanguageSession(
        "/usr/local/Wolfram/Wolfram/14.3/Executables/WolframKernel"
    )
    print("Loading Wolfram code")
    session.evaluate("""
    ClearALL["Coefficients"]
    ClearALL["ComputeEpsilons"]
    Coefficients[M_Integer, a_, b_, c_] := Module[
      {
       buf = {{}, {}, {1}},
       p, pm1, pm2, pm3, f1, f2, f3, i, pmax
       },
      f1 = a + b + c;
      f2 = a b + a c + b c;
      f3 = a b c;
      Do[
       pm1 = buf[[Mod[i - 2, 3] + 1]];
       pm2 = buf[[Mod[i - 3, 3] + 1]];
       pm3 = buf[[Mod[i - 4, 3] + 1]];
       pmax = i + 1;
       p = Table[0, pmax];
       Do[p[[j]] += pm1[[j]], {j, Length[pm1]}];
       Do[p[[j + 1]] -= f1 pm1[[j]], {j, Length[pm1]}];
       Do[p[[j + 2]] -= f2 pm2[[j]], {j, Length[pm2]}];
       Do[p[[j + 3]] -= f3 pm3[[j]], {j, Length[pm3]}];
       buf[[Mod[i - 1, 3] + 1]] = p;
       , {i, 1, M}
       ];
      poly = buf[[Mod[M - 1, 3] + 1]];
      poly3 = buf[[Mod[M - 2, 3] + 1]];
      <|"Poly" -> poly, "Poly3" -> poly3|>
      ]
    ComputeEpsilons[M_Integer, a_, b_, c_] := Module[
      {x, coeffs, polycoeffs, poly3coeffs, unsortedroots, roots, eps2, 
       eps, poly3, pmfactors},
      coeffs = Coefficients[M, a, b, c];
      polycoeffs = coeffs["Poly"];
      poly3coeffs = coeffs["Poly3"];
      unsortedroots = 
       x /. NSolve[
         Sum[polycoeffs[[i + 1]] x^i, {i, 0, Length[polycoeffs] - 1}] == 
          0, x, WorkingPrecision -> 50];
      roots = ReverseSort[unsortedroots];
      eps2 = 1/roots;
      eps = Sqrt[eps2];
      poly3[x_] := 
       Sum[poly3coeffs[[i + 1]] x^i, {i, 0, Length[poly3coeffs] - 1}];
      pmfactors = poly3 /@ roots;
      <|"Roots" -> roots, "EpsilonSquared" -> eps2, "Epsilons" -> eps, 
       "PMFactors" -> pmfactors|>
      ]
    """)
    print("Wolfram session is ready.")

    result = session.evaluate('ComputeEpsilons[24, 1.0, 1.0, 1.0]["Roots"]')
    import numpy as np
    roots = np.array([np.float128(r) for r in result])
    print(roots)

    print("Terminating Wolfram session...")
    session.terminate()
    print("Wolfram session terminated.")


def main():

    os.makedirs("output", exist_ok=True)

    # test()
    phase_diagram.run()
    # num_claws.run()
    # num_vertices.run()
    # run.run()
    # run.get_phase_diagram()
    # run.test_t()
    # run.currents_plot()
    # run.trying_to_reconstruct_fukai_from_bilinears()
    # import krylov
    # krylov.test_projections()


if __name__ == "__main__":
    main()
