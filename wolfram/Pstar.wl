ClearAll[PStar]

PStar[M_Integer, x_, \[Alpha]_, \[Beta]_, \[Gamma]_] := 
 Module[{memo = <||>}, Clear[p];
  p[0] = 1;
  p[-3] = 0;
  p[-6] = 0;
  p[n_] := 
   memo[n] = 
    Expand[(1 - x (\[Alpha] + \[Beta] + \[Gamma])) p[n - 3] - 
      x^2 (\[Alpha] \[Beta] + \[Alpha] \[Gamma] + \[Beta] \[Gamma]) p[
        n - 6] - x^3 \[Alpha] \[Beta] \[Gamma] p[n - 9]];
  p[M]]


ClearAll[ComputeEpsilons]

ComputeEpsilons[M_Integer, \[Alpha]_, \[Beta]_, \[Gamma]_] := 
 Module[{x, poly, roots, u2, eps2, eps}, 
  poly = PStar[M, x, \[Alpha], \[Beta], \[Gamma]];
  roots = x /. Solve[poly == 0, x];
  u2 = Sort[roots];
  eps2 = 1/u2;
  eps = Sqrt[eps2];
  <|"Polynomial" -> poly, "RootsU2" -> u2, "EpsilonSquared" -> eps2, 
   "Epsilons" -> eps|>]
