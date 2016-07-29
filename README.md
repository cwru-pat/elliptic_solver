# Elliptic Solver Code

Example compile && run command:
> `g++ main.cpp multigrid.cpp -O3 -Wall --std=c++11 -fopenmp && time ./a.out`

Example compile && run with profiling enabled (not parallelized):
> `g++ main.cpp multigrid.cpp -O3 -Wall --std=c++11 -pg && time ./a.out`

View profiling:
> `gprof a.out | less`

Bottlenecks according to gprof:

| % time in program  | function call |
| ------------- | ------------- |
| 46.70 | FASMultigrid::_evaluateEllipticEquationPt(long long, long long, long long, long long) |
| 21.79 | FASMultigrid::_jacobianRelax(long long, double, double, long long) |
| 18.00 | FASMultigrid::_relaxSolution_GaussSeidel(long long, long long) |
| 3.22 | FASMultigrid::_getLambda(long long, double) |
