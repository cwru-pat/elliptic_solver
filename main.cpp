#include "multigrid_inexact_Newton.h"
#include <iostream>
#include <cstdlib>

typedef double real_t;
typedef long int idx_t;


int main(int argc, char **argv)
{
   
  srand(129);
  std::cout.precision(15);
  
  #ifdef _OPENMP
    omp_set_num_threads(4);
  #endif
  
  std::cout << "Creating multigrid class...\n";
  idx_t min_depth = 2;
  idx_t max_depth = 7;
  real_t grid_length = 1.0;
  idx_t iter_num = 5;
  idx_t relax_scheme = 2;
  FASMultigrid<real_t, idx_t> multigrid (max_depth, min_depth, grid_length, iter_num, relax_scheme);
  std::cout << "  initializing...\n";
  multigrid.add_poly_srcs(0); 
  multigrid.setTrialSolution();
  std::cout << "  done.\n";
  std::cout << "Performing V-Cycles...\n";
  multigrid.VCycles(6);
  std::cout << "  done.\n";
  
  exit(EXIT_SUCCESS);
}
