#include "multigrid_arbitrary_grids.h"
#include <iostream>
#include <cstdlib>

typedef double real_t;
typedef long int idx_t;
typedef FASMultigrid<real_t, idx_t> multigrid_t;

int main(int argc, char **argv)
{
   
  srand(129);
  std::cout.precision(15);
  
  #ifdef _OPENMP
    omp_set_num_threads(4);
  #endif
 
  std::cout << "Creating multigrid class...\n";
  idx_t min_depth = 2;
  idx_t max_depth = 6;
  
  real_t grid_length = 1.0;
  idx_t iter_num = 5;

  multigrid_t::relax_t relax_scheme = multigrid_t::relax_t::inexact_newton_constrained;

  idx_t grid_length_x = 1, grid_length_y = 1, grid_length_z = 1;

  //set grid number of different direction, usually we require: grid_num_i % (2^iter_depth) == 0
  //to maintain fast convergent rate.
  //But the code actually can deal with arbitrary grid number at any direction.
  idx_t grid_num_x = 64, grid_num_y = 64, grid_num_z = 64;
  idx_t iter_depth = 5; //number of layers for multigrid interation

  multigrid_t multigrid (grid_num_x, grid_num_y, grid_num_z,
  grid_length_x, grid_length_y, grid_length_z, iter_depth, iter_num, relax_scheme);

  std::cout << "  initializing...\n";
  multigrid.setTrialSolution(0);
  multigrid.add_poly_srcs(0); 
  std::cout << "  done.\n";
  
  std::cout << "Performing V-Cycles...\n";
  multigrid.VCycles(3);
  std::cout << "  done.\n";
  
  exit(EXIT_SUCCESS);
}
