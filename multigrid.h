#ifndef FAS_MULTIGRID_H
#define FAS_MULTIGRID_H

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>

#define PI  (4.0*atan(1.0))

#define FAS_LOOP3_N(i, j, k, nx, ny, nz)  \
  for(i=0; i<nx; ++i)                     \
    for(j=0; j<ny; ++j)                   \
      for(k=0; k<nz; ++k)

typedef double REAL_T;
typedef long int IDX_T;

/**
 * @brief Full Approximation Storage ("FAS") Multigrid solver class
 * @details 
 * see: https://computation.llnl.gov/casc/people/henson/postscript/UCRL_JC_150259.pdf
 * section 3 for a very concise, clean description of the FAS Scheme.
 *
 * And also: http://www.scicomp.ucsd.edu/~mholst/pubs/dist/Hols94e.pdf
 * for introduction to inexact Newton method.  
 * 
 * In this implementation, the following step is done for restriction:
 *   restrict a fine grid approximation to a coarser grid:
 *     - compute the approximate solution on the coarse grid:
 *        v^2h = I^2h_h v^h
 *     - compute the source term on the coarse grid (coarse_src):
 *        f^2h = A^2h ( v^2h ) + I^2h_h ( f^h - A^h ( v^h ) )
 * And the following is done for prolongation:
 *     - compute the error
 *     (TODO: finish)
 *  
 * The set of grids (array of arrays) is referred to as a "heirarchy" of grids,
 * where each grid has some "depth".
 * 
 */
class FASMultigrid
{

protected:
  
  IDX_T max_depth, max_depth_idx;
  IDX_T min_depth, min_depth_idx;
  IDX_T total_depths, max_relax_iters;
  
  REAL_T grid_length_x, grid_length_y, grid_length_z;
  IDX_T * nx_h, * ny_h, * nz_h; // grid points in each direction at different depths

  // grid (array) type
  typedef REAL_T * fas_grid_t;
  // heirarchy type (set of some grids at different depths)
  typedef fas_grid_t * fas_heirarchy_t;
  // set of heirarchies
  typedef fas_heirarchy_t * fas_heirarchy_set_t;
  // enum for relaxation type
  enum relax_t { 
    inexact_newton,
    inexact_newton_constrained, // inexact Newton with volume constraint enforced
    newton
  };

  relax_t relax_scheme;

  // define heirarchy of references to grids
  fas_heirarchy_t tmp_h,  // reusable grid for storing intermediate calculations
    coarse_src_h,         // multigrid source term
    u_h,                  // field seeking a solution for
    appx_u_h,             // field containing an approximate solution
    damping_tmp_h,        // _lap (u) - f, used to calculate F(u + \lambda v)
    lap_v_h,              // _lap(v), used to caculate F(u + \lambda v)
    damping_v_h,          // v, used to update u through u_{n+1} = u{n} + v in interation
    jac_rhs_h;            // - F(u) which is rhs of Jacob Linear function
  
  // source terms: u^u_exp * rho
  IDX_T rho_num; // number of source terms
  IDX_T * u_exp; // exponent of u for each term, has rho_num terms in total 
  fas_heirarchy_set_t rho_h; // source matter terms with number being rho_num;

  /**
   * @brief indexing scheme of a grid heirarchy
   * @description return index of grid at a particular depth
   *  in a grid heirarchy
   * 
   * @param depth "depth" of grid
   * @return index
   */
  inline IDX_T _dIdx(IDX_T depth)
  {
    return depth - min_depth;
  }

  /**
   * @brief return sign of argument
   * @details return zerp when argument is zero
   */
  inline IDX_T _sign(REAL_T x)
  {
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
  }

  /**
   * @brief      Return array index of a point located at (i,j,k) on a
   * grid with periodic boundaries, of size n^3.
   *
   * @param[in]  i     label of points in x-dir
   * @param[in]  j     label of points in y-dir
   * @param[in]  k     label of points in z-dir
   * @param[in]  nx    number of grids in x-dir
   * @param[in]  ny    number of grids in y-dir
   * @param[in]  nz    number of grids in z-dir
   * 
   * @return     array index
   */
  inline IDX_T _gIdx(IDX_T i, IDX_T j, IDX_T k, IDX_T nx, IDX_T ny, IDX_T nz)
  {
    return ((i+nx)%nx)*ny*nz + ((j+ny)%ny)*nz + (k+nz)%nz;
  }

  /**
   * @brief compute power of 2
   * 
   * @param pwr power to raise 2 to
   * @return 2^pwr
   */
  inline IDX_T _2toPwr(IDX_T pwr)
  {
    return 1<<pwr;
  }

  /**
   * @brief compute integer number to power of 3
   * 
   * @param number to raise to ^3
   * @return pwr^3
   */
  inline IDX_T _Pwr3(IDX_T num)
  {
    return num*num*num;
  }

  /**
   * @brief compute integer number to power of 2
   * 
   * @param number to raise to ^2
   * @return pwr^2
   */
  inline REAL_T _Pwr2(REAL_T num)
  {
    return num * num;
  }

  /**
   * @brief      Take a laplacian stencil on "grid" of size n^3 at point i,j,k
   *
   * @return     laplacian stencil
   */
  inline REAL_T _laplacian(fas_grid_t grid, IDX_T i, IDX_T j, IDX_T k, IDX_T nx, IDX_T ny, IDX_T nz)
  {
    REAL_T dx = grid_length_x/nx, dy = grid_length_y/ny, dz = grid_length_z/nz;

    return (
      (grid[_gIdx(i+1, j, k, nx, ny, nz)] + grid[_gIdx(i-1, j, k, nx, ny, nz)]
        - 2.0 * grid[_gIdx(i, j, k, nx, ny, nz)] ) / (dx*dx)
      + (grid[_gIdx(i, j+1, k, nx, ny, nz)] + grid[_gIdx(i, j-1, k, nx, ny, nz)]
       - 2.0 * grid[_gIdx(i, j, k, nx, ny, nz)] ) / (dy*dy)
      + (grid[_gIdx(i, j, k+1, nx, ny, nz)] + grid[_gIdx(i, j, k-1, nx, ny, nz)]
       - 2.0 * grid[_gIdx(i, j, k, nx, ny, nz)] ) / (dz*dz)
    );
  }

  REAL_T _srcVal(IDX_T pos_idx, IDX_T depth_idx, REAL_T u);
  REAL_T _srcValDir(IDX_T pos_idx, IDX_T depth_idx, REAL_T u);

  void _zeroGrid(fas_grid_t grid, IDX_T points);

  REAL_T _totalGrid(fas_grid_t grid, IDX_T points);

  void _shiftGridVals(fas_grid_t grid, REAL_T shift, IDX_T points);

  void _restrictFine2coarse(fas_heirarchy_t grid_heirarchy, IDX_T fine_depth);
  void _interpolateCoarse2fine(fas_heirarchy_t grid_heirarchy,  IDX_T coarse_depth);

  void _evaluateEllipticEquation(fas_heirarchy_t result_h, IDX_T depth);

  REAL_T _evaluateEllipticEquationPt(IDX_T depth, IDX_T i, IDX_T j, IDX_T k);

  void _shiftConstrainedFieldMonopole(IDX_T depth);

  REAL_T _monopoleConstraintTotal(IDX_T depth, REAL_T shift);

  REAL_T _monopoleConstraintDerivativeTotal(IDX_T depth, REAL_T shift);

  void _computeResidual(fas_heirarchy_t residual_h, IDX_T depth);

  REAL_T _getMaxResidual(IDX_T depth);

  void _computeCoarseRestrictions(IDX_T fine_depth);

  void _changeApproximateSolutionToError(fas_heirarchy_t appx_to_err_h,
           fas_heirarchy_t exact_soln_h, IDX_T depth);

  void _correctFineFromCoarseErr_Err2Appx(fas_heirarchy_t err2appx_h,
            fas_heirarchy_t appx_soln_h, IDX_T fine_depth);

  void _copyGrid(fas_heirarchy_t from_h, fas_heirarchy_t to_h, IDX_T depth);

  REAL_T _getLambda(IDX_T depth, REAL_T norm);

  REAL_T _dampingConstraintTotal(IDX_T depth, REAL_T shift);

  REAL_T _dampingConstraintDerivativeTotal(IDX_T depth, REAL_T shift);

  void _shiftConstrainedDamping(IDX_T depth);

  bool _jacobianRelax(IDX_T depth, REAL_T norm, REAL_T C, IDX_T p);

  bool _singularityExists(IDX_T depth);

  void _relaxSolution_GaussSeidel(IDX_T depth, IDX_T max_iterations);

  void _initializeMultigrid(IDX_T grid_num_x_in, IDX_T grid_num_y_in, IDX_T grid_num_z_in,
         REAL_T grid_length_x_in, REAL_T grid_length_y_in, REAL_T grid_length_z_in,
         IDX_T max_depth_in, IDX_T max_relax_iters_in, relax_t relax_scheme_in);

  void _printStrip(fas_heirarchy_t out_h, IDX_T depth);


  void _printAll(fas_heirarchy_t out_h, IDX_T depth);

public:

  // expose relax_t enum
  typedef relax_t relax_t;

  FASMultigrid(IDX_T grid_num_x_in, IDX_T grid_num_y_in, IDX_T grid_num_z_in,
    REAL_T grid_length_x_in, REAL_T grid_length_y_in, REAL_T grid_length_z_in,
    IDX_T max_depth_in, IDX_T max_relax_iters_in, relax_t relax_scheme_in);

  FASMultigrid(IDX_T grid_num_in, REAL_T grid_length_in, IDX_T max_depth_in);

  ~FASMultigrid();

  void build_rho(IDX_T src_num_in, IDX_T u_exp_in[]);
  void build_rho(IDX_T src_num);
  void initializeRhoHeirarchy();

  void VCycle();
  void VCycles(IDX_T num_cycles);

  void setTrialSolution(IDX_T type);
  void add_poly_srcs(IDX_T type);

  void printSolutionStrip(IDX_T depth);
  void printSourceStrip(IDX_T rho_n, IDX_T depth);

  void setPolySrcAtPt(IDX_T i, IDX_T j, IDX_T k, IDX_T n, REAL_T value);

  REAL_T * getSolution();

};

#endif
