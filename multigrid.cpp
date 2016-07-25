#include "multigrid.h"

/**
 * @brief return the value of the "left hand side" of the elliptical
 * equation, excluding the Laplacian term
 * 
 * @param index of position
 * @param index of depth
 * @param the value of u
 */
REAL_T FASMultigrid::_srcVal(IDX_T pos_idx, IDX_T depth_idx, REAL_T u)
{
  REAL_T ans = 0.0;
  for(IDX_T I = 0; I < rho_num; I++)
  {
    ans += rho_h[I][depth_idx][pos_idx] * std::pow(u, (REAL_T)u_exp[I]);
  }
  return ans;
}

/**
 * @brief return the derivative of the "left hand side" of the elliptical
 * equation, excluding the Laplacian term
 * 
 * @param index of position
 * @param index of depth
 * @param the value of u
 */
REAL_T FASMultigrid::_srcValDir(IDX_T pos_idx, IDX_T depth_idx, REAL_T u)
{
  REAL_T ans = 0.0;
  
  for(IDX_T p = 0; p < rho_num; p++)
  {
    ans += (REAL_T)u_exp[p] * rho_h[p][depth_idx][pos_idx]
      * std::pow(u, (REAL_T)u_exp[p]-1.0);
  }

  return ans;
}

/**
 * @brief      initialize a grid to 0
 *
 * @param      grid    grid (array) to initialize
 * @param[in]  points  # points in array / on grid
 */
void FASMultigrid::_zeroGrid(fas_grid_t grid, IDX_T points)
{
  for(IDX_T i=0; i < points; i++)
    grid[i] = 0;
}

/**
 * @brief Compute total of grid (sum of all elements)
 * 
 * @param grid grid to total
 * @param points # points in grid
 * 
 * @return total
 */
REAL_T FASMultigrid::_totalGrid(fas_grid_t grid, IDX_T points)
{
  REAL_T total = 0;
  #pragma omp parallel for reduction(+:total)
  for(IDX_T i=0; i < points; i++)
    total += grid[i];

  return total;
}

/**
 * @brief Compute average of a grid
 * 
 * @param grid grid to average
 * @param points # points in grid
 * 
 * @return average
 */
REAL_T FASMultigrid::_averageGrid(fas_grid_t grid, IDX_T points)
{
  return _totalGrid(grid, points)/points;
}

/**
 * @brief Compute max value in a grid
 * 
 * @param grid grid to find max of
 * @param points # points in grid
 * 
 * @return max
 */
REAL_T FASMultigrid::_maxGrid(fas_grid_t grid, IDX_T points)
{
  REAL_T max = grid[0];
  #pragma omp parallel for
  for(IDX_T i=0; i < points; i++)
    #pragma omp critical
    {
      if(grid[i] > max)
        max = grid[i];
    }

  return max;
}

/**
 * @brief Compute min value in a grid
 * 
 * @param grid grid to find min of
 * @param points # points in grid
 * 
 * @return min
 */
REAL_T FASMultigrid::_minGrid(fas_grid_t grid, IDX_T points)
{
  REAL_T min = grid[0];
  #pragma omp parallel for
  for(IDX_T i=0; i < points; i++)
    #pragma omp critical
    {
      if(grid[i] < min)
        min = grid[i];
    }

  return min;
}

/**
 * @brief Shift all values in grid by a value
 * @details eg; grid[i] += shift for all i
 * 
 * @param grid grid to shift
 * @param shift amount to shift by
 * @param points # points in grid
 */
void FASMultigrid::_shiftGridVals(fas_grid_t grid, REAL_T shift, IDX_T points)
{
  #pragma omp parallel for
  for(IDX_T i=0; i < points; i++)
    grid[i] += shift;
}

/**
 * @brief "restrict" a fine grid to coarser grid
 * @details Restriction scheme:
 *  (1 given cell)*(1/8) + (6 adjacent "faces") * (1/16)
 *  + (12 adjacent "edges") * (1/32) + (8 adjacent "corners") * (1/64)
 * 
 * @param field_heirarchy field to restrict
 * @param fine_depth "depth" of finer grid
 */
void FASMultigrid::_restrictFine2coarse(fas_heirarchy_t grid_heirarchy, IDX_T fine_depth)
{
  
  IDX_T fine_idx = _dIdx(fine_depth);
  IDX_T coarse_idx = fine_idx-1;

  IDX_T n_fine_x = nx_h[fine_idx], n_fine_y = ny_h[fine_idx], n_fine_z = nz_h[fine_idx];
  IDX_T n_coarse_x = n_fine_x / 2, n_coarse_y = n_fine_y / 2, n_coarse_z = n_fine_z / 2 ;
  
  fas_grid_t const fine_grid = grid_heirarchy[fine_idx];
  fas_grid_t const coarse_grid = grid_heirarchy[coarse_idx];

  IDX_T i, j, k; // coarse grid iterator
  IDX_T fi, fj, fk; // fine grid indexes

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i, j, k, n_coarse_x, n_coarse_y, n_coarse_z)
  {
    fi = i*2;
    fj = j*2;
    fk = k*2;

    coarse_grid[_gIdx(i,j,k,n_coarse_x, n_coarse_y, n_coarse_z)] =
      0.125 * fine_grid[_gIdx(fi,fj,fk,n_fine_x, n_fine_y, n_fine_z)]
      + 0.0625 * (
        fine_grid[_gIdx(fi+1,fj,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj+1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj-1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj,fk-1,n_fine_x, n_fine_y, n_fine_z)]
      ) + 0.03125 * (
        fine_grid[_gIdx(fi+1,fj+1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj-1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj+1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj-1,fk,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj+1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj+1,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj-1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi,fj-1,fk-1,n_fine_x, n_fine_y, n_fine_z)]
      ) + 0.015625 * (
        fine_grid[_gIdx(fi+1,fj+1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj+1,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj-1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj+1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi+1,fj-1,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj+1,fk-1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj-1,fk+1,n_fine_x, n_fine_y, n_fine_z)] +
        fine_grid[_gIdx(fi-1,fj-1,fk-1,n_fine_x, n_fine_y, n_fine_z)]
      );

  } // end loop
} // end restrict_fine2coarse

/**
 * @brief interpolate a coarse grid to a finer grid
 * @details using a lot of "if" before updating to deal with the boundary probs when
 * n_coarse * 2 != n_fine 
 */
void FASMultigrid::_interpolateCoarse2fine(fas_heirarchy_t grid_heirarchy,  IDX_T coarse_depth) 
{   
  
  IDX_T fine_idx = _dIdx(coarse_depth+1);
  IDX_T coarse_idx = _dIdx(coarse_depth);

  IDX_T n_coarse_x = nx_h[coarse_idx], n_coarse_y = ny_h[coarse_idx], n_coarse_z = nz_h[coarse_idx];
  IDX_T n_fine_x =  nx_h[fine_idx], n_fine_y = ny_h[fine_idx], n_fine_z = nz_h[fine_idx];
  
  fas_grid_t const coarse_grid = grid_heirarchy[coarse_idx];
  fas_grid_t const fine_grid = grid_heirarchy[fine_idx];

  IDX_T i, j, k;
  IDX_T fi, fj, fk;

  _zeroGrid(fine_grid, n_fine_x * n_fine_y * n_fine_z);

  
  #pragma omp parallel for private(i, j, k, fi, fj, fk)
  FAS_LOOP3_N(i, j, k, n_coarse_x, n_coarse_y, n_coarse_z)
  {
    fi = i*2;
    fj = j*2;
    fk = k*2;

    REAL_T coarse_grid_val = coarse_grid[_gIdx(i,j,k,n_coarse_x, n_coarse_y, n_coarse_z)];

    // loop over adjacent cells.
    for(IDX_T  i_adj = -1; i_adj <= 1; ++i_adj )
      for(IDX_T  j_adj = -1; j_adj <= 1; ++j_adj )
        for(IDX_T  k_adj = -1; k_adj <= 1; ++k_adj )
        {
          IDX_T fine_grid_loc = _gIdx(fi + i_adj, fj + j_adj, fk + k_adj,
            n_fine_x, n_fine_y, n_fine_z);
          IDX_T coarse_grid_loc = _gIdx(fi + i_adj, fj + j_adj, fk + k_adj,
            n_coarse_x*2, n_coarse_y*2, n_coarse_z*2);

          if(i_adj == 0 && j_adj == 0 && k_adj == 0)
          {
            #pragma omp atomic
            fine_grid[fine_grid_loc] += coarse_grid_val;
          }
          else if(fine_grid_loc == coarse_grid_loc)
          {
            REAL_T divisor = std::pow( 2.0,
              std::abs(i_adj) + std::abs(j_adj) + std::abs(k_adj) );
            #pragma omp atomic
            fine_grid[fine_grid_loc] += coarse_grid_val/divisor;
          }

        }
  }

}

/**
 * @brief      Evaluate elliptic operator at given depth
 *
 * @param[in]  depth     depth to evaluate at
 * @param      result_h  grid to store result on
 */
void FASMultigrid::_evaluateEllipticEquation(fas_heirarchy_t result_h, IDX_T depth)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const result = result_h[depth_idx];

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i, j, k, nx, ny, nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    result[idx] = _evaluateEllipticEquationPt(depth, i, j, k);
  }
}

/**
 * @brief      Evaluate elliptic operator at given depth at a point
 *
 * @param[in]  depth     depth to evaluate at
 * @param      result_h  grid to store result on
 * 
 * @return elliptic operator evaluated at a point
 */
REAL_T FASMultigrid::_evaluateEllipticEquationPt(IDX_T depth, IDX_T i, IDX_T j, IDX_T k)
{
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const u = u_h[depth_idx];
 
  IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

  return _laplacian(u, i, j, k, nx, ny, nz) + _srcVal(idx, depth_idx, u[idx]);
}

/**
 * @brief Find value of A that gives a root of the function:
 * f(A) = \integral (\rho * (\u + A)^5 - coarse_src  ) d V
 * and offset phi by this
 *
 * @param depth depth to compute at
 * @return "A"
 */
void FASMultigrid::_shiftConstrainedFieldMonopole(IDX_T depth)
{
  REAL_T prev_frac = 0.0, shift = 0.0;
  REAL_T num = 1, den = 1;

  while(fabs(fabs(num/den) - prev_frac) < 1e-9)
  {
    num = _monopoleConstraintTotal(depth, shift);
    den = _monopoleConstraintDerivativeTotal(depth, shift);

    shift -= num/den;

    prev_frac = fabs(num/den);
  }

  IDX_T depth_idx = _dIdx(depth);
  IDX_T grid_pts = nx_h[depth_idx] * ny_h[depth_idx] * nz_h[depth_idx];
  _shiftGridVals(u_h[depth_idx], shift, grid_pts);
}

REAL_T FASMultigrid::_monopoleConstraintTotal(IDX_T depth, REAL_T shift)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  //IDX_T n = _2toPwr(depth);

  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  
  fas_grid_t const u = u_h[depth_idx];
  
  fas_grid_t const coarse_src = coarse_src_h[depth_idx];

  REAL_T total = 0.0;
  #pragma omp parallel for default(shared) private(i,j,k) reduction(+:total)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

    total += _srcVal(idx, depth_idx, u[idx] + shift) - coarse_src[idx];
  }

  return total;
}

REAL_T FASMultigrid::_monopoleConstraintDerivativeTotal(IDX_T depth, REAL_T shift)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const u = u_h[depth_idx];

  fas_grid_t const coarse_src = coarse_src_h[depth_idx];

  REAL_T total = 0.0;
  #pragma omp parallel for default(shared) private(i,j,k) reduction(+:total)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

    total += _srcValDir(idx, depth_idx, u[idx]);
  }

  return total;
}

/**
 * @brief      Computes residual, stores result in "tmp" heirarchy
 *
 * @param[in]  depth  depth to compute residual at
 * @param      residual_h  heirarchy of arrays to store residual in
 */
void FASMultigrid::_computeResidual(fas_heirarchy_t residual_h, IDX_T depth)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const u = u_h[depth_idx];

  fas_grid_t const coarse_src = coarse_src_h[depth_idx];
  fas_grid_t const residual = residual_h[depth_idx];

  // intermediately store elliptic operator result on residual grid
  _evaluateEllipticEquation(residual_h, depth);
  
  // residual is coarse_src - elliptic operator
  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    residual[idx] = coarse_src[idx] - residual[idx];
  }
}

/**
 * @brief      Computes max residual, uses intermediate tmp_h
 *
 * @param[in]  depth  depth to compute residual at
 * @return residual
 */
REAL_T FASMultigrid::_getMaxResidual(IDX_T depth)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  fas_grid_t const coarse_src = coarse_src_h[depth_idx];

  REAL_T max_residual = 0.0;

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    REAL_T current_residual = std::fabs(coarse_src[idx]
          - _evaluateEllipticEquationPt(depth, i, j, k));
      
    #pragma omp critical
    {
      if(current_residual > max_residual)
        max_residual = current_residual;
    }
  }

  return max_residual;
}

/**
 * @brief      Compute coarse_src and u on a coarser grid
 * using tmp_h for some computations
 *
 * @param[in]  fine_depth  depth of grid to coarsen
 */
void FASMultigrid::_computeCoarseRestrictions(IDX_T fine_depth)
{
  IDX_T i, j, k;

  // restrict approximate solution on coarse grid
  _restrictFine2coarse(u_h, fine_depth);

  // compute residual on fine grid; intermediately store the result
  // in the tmp grid
  _computeResidual(tmp_h, fine_depth);
  // restrict the residual to the coarse grid
  _restrictFine2coarse(tmp_h, fine_depth);
  // compute elliptic operator on coarse grid; store in source
  _evaluateEllipticEquation(coarse_src_h, fine_depth-1);

  // add in restricted residual to coarse source;
  // coarse source is then set.

  IDX_T coarse_idx = _dIdx(fine_depth-1);
  IDX_T nx = nx_h[coarse_idx], ny = ny_h[coarse_idx], nz = nz_h[coarse_idx];
  fas_grid_t const coarse_src = coarse_src_h[coarse_idx];
  fas_grid_t const tmp = tmp_h[coarse_idx];

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    coarse_src[idx] += tmp[idx];
  }

}

/**
 * @brief      Convert a grid containing an approximate solution
 *  to a grid containing the solution error, err = true - appx.
 *
 * @param      appx_to_err_h  grid heirarchy containing appx'n to convert
 * @param      exact_soln_h   heirarchy containing exact solution
 * @param[in]  depth          depth to perform computation at
 */
void FASMultigrid::_changeApproximateSolutionToError(fas_heirarchy_t appx_to_err_h,
         fas_heirarchy_t exact_soln_h, IDX_T depth)
{
  IDX_T i, j, k;

  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];


  fas_grid_t const appx_to_err = appx_to_err_h[depth_idx];
  fas_grid_t const exact_soln = exact_soln_h[depth_idx];

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    appx_to_err[idx] = exact_soln[idx] - appx_to_err[idx];
  }
}

/**
 * @brief Compute and add in correction to fine grid from error
 * on coarser grid; replace error with appx. solution
 * 
 * @param err_h grid heirarchy containing error
 * @param err2appx_h heirarchy containing approximate solution
 * @param fine_depth depth of fine grid to correct
 */
void FASMultigrid::_correctFineFromCoarseErr_Err2Appx(fas_heirarchy_t err2appx_h,
          fas_heirarchy_t appx_soln_h, IDX_T fine_depth)
{
  IDX_T i, j, k;
  IDX_T coarse_depth = fine_depth-1;

  IDX_T fine_depth_idx = _dIdx(fine_depth);
  //IDX_T n_fine = _2toPwr(fine_depth);
  IDX_T n_fine_x = nx_h[fine_depth_idx], n_fine_y = ny_h[fine_depth_idx], n_fine_z = nz_h[fine_depth_idx];
  _interpolateCoarse2fine(err2appx_h, coarse_depth);

  fas_grid_t const err2appx = err2appx_h[fine_depth_idx];
  fas_grid_t const appx_soln = appx_soln_h[fine_depth_idx];

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i,j,k, n_fine_x, n_fine_y, n_fine_z)
  {
    IDX_T idx = _gIdx(i, j, k, n_fine_x,n_fine_y,n_fine_z);
    // appx. solution in intermediate variable
    REAL_T appx_val = appx_soln[idx];
    // correct approximate solution with error
    appx_soln[idx] += err2appx[idx];
    // store approximate solution in err2appx
    err2appx[idx] = appx_val;
  }
}

/**
 * @brief Copy grid from one heirarchy to another
 * 
 * @param from_h copy from this heirarchy
 * @param to_h to this heirarchy
 * @param depth at this depth
 */
void FASMultigrid::_copyGrid(fas_heirarchy_t from_h, fas_heirarchy_t to_h, IDX_T depth)
{
  IDX_T depth_idx = _dIdx(depth);
  IDX_T points = nx_h[depth_idx] * ny_h[depth_idx] * nz_h[depth_idx];

  fas_grid_t const from = from_h[depth_idx];
  fas_grid_t const to = to_h[depth_idx];

  std::copy(from, from + points, to);
}

/**
 * @brief iterative method to find a \lambda between 1 and zero,
 *        returning the largest value that satisfies
 *        norm less than the norm of F(u)
 * @param depth
 * @param norm
 */
REAL_T FASMultigrid::_getLambda(IDX_T depth, REAL_T norm)
{
  //l inearly inumerate corresponding damping parameter \lambda
  IDX_T i, j, k, s;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  REAL_T lambda = 0.0, temp, sum = 0.0;
  fas_grid_t const damping_tmp = damping_tmp_h[depth_idx];
  fas_grid_t const lap_v = lap_v_h[depth_idx];
  fas_grid_t const damping_v = damping_v_h[depth_idx];

  fas_grid_t const u = u_h[depth_idx];
  
  for( s = 0; s < 100; s++)
  {
    lambda = 1.0 - (REAL_T)s * 0.01; //should always start with \lambda = 1
    sum = 0.0;
    
    #pragma omp parallel for default(shared) private(i,j,k,temp) reduction(+:sum)
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i, j, k, nx,ny,nz);

      temp = damping_tmp[idx] + lambda * lap_v[idx]
            + _srcVal(idx, depth_idx, u[idx] + lambda * damping_v[idx]);
      sum += temp * temp;
    }

    if(sum <= norm)  // when | F(u + \lambda v) | < | F(u) | stop
      return lambda;
  }
  
  return lambda;
}

/**
 * @brief calculating total of Jacob constraint
 * @param depth
 * @param shift
 */
REAL_T FASMultigrid::_dampingConstraintTotal(IDX_T depth, REAL_T shift)
{
  //calculating total  of Jacob equation after shift
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const u = u_h[depth_idx];
  fas_grid_t const damping_v = damping_v_h[depth_idx];
  fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];

  REAL_T total = 0.0;
  #pragma omp parallel for default(shared) private(i,j,k) reduction(+:total)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    
    total += _srcValDir(idx, depth_idx, u[idx])*(damping_v[idx] + shift)
      - jac_rhs[idx];
  }

  return total;
}

REAL_T FASMultigrid::_dampingConstraintDerivativeTotal(IDX_T depth, REAL_T shift)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(depth);

  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  fas_grid_t const u = u_h[depth_idx];


  REAL_T total = 0.0;
  #pragma omp parallel for default(shared) private(i,j,k) reduction(+:total)
  FAS_LOOP3_N(i,j,k,nx,ny,nz)
  {
    IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
    
    total += _srcValDir(idx, depth_idx, u[idx]);
  }

  return total;
}

/**
 * @brief find shift to Jacob equation to satisfy the zero integration constraint
 * @param depth
 */
void FASMultigrid::_shiftConstrainedDamping(IDX_T depth) 
{
  
  //shift the Jacob equation to satisy constraint, sometime does not helpful
  REAL_T eps = 0.0, shift = 0.0;
  REAL_T num, den, cnt=0;
  
  num = _dampingConstraintTotal(depth, 0.0);
  den = _dampingConstraintDerivativeTotal(depth, 0.0);
  shift = -num / den;
  
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  _shiftGridVals(damping_v_h[depth_idx], shift, nx * ny *nz );
}

/**
 * @brief perform Jacobian relaxation until a desired precision is reached
 * @details can be controled to use constrait or not, 
 * @param depth
 * @param norm of F(u)
 * @param parameter can control the converge speed
 * @param parameter can control the converge speed
 *
 */
bool FASMultigrid::_jacobianRelax(IDX_T depth, REAL_T norm, REAL_T C, IDX_T p)
{
  // C and p set the convergent speed of interation
  IDX_T i, j, k, s;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx], cnt = 0;

  REAL_T   norm_r = 1e100,
    dx = grid_length_x/nx, dy = grid_length_y/ny, dz = grid_length_z/nz,
    norm_pre, temp;

  fas_grid_t const u = u_h[depth_idx];
  fas_grid_t const damping_v = damping_v_h[depth_idx];
  fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];
  fas_grid_t const coarse_src = coarse_src_h[depth_idx];

  #pragma omp parallel for default(shared) private(i,j,k)
  FAS_LOOP3_N(i, j, k, nx, ny, nz)
  {
    damping_v[_gIdx(i,j,k,nx, ny, nz)] = 0.0;
  }
  
  while( norm_r >= std::min(pow(norm, (REAL_T)(p+1)) * C, norm)) 
  {
    //relax until the convergent condition got satisfy 
    norm_r = 0.0;
    norm_pre = 0.0;

    // TODO: parallelize
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

      damping_v[idx] = (
        (damping_v[_gIdx(i+1, j, k, nx, ny, nz)] + damping_v[_gIdx(i-1, j, k, nx, ny, nz)]) * dy*dy *dz*dz
        +(damping_v[_gIdx(i, j+1, k, nx, ny, nz)] + damping_v[_gIdx(i, j-1, k, nx, ny, nz)]) * dx*dx * dz*dz  
        +(damping_v[_gIdx(i, j, k+1, nx, ny, nz)] + damping_v[_gIdx(i, j, k-1, nx, ny, nz)]) * dx*dx * dy*dy
            - jac_rhs[idx] * dx *dx * dy * dy * dz * dz) /
        ( 2.0 * dy * dy * dz * dz + 2.0 * dx * dx * dz * dz + 2.0 * dx * dx * dy *dy
          - _srcValDir(idx, depth_idx, u[idx]) *dx *dx * dy * dy * dz * dz);
    }

    #pragma omp parallel for default(shared) private(i,j,k) reduction(+:norm_r)
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

      temp = _laplacian(damping_v, i, j, k, nx, ny, nz)
        + _srcValDir(idx, depth_idx, u[idx]) * damping_v[idx]
        - jac_rhs[idx];
            
      norm_r += temp * temp;
    }

    if(relax_scheme == inexact_newton_constrained)
      _shiftConstrainedDamping(depth); 
          
    cnt++;

    if(cnt > 500 && norm_r > norm_pre) 
    {
      //cannot solve Jacobian equation to precision needed
      std::cout << "Unable to achieve a precise enough solution within "
                << cnt << " iterations.\n";
      return false;
    }
  }

  return true;
}

/**
 * @brief find singularity in the solution, returns 1 if singularity found
 *
 * @param depth 
 */
bool FASMultigrid::_singularityExists(IDX_T depth)
{
  IDX_T i;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  fas_grid_t const u = u_h[depth_idx];
    
  for(i = 1; i < nx * ny * nz ; i++)
  {
    if ( _sign(u[i]) * _sign(u[0]) < 0 )
      return true;
  }

  return false;
}

/**
 * @brief relax u using the inexact Newton iterative method
 */
void FASMultigrid::_relaxSolution_GaussSeidel(IDX_T depth, IDX_T max_iterations)
{
  IDX_T i, j, k, s;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  REAL_T temp, lambda, norm,  dx = grid_length_x/nx, dy = grid_length_y/ny, dz = grid_length_z/nz;

  fas_grid_t const u = u_h[depth_idx];
  fas_grid_t const damping_v = damping_v_h[depth_idx];
  fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];
  fas_grid_t const coarse_src = coarse_src_h[depth_idx];
  fas_grid_t const damping_tmp = damping_tmp_h[depth_idx];
  fas_grid_t const lap_v = lap_v_h[depth_idx];

  for(s=0; s<max_iterations; ++s)
  {
    temp = _getMaxResidual(depth);
    // move this precision condition to the beginning in case
    // perfect initial geuss causes infinite number of
    // iterations for function: _jacobianRelax()
    if(temp < relaxation_precision) // set precision
      break;

    if(relax_scheme == inexact_newton
        || relax_scheme == inexact_newton_constrained)
    {
      norm = 0.0;
      
      #pragma omp parallel for default(shared) private(i,j,k) reduction(+:norm)
      FAS_LOOP3_N(i,j,k,nx,ny,nz)
      {
      
        IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

        temp = _laplacian(u, i, j, k, nx, ny, nz)
          + _srcVal(idx, depth_idx, u[idx]) - coarse_src[idx];

        // updating norm of F(u)
        norm += temp * temp;

        //updating _lap(u) - f
        damping_tmp[idx] = _laplacian(u, i, j, k, nx, ny, nz) - coarse_src[idx];

        //evalue jac_source at right hand side of Jacobian linear equation
        jac_rhs[idx] = -_evaluateEllipticEquationPt(depth, i, j, k) + coarse_src[idx];  
      }
   
      if( _jacobianRelax(depth, norm, 1, 0) == false) // relax to solve Jacobian linear equation
        break;
      
      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,nx,ny,nz)
      {
        //updating _lap(v) which help to calculate F(u + \lambda v)
        IDX_T idx = _gIdx(i,j,k,nx,ny,nz);
        lap_v[idx] = _laplacian(damping_v, i, j, k, nx, ny, nz);
      }

      //get damping parameter lambda
      lambda = _getLambda(depth, norm);

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,nx,ny,nz)
      {
        IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

        // update u according to lambda and v
        u[idx] += damping_v[idx] * lambda;
      }
            

    }
    else if (relax_scheme == newton)
    {  
      FAS_LOOP3_N(i,j,k,nx,ny,nz)
      {
        IDX_T idx = _gIdx(i, j, k, nx, ny, nz);

        // Gauss-Seidel step for equation of interest
        REAL_T current_residual = _laplacian(u, i, j, k, nx, ny, nz)
          + _srcVal(idx, depth_idx, u[idx]) - coarse_src[idx];

        u[idx] -= current_residual / (
            -2.0/dx/dx - 2.0/dy/dy - 2.0/dz/dz
            + _srcValDir(idx, depth_idx, u[idx])
          );
      }
    }
  } // end iterations loop

}

/**
 * @brief Method to initialize internal variables, allocate memory
 * 
 * @param[in]  number grid points in x direction
 * @param[in]  number grid points in y direction
 * @param[in]  number grid points in z direction
 * @param[in]  grid length in x direction
 * @param[in]  grid length in y direction
 * @param[in]  grid length in z direction
 * @param[in]  number of multigrid layers
 * @param[in]  maximum iterations when relaxing
 * @param[in]  relaxation scheme (enum)
 */
void FASMultigrid::_initializeMultigrid(IDX_T grid_num_x_in, IDX_T grid_num_y_in, IDX_T grid_num_z_in,
       REAL_T grid_length_x_in, REAL_T grid_length_y_in, REAL_T grid_length_z_in,
					IDX_T max_depth_in, IDX_T max_relax_iters_in, relax_t relax_scheme_in, REAL_T eps)
{
  IDX_T depth_idx, points;

  relax_scheme = relax_scheme_in;
  max_relax_iters = max_relax_iters_in;
  max_depth = max_depth_in;
  min_depth = 1;
  max_depth_idx = _dIdx(max_depth);
  min_depth_idx = _dIdx(min_depth);
  relaxation_precision = eps;
  
  if( grid_num_x_in % _2toPwr(max_depth) != 0
    || grid_num_y_in % _2toPwr(max_depth) != 0
    || grid_num_z_in % _2toPwr(max_depth) != 0)
  {
    std::cout << "Warning: Grid size is not divisible by 2^"
      << max_depth << ".\n";
  }

  total_depths = max_depth - min_depth + 1;
  grid_length_x = grid_length_x_in;
  grid_length_y = grid_length_y_in;
  grid_length_z = grid_length_z_in;

  u_h = new fas_grid_t[total_depths];

  coarse_src_h = new fas_grid_t[total_depths];
  tmp_h = new fas_grid_t[total_depths];
  
  damping_tmp_h = new fas_grid_t[total_depths];
  lap_v_h = new fas_grid_t[total_depths];
  damping_v_h = new fas_grid_t[total_depths];
  
  nx_h = new IDX_T[total_depths];
  ny_h = new IDX_T[total_depths];
  nz_h = new IDX_T[total_depths];

  jac_rhs_h = new fas_grid_t[total_depths];
  
  for(IDX_T depth = max_depth; depth >= min_depth; --depth)
  {
    depth_idx = _dIdx(depth);

    if(depth_idx == _dIdx(max_depth))
    {
      nx_h[depth_idx] = grid_num_x_in;
      ny_h[depth_idx] = grid_num_y_in;
      nz_h[depth_idx] = grid_num_z_in;
    }
    else
    {
      nx_h[depth_idx] = nx_h[depth_idx+1] / 2 + (nx_h[depth_idx+1] % 2);
      ny_h[depth_idx] = ny_h[depth_idx+1] / 2 + (ny_h[depth_idx+1] % 2);
      nz_h[depth_idx] = nz_h[depth_idx+1] / 2 + (nz_h[depth_idx+1] % 2);
    }
    
    points = nx_h[depth_idx] * ny_h[depth_idx] * nz_h[depth_idx];

    u_h[depth_idx] = new REAL_T[points];

    coarse_src_h[depth_idx] = new REAL_T[points];
    tmp_h[depth_idx] = new REAL_T[points];
  
    damping_tmp_h[depth_idx] = new REAL_T[points];
    lap_v_h[depth_idx] = new REAL_T[points];
    damping_v_h[depth_idx] = new REAL_T[points];
  
    jac_rhs_h[depth_idx] = new REAL_T[points];

    _zeroGrid(u_h[depth_idx], points);
    _zeroGrid(coarse_src_h[depth_idx], points);
    _zeroGrid(tmp_h[depth_idx], points);
    _zeroGrid(damping_tmp_h[depth_idx], points);
    _zeroGrid(lap_v_h[depth_idx], points);
    _zeroGrid(damping_v_h[depth_idx], points);
  
    _zeroGrid(jac_rhs_h[depth_idx], points);
  }
}

void FASMultigrid::_printStrip(fas_heirarchy_t out_h, IDX_T depth)
{
  IDX_T i;
  IDX_T depth_idx = _dIdx(depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  fas_grid_t const out = out_h[_dIdx(depth)];
  std::cout << std::fixed << std::setprecision(15) << "Values: { ";
  for(i=0; i<nx; i++)
  {
    IDX_T idx = _gIdx(i,nx/4,ny/4, nx, ny, nz);
    std::cout << out[idx];
    std::cout << ", ";
  }
  std::cout << "}\n";
}

void FASMultigrid::_printAll(fas_heirarchy_t out_h, IDX_T depth)
{
  IDX_T depth_idx = _dIdx(depth);
  fas_grid_t const m = out_h[_dIdx(depth)];
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  std::cout << "{";
  
  for(int i = 0; i < nx; i++)
  {
    std::cout << "{";
    for(int j = 0; j < ny; j++)
    {
      std::cout << "{";
      std::cout<<std::fixed<<m[_gIdx(i,j,0,nx,ny,nz)];
      for(int k = 1; k < nz; k++)
      {
        std::cout<<std::fixed<<","<<m[_gIdx(i,j,k,nx,ny,nz)];
      }
      std::cout << "}";
      if(j != ny-1)
        std::cout << ",";
    }
    std::cout << "}";
    if(i != nx-1)
      std::cout<<',';
  }
  std::cout << "}";
}

/**
 * @brief Full Constructor
 * @details Initialize internal variables, allocate memory
 * 
 * @param[in]  number grid points in x direction
 * @param[in]  number grid points in y direction
 * @param[in]  number grid points in z direction
 * @param[in]  grid length in x direction
 * @param[in]  grid length in y direction
 * @param[in]  grid length in z direction
 * @param[in]  number of multigrid layers
 * @param[in]  maximum iterations when relaxing
 * @param[in]  relaxation scheme (enum)
 */
FASMultigrid::FASMultigrid(IDX_T grid_num_x_in, IDX_T grid_num_y_in, IDX_T grid_num_z_in,
  REAL_T grid_length_x_in, REAL_T grid_length_y_in, REAL_T grid_length_z_in,
			   IDX_T max_depth_in, IDX_T max_relax_iters_in, relax_t relax_scheme_in, REAL_T eps)
{
  _initializeMultigrid(grid_num_x_in, grid_num_y_in, grid_num_z_in,
       grid_length_x_in, grid_length_y_in, grid_length_z_in,
		       max_depth_in, max_relax_iters_in, relax_scheme_in, eps);
} // constructor

/**
 * @brief Partial Constructor, with some defaults
 * 
 * @param[in]  number grid points in each direction
 * @param[in]  grid length in each direction
 * @param[in]  number of multigrid layers
 */
FASMultigrid::FASMultigrid(IDX_T grid_num_in, REAL_T grid_length_in, IDX_T max_depth_in, REAL_T eps)
{
  relax_t relax_scheme_in = relax_t::inexact_newton_constrained;
  IDX_T max_relax_iters_in = 5;

  _initializeMultigrid(
    grid_num_in, grid_num_in, grid_num_in,
    grid_length_in, grid_length_in, grid_length_in,
    max_depth_in, max_relax_iters_in, relax_scheme_in, eps
  );
} // constructor


/**
 * @brief      destructor
 */
FASMultigrid::~FASMultigrid()
{
  for(IDX_T depth = max_depth; depth >= min_depth; --depth)
  {
    IDX_T depth_idx = _dIdx(depth);
    
    // just delete these for now; most of the memory is allocated in these
    
    delete [] u_h[depth_idx];
    delete [] coarse_src_h[depth_idx];
    delete [] tmp_h[depth_idx];
    delete [] damping_tmp_h[depth_idx];
    delete [] lap_v_h[depth_idx];
    delete [] damping_v_h[depth_idx];
    delete [] jac_rhs_h[depth_idx];

    for( IDX_T I = 0; I < rho_num; I++)
    {
      delete [] rho_h[I][depth_idx];
    }

  }
}


void FASMultigrid::build_rho(IDX_T src_num_in, IDX_T u_exp_in[])
{
  build_rho(src_num_in);
  for(IDX_T i = 0; i < src_num_in; ++i)
  {
    u_exp[i] = u_exp_in[i];
  }
}

/**
 * @brief allocate the space of rho with src_num of sources
 *
 * @param number of source terms  
 */  
void FASMultigrid::build_rho(IDX_T src_num)
{
  rho_num = src_num;
  rho_h = new  fas_heirarchy_t[rho_num];
  u_exp = new IDX_T[rho_num];
  for( IDX_T I = 0; I < rho_num; I++)
  {
    rho_h[I] = new fas_grid_t[total_depths];
    for(IDX_T depth = min_depth; depth <=max_depth; depth++)
    {
      IDX_T depth_idx = _dIdx(depth);
  
      //IDX_T n = _2toPwr(depth);
      IDX_T points = nx_h[depth_idx] * ny_h[depth_idx] * nz_h[depth_idx];
      rho_h[I][depth_idx] = new REAL_T[points]; 
      _zeroGrid(rho_h[I][depth_idx], points);
    }

  }
}

/**
 * @brief      Initialize matter source on all grids
 *
 */
void FASMultigrid::initializeRhoHeirarchy()
{
  // initialize values on fine grid
  IDX_T points = nx_h[_dIdx(max_depth)] * ny_h[_dIdx(max_depth)] * nz_h[_dIdx(max_depth)];
  // restrict supplied rho to coarser grids
  for(IDX_T I = 0; I < rho_num; I++)
  {
    for(IDX_T depth = max_depth; depth > min_depth; --depth)
    {
      _restrictFine2coarse(rho_h[I], depth);
    }
  }
}
  
  
void FASMultigrid::VCycle()
{

  // initial residual
  _relaxSolution_GaussSeidel(max_depth, max_relax_iters);

  std::cout << "  Initial max. residual on fine grid is: "
      << _getMaxResidual(max_depth) << ".\n" << std::flush;

  IDX_T depth, coarse_depth;
  // "downstroke": restrict until coarsest grid is reached
  // restricts phi, computes coarse_src using tmp_h

  for(depth = max_depth; min_depth < depth; --depth)
  {
    _computeCoarseRestrictions(depth);
  }

  // back up approximate solutionp in tmp (needed for subsequent error calc)
  _copyGrid(u_h, tmp_h, min_depth);

  // prolongate solution to finest grid
  // "upward" stroke
  for(coarse_depth = min_depth; coarse_depth < max_depth; coarse_depth++)
  {
    // relax "true" solution in u_h
    _relaxSolution_GaussSeidel(coarse_depth, max_relax_iters);
    
    std::cout << "    Working on upward stroke at depth " << coarse_depth
  << "; residual after solving is: "
  << _getMaxResidual(coarse_depth) << ".\n" << std::flush;

    // tmp should hold appx. soln; convert to error
    _changeApproximateSolutionToError(tmp_h, u_h, coarse_depth);
    // tmp should hold error
    _correctFineFromCoarseErr_Err2Appx(tmp_h, u_h, coarse_depth+1);
    // tmp now holds appx. soln on finer grid;
    // phi_h now holds corrected solution on finer grid
  }

  // final relaxation
  _relaxSolution_GaussSeidel(max_depth, max_relax_iters);
  std::cout << "  Final max. residual on fine grid is: "
      << _getMaxResidual(max_depth) << ".\n" << std::flush;
}

void FASMultigrid::VCycles(IDX_T num_cycles)
{
  for(int cycle=0; cycle < num_cycles; ++cycle)
  {
    VCycle();
  }

  IDX_T fine_idx = _dIdx(max_depth);
  IDX_T n_fine_x = nx_h[fine_idx], n_fine_y = ny_h[fine_idx], n_fine_z = nz_h[fine_idx];
  fas_grid_t const fine_u = u_h[fine_idx];
  REAL_T avg_sol = _averageGrid(fine_u, n_fine_x * n_fine_y * n_fine_z);
  REAL_T min_sol = _minGrid(fine_u, n_fine_x * n_fine_y * n_fine_z);
  REAL_T max_sol = _maxGrid(fine_u, n_fine_x * n_fine_y * n_fine_z);

  _relaxSolution_GaussSeidel(max_depth, 10);
  std::cout << "  Final solution residual is: "
      << _getMaxResidual(max_depth) << "\n" << std::flush;
  std::cout << "  With average / min / max phi: "
      << avg_sol << " / " << min_sol << " / " << max_sol << ".\n" << std::flush;

  if(_singularityExists(max_depth))
    std::cout << "  Warning! Solution crosses 0; solution may be singular at some points.\n";
  else
    std::cout << "  Solution stays positive or negative (no singularities seem to exist).\n";
}

/**
 * @brief set trial solution for different type
 *
 * @param type of different rho, see parameter explanation for add_poly_srcs();
 */
void FASMultigrid::setTrialSolution(IDX_T type)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(max_depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];

  fas_grid_t const u = u_h[max_depth_idx];

  if(type == 0)
  {
    // frequency and phase of waves
    REAL_T n1 = 1.0, n2 = 1.0, n3 = 1.0;
    REAL_T phi1 = 0, phi2 = 0, phi3 = 0;

    // generate trial solution u
  
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i,j,k,nx,ny,nz);
      u[idx] = 2.0;
    }
  }
  else if( type == 1 )
  {
    // frequency and phase of waves
    REAL_T n1 = 1.0, n2 = 1.0, n3 = 1.0;
    REAL_T phi1 = 0, phi2 = 0, phi3 = 0;

    // generate trial solution u
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i,j,k,nx,ny,nz);

      // generate trial solution
      u[idx] = 1.0 - std::sin( 2.0 * 3.14159265 * n1 * (REAL_T)i/ (nx) + phi1)
        * std::sin( 2.0 * 3.14159265 * n2 * (REAL_T)j/ (ny) + phi2)
        * std::sin( 2.0 * 3.14159265 * n3 * (REAL_T)k/ (nz) + phi3)/10.0;
    }
  }    
}

 /**
 * @brief add source term to the elliptical equation
 * @details 
 * type = 0: This is the case in the paper: 1511.05143 where \rho is given by a inhomogeneous
 * scalar field. The solver can deal with this case well with inexact Newton iteration
 * with constraint to speedup (relax_scheme = 1).
 *
 * type = 1: This is the case for the case with analytical solution (we can get this 
 * by calculating -\lap(u) / u^5 for a given inhomogenous function of u).
 * In this case, for a very good guess ( u(exact) + (rand()/RAND_MAX - 0.5) / 100 ),
 * it converges to precision 0.0002 without applying constraint to speed up 
 * (actually, applying constraint makes thing worse in this case). 
 * For worse guess, it may converge to zero.
 * 
 * @param u_guess guess for \u on the finest grid
 */
void FASMultigrid::add_poly_srcs(IDX_T type)
{
  IDX_T i, j, k;
  IDX_T depth_idx = _dIdx(max_depth);
  IDX_T nx = nx_h[depth_idx], ny = ny_h[depth_idx], nz = nz_h[depth_idx];
  REAL_T dx = grid_length_x/nx, dy = grid_length_y/ny, dz = grid_length_z/nz;
  REAL_T K = 0.0, delta_phi = 0.01 , Lambda = 0.00001;//set potential equals constant \Lambda
  if(type == 0) //type == 0 for constant K case
  {
    // two sources for constant K: \Psi^5 polynomial term and \Psi^1 term
    build_rho(2);

    // initialize \Psi^1 term
    u_exp[0] = 1;
    FAS_LOOP3_N(i, j, k, nx, ny, nz)
    {
      IDX_T idx = _gIdx(i, j ,k, nx, ny, nz);
      rho_h[0][depth_idx][idx] =  PI * (
        _Pwr2( delta_phi * 2.0 * PI / (grid_length_x) )
          * _Pwr2(  -std::sin(2.0 * PI *(REAL_T)i * dx / grid_length_x
                              + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                    + std::sin(-2.0 * PI *(REAL_T)i * dx /grid_length_x
                              + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )
        + _Pwr2( delta_phi* 2.0 * PI /(grid_length_y) )
          * _Pwr2(-std::sin(2.0 * PI *(REAL_T)j * dy / grid_length_y
                              + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                  + std::sin(-2.0 * PI *(REAL_T)j * dy / grid_length_y
                              + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )
        +  _Pwr2( delta_phi* 2.0 * PI /(grid_length_z) )
          * _Pwr2(-std::sin(2.0 * PI *(REAL_T)k * dz / grid_length_z
                            + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                  + std::sin(-2.0 * PI *(REAL_T)k * dz /grid_length_z
                            + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )
      );
    }

    // initialize \Psi^5 term
    u_exp[1] = 5;
    FAS_LOOP3_N(i, j, k, nx, ny, nz)
    {
      IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
      K += Lambda * dx * dy * dz/ 4.0 + (
        _Pwr2( delta_phi* 2.0 * PI /(grid_length_x) )
          * _Pwr2( -std::sin(2.0 * PI *(REAL_T)i * dx / grid_length_x
                            + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                  + std::sin(-2.0 * PI *(REAL_T)i * dx /grid_length_x
                            + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )
        + _Pwr2( delta_phi* 2.0 * PI /(grid_length_y) )
          * _Pwr2( -std::sin(2.0 * PI *(REAL_T)j * dy / grid_length_y
                        + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                    + std::sin(-2.0 * PI *(REAL_T)j * dy / grid_length_y
                        + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )

        + _Pwr2( delta_phi* 2.0 * PI /(grid_length_z) )
          * _Pwr2(-std::sin(2.0 * PI *(REAL_T)k * dz / grid_length_z
                          + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI)
                  + std::sin(-2.0 * PI *(REAL_T)k * dz /grid_length_z
                          + (REAL_T)std::rand()/RAND_MAX * 2.0 * PI) )
      ) * dx * dy * dz / 8.0;
    }

    K = - K / (grid_length_x * grid_length_y * grid_length_z);
    
    FAS_LOOP3_N(i, j, k, nx, ny, nz)
    {
      IDX_T idx = _gIdx(i, j, k, nx, ny, nz);
      rho_h[1][depth_idx][idx] = K + PI * Lambda;
    }

    std::cout << "    K + PI * Lambda = " << K + PI * Lambda << "\n";
    initializeRhoHeirarchy();      
  }
  else if(type == 1)
  {
    // set rho = /lap u / u^5, which has exact solution: u,
    // then perturb u, solve, and compare the result
    build_rho(1);
    u_exp[0] = 5;
    rho_num = 1;
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i,j,k,nx,ny,nz);
      rho_h[0][depth_idx][idx] = -_laplacian(u_h[depth_idx], i, j, k, nx, ny, nz)
        / std::pow(u_h[depth_idx][idx], 5.0);
    }

    initializeRhoHeirarchy();
    FAS_LOOP3_N(i,j,k,nx,ny,nz)
    {
      IDX_T idx = _gIdx(i,j,k,nx,ny,nz);
      u_h[depth_idx][idx] +=( (REAL_T) rand()/RAND_MAX - 0.5 )/100;
    }
  }
  
}

void FASMultigrid::printSolutionStrip(IDX_T depth)
{
  _printStrip(u_h, depth);
}

void FASMultigrid::printSourceStrip(IDX_T rho_n, IDX_T depth)
{
  _printStrip(rho_h[rho_n], depth);
}

void FASMultigrid::setPolySrcAtPt(IDX_T i, IDX_T j, IDX_T k, IDX_T n, REAL_T value)
{
  IDX_T idx = _gIdx(i, j, k,
    nx_h[max_depth_idx], ny_h[max_depth_idx], nz_h[max_depth_idx]);

  rho_h[n][max_depth_idx][idx] = value;
}

REAL_T * FASMultigrid::getSolution()
{
  return u_h[max_depth_idx];
}

