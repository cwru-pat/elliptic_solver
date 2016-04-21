#ifndef FAS_MULTIGRID_H
#define FAS_MULTIGRID_H

#include <omp.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#define FAS_LOOP3_N(i, j, k, n) \
  for(i=0; i<n; ++i) \
    for(j=0; j<n; ++j) \
      for(k=0; k<n; ++k)

/**
 * @brief Full Approximation Storage ("FAS") Multigrid solver class
 * @details 
 * see: https://computation.llnl.gov/casc/people/henson/postscript/UCRL_JC_150259.pdf
 * section 3 for a very concise, clean description of the FAS Scheme.
 *  
 * In this implementation, the following step is done for restriction:
 *   restrict a fine grid approximation to a coarser grid:
 *     - compute the approximate solution on the coarse grid:
 *        v^2h = I^2h_h v^h
 *     - compute the source term on the coarse grid (coarse_src):
 *        f^2h = A^2h ( v^2h ) + I^2h_h ( f^h - A^h ( v^h ) )
 * And the following is done for prolongation:
 *     - compute the error
 *  
 * The set of grids (array of arrays) is referred to as a "heirarchy" of grids,
 * where each grid has some "depth". The size of the grid is N^3 = ( 2^(depth) )^3.
 */
template <typename REAL_T, typename IDX_T>
class FASMultigrid
{
  private:

    typedef REAL_T * fas_grid_t; // grid (array) type
    typedef fas_grid_t * fas_heirarchy_t; // heirarchy type (set of some grids)

    // define a heirarchy of references to grids
    fas_heirarchy_t tmp_h, // reusable grid for storing intermediate calculations
                    coarse_src_h, // multigrid source term
                    psi_h, // field seeking a solution for
                    appx_psi_h, // field containing an approximate solution
                    rho_h; // matter source in elliptic pde

    IDX_T max_depth, max_depth_idx;
    IDX_T min_depth, min_depth_idx;
    IDX_T total_depths;
    REAL_T grid_length;

    /**
     * @brief indexing scheme of a grid heirarchy
     * @description return index of grid at a particular depth
     *  in a grid heirarchy
     * 
     * @param depth "depth" of grid
     * @return index
     */
    IDX_T _dIdx(IDX_T depth)
    {
      return depth - min_depth;
    }

    /**
     * @brief      Return array index of a point located at (i,j,k) on a
     * grid with periodic boundaries, of size n^3.
     *
     * @param[in]  i     label of points in x-dir
     * @param[in]  j     label of points in y-dir
     * @param[in]  k     label of points in z-dir
     * @param[in]  n     Assuming dimension of n^3
     * 
     * @return     array index
     */
    IDX_T _gIdx(IDX_T i, IDX_T j, IDX_T k, IDX_T n)
    {
      return ((i+n)%n)*n*n + ((j+n)%n)*n + (k+n)%n;
    }

    /**
     * @brief      initialize a grid to 0
     *
     * @param      grid    grid (array) to initialize
     * @param[in]  points  # points in array / on grid
     */
    void _zeroGrid(fas_grid_t grid, IDX_T points)
    {
      for(IDX_T i=0; i < points; i++)
        grid[i] = 0;
    }

    /** TODO: doc */
    REAL_T _totalGrid(fas_grid_t grid, IDX_T points)
    {
      REAL_T total = 0;
      for(IDX_T i=0; i < points; i++)
        total += grid[i];
      return total;
    }

    /** TODO: doc */
    void _shiftGridVals(fas_grid_t grid, REAL_T shift, IDX_T points)
    {
      #pragma omp parallel for
      for(IDX_T i=0; i < points; i++)
        grid[i] += shift;
    }

    /**
     * @brief compute power of 2
     * 
     * @param pwr power to raise 2 to
     * @return 2^pwr
     */
    IDX_T _2toPwr(IDX_T pwr)
    {
      return 1<<pwr;
    }

    /**
     * @brief compute integer number to power of 3
     * 
     * @param number to raise to ^3
     * @return pwr^3
     */
    IDX_T _Pwr3(IDX_T num)
    {
      return num*num*num;
    }

    /**
     * @brief      Take a laplacian stencil on "grid" of size n^3 at point i,j,k
     *
     * @return     laplacian stencil
     */
    REAL_T _laplacian(fas_grid_t grid, IDX_T i, IDX_T j, IDX_T k, IDX_T n)
    {
      REAL_T dx = grid_length/n;

      return  (
        grid[_gIdx(i+1, j, k, n)] + grid[_gIdx(i-1, j, k, n)] +
        grid[_gIdx(i, j+1, k, n)] + grid[_gIdx(i, j-1, k, n)] +
        grid[_gIdx(i, j, k+1, n)] + grid[_gIdx(i, j, k-1, n)]
        - 6.0*grid[_gIdx(i, j, k, n)]
      )/(dx*dx);
    }

    /**
     * @brief "restrict" a fine grid to coarser grid
     * @details Restriction scheme:
     *  TODO: document
     * 
     * @param field_heirarchy field to restrict
     * @param fine_depth "depth" of finer grid
     */
    void _restrictFine2coarse(fas_heirarchy_t grid_heirarchy, IDX_T fine_depth)
    {
      // restrict scheme: (1 given cell)*(1/8) + (6 adjacent "faces") * (1/16)
      //    + (12 adjacent "edges") * (1/32) + (8 adjacent "corners") * (1/64)

      IDX_T n_fine = _2toPwr(fine_depth);
      IDX_T n_coarse = _2toPwr(fine_depth-1);
      IDX_T fine_idx = _dIdx(fine_depth);
      IDX_T coarse_idx = _dIdx(fine_depth-1);

      fas_grid_t const fine_grid = grid_heirarchy[fine_idx];
      fas_grid_t const coarse_grid = grid_heirarchy[coarse_idx];

      IDX_T i, j, k; // coarse grid iterator
      IDX_T fi, fj, fk; // fine grid indexes

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n_coarse)
      {
        fi = i*2;
        fj = j*2;
        fk = k*2;

        coarse_grid[_gIdx(i,j,k,n_coarse)] =
          0.125 * fine_grid[_gIdx(fi,fj,fk,n_fine)]
          + 0.0625 * (
            fine_grid[_gIdx(fi+1,fj,fk,n_fine)] +
            fine_grid[_gIdx(fi,fj+1,fk,n_fine)] +
            fine_grid[_gIdx(fi,fj,fk+1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj,fk,n_fine)] +
            fine_grid[_gIdx(fi,fj-1,fk,n_fine)] +
            fine_grid[_gIdx(fi,fj,fk-1,n_fine)]
          )
          + 0.03125 * (
            fine_grid[_gIdx(fi+1,fj+1,fk,n_fine)] +
            fine_grid[_gIdx(fi+1,fj-1,fk,n_fine)] +
            fine_grid[_gIdx(fi-1,fj+1,fk,n_fine)] +
            fine_grid[_gIdx(fi-1,fj-1,fk,n_fine)] +
            fine_grid[_gIdx(fi+1,fj,fk+1,n_fine)] +
            fine_grid[_gIdx(fi+1,fj,fk-1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj,fk+1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj,fk-1,n_fine)] +
            fine_grid[_gIdx(fi,fj+1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi,fj+1,fk-1,n_fine)] +
            fine_grid[_gIdx(fi,fj-1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi,fj-1,fk-1,n_fine)] 
          )
          + 0.015625 * (
            fine_grid[_gIdx(fi+1,fj+1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi+1,fj+1,fk-1,n_fine)] +
            fine_grid[_gIdx(fi+1,fj-1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj+1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi+1,fj-1,fk-1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj+1,fk-1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj-1,fk+1,n_fine)] +
            fine_grid[_gIdx(fi-1,fj-1,fk-1,n_fine)]
          );
      } // end loop
    } // end restrict_fine2coarse

    /**
     * @brief interpolate a coarse grid to a finer grid
     *  TODO: document
     */
    void _interpolateCoarse2fine(fas_heirarchy_t grid_heirarchy, IDX_T coarse_depth) 
    {
      IDX_T n_coarse = _2toPwr(coarse_depth);
      IDX_T n_fine = _2toPwr(coarse_depth+1);
      IDX_T coarse_idx = _dIdx(coarse_depth);
      IDX_T fine_idx = _dIdx(coarse_depth+1);

      fas_grid_t const coarse_grid = grid_heirarchy[coarse_idx];
      fas_grid_t const fine_grid = grid_heirarchy[fine_idx];

      IDX_T i, j, k;
      IDX_T fi, fj, fk;

      _zeroGrid(fine_grid, _Pwr3(n_fine));

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n_coarse)
      {
        fi = i*2;
        fj = j*2;
        fk = k*2;

        REAL_T cc = coarse_grid[_gIdx(i,j,k,n_coarse)];
        fine_grid[_gIdx(fi,fj,fk,n_fine)] += cc;

        fine_grid[_gIdx(fi+1,fj,fk,n_fine)] += cc/2.0;
        fine_grid[_gIdx(fi,fj+1,fk,n_fine)] += cc/2.0;
        fine_grid[_gIdx(fi,fj,fk+1,n_fine)] += cc/2.0;
        fine_grid[_gIdx(fi-1,fj,fk,n_fine)] += cc/2.0;
        fine_grid[_gIdx(fi,fj-1,fk,n_fine)] += cc/2.0;
        fine_grid[_gIdx(fi,fj,fk-1,n_fine)] += cc/2.0;

        fine_grid[_gIdx(fi+1,fj+1,fk,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi+1,fj-1,fk,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi-1,fj+1,fk,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi-1,fj-1,fk,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi+1,fj,fk+1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi+1,fj,fk-1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi-1,fj,fk+1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi-1,fj,fk-1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi,fj+1,fk+1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi,fj+1,fk-1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi,fj-1,fk+1,n_fine)] += cc/4.0;
        fine_grid[_gIdx(fi,fj-1,fk-1,n_fine)] += cc/4.0;
                
        fine_grid[_gIdx(fi+1,fj+1,fk+1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi+1,fj+1,fk-1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi+1,fj-1,fk+1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi-1,fj+1,fk+1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi+1,fj-1,fk-1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi-1,fj+1,fk-1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi-1,fj-1,fk+1,n_fine)] += cc/8.0;
        fine_grid[_gIdx(fi-1,fj-1,fk-1,n_fine)] += cc/8.0;
      }
    }

    /**
     * @brief      Evaluate elliptic operator at given depth
     *
     * @param[in]  depth     depth to evaluate at
     * @param      result_h  grid to store result on
     */
    void _evaluateEllipticEquation(fas_heirarchy_t result_h, IDX_T depth)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const result = result_h[depth_idx];

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
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
    REAL_T _evaluateEllipticEquationPt(IDX_T depth, IDX_T i, IDX_T j, IDX_T k)
    {
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const psi = psi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];

      IDX_T idx = _gIdx(i, j, k, n);
      return _laplacian(psi, i, j, k, n) + rho[idx] * std::pow(psi[idx], 5.0);
    }

    /**
     * @brief Find value of A that gives a root of the function:
     * f(A) = \integral (\rho * (\psi + A)^5 - coarse_src  ) d V
     * and offset phi by this
     *
     * @param depth depth to compute at
     * @return "A"
     */
    void _shiftConstrainedFieldMonopole(IDX_T depth)
    {
      REAL_T eps = 0.0, shift = 0.0;
      REAL_T num, den;

      do
      {
        num = _monopoleConstraintTotal(depth, shift);
        den = _monopoleConstraintDerivativeTotal(depth, shift);

        shift -= num/den;
        if( fabs(fabs(num/den) - eps) < 1e-9 )
          break;

        eps = fabs(num/den);

      } while(1);

      IDX_T n = _2toPwr(depth);
      IDX_T depth_idx = _dIdx(depth);
      _shiftGridVals(psi_h[depth_idx], shift, _Pwr3(n));
    }

    REAL_T _monopoleConstraintTotal(IDX_T depth, REAL_T shift)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const psi = psi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += rho[idx] * std::pow(psi[idx] + shift, 5.0) - coarse_src[idx];
      }

      return total;
    }

    REAL_T _monopoleConstraintDerivativeTotal(IDX_T depth, REAL_T shift)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const psi = psi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += rho[idx] * std::pow(psi[idx] + shift, 4.0);
      }

      return 5.0*total;
    }

    /**
     * @brief      Computes residual, stores result in "tmp" heirarchy
     *
     * @param[in]  depth  depth to compute residual at
     * @param      residual_h  heirarchy of arrays to store residual in
     */
    void _computeResidual(fas_heirarchy_t residual_h, IDX_T depth)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const psi = psi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];
      fas_grid_t const residual = residual_h[depth_idx];

      // intermediately store elliptic operator result on residual grid
      _evaluateEllipticEquation(residual_h, depth);
      // residual is coarse_src - elliptic operator
      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        residual[idx] = coarse_src[idx] - residual[idx];
      }
    }

    /**
     * @brief      Computes max residual, uses intermediate tmp_h
     *
     * @param[in]  depth  depth to compute residual at
     * @return residual
     */
    REAL_T _getMaxResidual(IDX_T depth)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T max_residual = 0.0;

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
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
     * @brief      Compute coarse_src and psi on a coarser grid
     * using tmp_h for some computations
     *
     * @param[in]  fine_depth  depth of grid to coarsen
     */
    void _computeCoarseRestrictions(IDX_T fine_depth)
    {
      IDX_T i, j, k;

      // restrict approximate solution on coarse grid
      _restrictFine2coarse(psi_h, fine_depth);

      // compute residual on fine grid; intermediately store the result
      // in the tmp grid
      _computeResidual(tmp_h, fine_depth);
      // restrict the residual to the coarse grid
      _restrictFine2coarse(tmp_h, fine_depth);
      // compute elliptic operator on coarse grid; store in source
      _evaluateEllipticEquation(coarse_src_h, fine_depth-1);

      // add in restricted residual to coarse source;
      // coarse source is then set.
      IDX_T n_coarse = _2toPwr(fine_depth-1);
      IDX_T coarse_idx = _dIdx(fine_depth-1);
      fas_grid_t const coarse_src = coarse_src_h[coarse_idx];
      fas_grid_t const tmp = tmp_h[coarse_idx];
      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n_coarse)
      {
        IDX_T idx = _gIdx(i, j, k, n_coarse);
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
    void _changeApproximateSolutionToError(fas_heirarchy_t appx_to_err_h,
      fas_heirarchy_t exact_soln_h, IDX_T depth)
    {
      IDX_T i, j, k;

      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const appx_to_err = appx_to_err_h[depth_idx];
      fas_grid_t const exact_soln = exact_soln_h[depth_idx];

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
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
    void _correctFineFromCoarseErr_Err2Appx(fas_heirarchy_t err2appx_h,
      fas_heirarchy_t appx_soln_h, IDX_T fine_depth)
    {
      IDX_T i, j, k;
      IDX_T coarse_depth = fine_depth-1;

      IDX_T fine_depth_idx = _dIdx(fine_depth);
      IDX_T n_fine = _2toPwr(fine_depth);

      _interpolateCoarse2fine(err2appx_h, coarse_depth);

      fas_grid_t const err2appx = err2appx_h[fine_depth_idx];
      fas_grid_t const appx_soln = appx_soln_h[fine_depth_idx];

      #pragma omp parallel for default(shared) private(i,j,k)
      FAS_LOOP3_N(i,j,k,n_fine)
      {
        IDX_T idx = _gIdx(i, j, k, n_fine);
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
    void _copyGrid(fas_heirarchy_t from_h, fas_heirarchy_t to_h, IDX_T depth)
    {
      IDX_T depth_idx = _dIdx(depth);
      IDX_T points = _Pwr3(_2toPwr(depth));

      fas_grid_t const from = from_h[depth_idx];
      fas_grid_t const to = to_h[depth_idx];

      std::copy(from, from + points, to);
    }

    void _relaxSolution_GaussSeidel(IDX_T depth)
    {
      _relaxSolution_GaussSeidel(depth, 1);
    }

    void _relaxSolution_GaussSeidel(IDX_T depth, IDX_T iterations)
    {

      // relax psi using the constraint equation
      IDX_T i, j, k, s;

      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);
      REAL_T dx = grid_length/n;

      fas_grid_t const psi = psi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T max_residual = 0.0;
      REAL_T prev_max_residual = 1e100;
      // run some # iterations
      for(s=0; s<iterations; ++s)
      {
        if(s==iterations/2)
          _shiftConstrainedFieldMonopole(depth);

        // Note that this method has an implicit race condition
        #pragma omp parallel for default(shared) private(i,j,k)
        FAS_LOOP3_N(i,j,k,n)
        {
          IDX_T idx = _gIdx(i, j, k, n);

          // Gauss-Seidel step for equation of interest
          REAL_T current_residual = _laplacian(psi, i, j, k, n)
            + rho[idx]*std::pow(psi[idx], 5.0) - coarse_src[idx];
          psi[idx] -= current_residual/(
              -6.0/dx/dx + 5.0*rho[idx]*std::pow(psi[idx], 4.0)
            );

          // information about current residual
          #pragma omp critical
          {
           if(current_residual > max_residual)
             max_residual = current_residual;
          }
        }
        // make sure we are still converging using this method...
        if(max_residual >= prev_max_residual) return;
        prev_max_residual = max_residual;
      }
    }

  public:

    /**
     * @brief Constructor
     * @details Initialize internal variables, allocate memory
     * 
     * @param[in]  max_depth_in "depth" of finest grid (size is n = 2^max_depth)
     * @param[in]  min_depth_in "depth" of coarsest grid (size is n = 2^min_depth)
     * @param[in]  grid_length physical length of a side of the grid
     */
    FASMultigrid(IDX_T max_depth_in, IDX_T min_depth_in, REAL_T grid_length_in)
    {
      grid_length = grid_length_in;

      max_depth = max_depth_in;
      min_depth = min_depth_in;
      max_depth_idx = _dIdx(max_depth);
      min_depth_idx = _dIdx(min_depth);

      total_depths = max_depth - min_depth + 1;

      psi_h = new fas_grid_t[total_depths];
      rho_h = new fas_grid_t[total_depths];
      coarse_src_h = new fas_grid_t[total_depths];
      tmp_h = new fas_grid_t[total_depths];

      for(IDX_T depth = min_depth; depth <= max_depth; ++depth)
      {
        IDX_T depth_idx = _dIdx(depth);
        IDX_T n = _2toPwr(depth);
        IDX_T points = _Pwr3(n);

        psi_h[depth_idx] = new REAL_T[points];
        rho_h[depth_idx] = new REAL_T[points];
        coarse_src_h[depth_idx] = new REAL_T[points];
        tmp_h[depth_idx] = new REAL_T[points];

        _zeroGrid(psi_h[depth_idx], points);
        _zeroGrid(rho_h[depth_idx], points);
        _zeroGrid(coarse_src_h[depth_idx], points);
        _zeroGrid(tmp_h[depth_idx], points);
      }
    }; // constructor
 
    /**
     * @brief Initialize psi on finest grid
     * 
     * @param psi_guess guess for \psi on the finest grid
     */
    void initializeFinePsi(REAL_T *psi)
    {
      IDX_T points = _Pwr3(_2toPwr(max_depth));
      std::copy(psi, psi + points, psi_h[max_depth_idx]);
    }

    /**
     * @brief      Initialize matter source on all grids
     *
     * @param      rho   matter source on finest grid
     */
    void initializeRhoHeirarchy(REAL_T * rho)
    {
      // initialize values on fine grid
      IDX_T points = _Pwr3(_2toPwr(max_depth));
      std::copy(rho, rho + points, rho_h[max_depth_idx]);

      // restrict supplied rho to coarser grids
      for(IDX_T depth = max_depth; depth > min_depth; --depth)
      {
        _restrictFine2coarse(rho_h, depth);
      }
    }

    void VCycle()
    {
      IDX_T relax_iters = 5000;

      // initial residual
      _relaxSolution_GaussSeidel(max_depth, relax_iters);
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
      _copyGrid(psi_h, tmp_h, min_depth);

      // prolongate solution to finest grid
      // "upward" stroke
      for(coarse_depth = min_depth; coarse_depth < max_depth; coarse_depth++)
      {
        // relax "true" solution in psi_h
        _relaxSolution_GaussSeidel(coarse_depth, relax_iters);

        std::cout << "    Working on upward stroke at depth " << coarse_depth
                  << "; residual after solving is: "
                  << _getMaxResidual(coarse_depth) << ".\n" << std::flush;

        // tmp should hold appx. soln; convert to error
        _changeApproximateSolutionToError(tmp_h, psi_h, coarse_depth);
        // tmp should hold error
        _correctFineFromCoarseErr_Err2Appx(tmp_h, psi_h, coarse_depth+1);
        // tmp now holds appx. soln on finer grid;
        // phi_h now holds corrected solution on finer grid
      }

      // final relaxation
      _relaxSolution_GaussSeidel(max_depth, relax_iters);
      std::cout << "  Final max. residual on fine grid is: "
                << _getMaxResidual(max_depth) << ".\n" << std::flush;
      // std::cout << "  psi (fine) slice is: "; printStrip(psi_h, max_depth);
    }

    void VCycles(IDX_T num_cycles)
    {
      for(int cycle=0; cycle < num_cycles; ++cycle)
      {
        VCycle();
      }
      _relaxSolution_GaussSeidel(max_depth, 5000);
      std::cout << "  Final solution residual is: "
                << _getMaxResidual(max_depth) << ".\n" << std::flush;
    }


    /* TODO: split initialization into child class */
    void setTrialSolution()
    {
      IDX_T i, j, k;
      IDX_T n = _2toPwr(max_depth);

      fas_grid_t const psi = psi_h[max_depth_idx];
      fas_grid_t const rho = rho_h[max_depth_idx];

      // frequency and phase of waves
      REAL_T n1 = 1.0, n2 = 1.0, n3 = 1.0;
      REAL_T phi1 = 0, phi2 = 0, phi3 = 0;

      // generate trial solution psi
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);

        // generate trial solution
        psi[idx] = 1.0 - std::sin( 2.0 * 3.14159265 * n1 * (REAL_T)i/ (n) + phi1)
                         * std::sin( 2.0 * 3.14159265 * n2 * (REAL_T)j/ (n) + phi2)
                         * std::sin( 2.0 * 3.14159265 * n3 * (REAL_T)k/ (n) + phi3)/10.0;
      }

      // reconstruct rho from psi
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);
        // generate rho according to trial solution
        rho[idx] = -_laplacian(psi, i, j, k, n) / std::pow(psi[idx], 5.0);
      }

      // change psi to a "worse" guess
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);
        // add a random component ("worse" guess)
        psi[idx] = psi[idx] + ( ((REAL_T) std::rand())/RAND_MAX - 0.5)/10.0;
      }

      initializeFinePsi(psi);
      initializeRhoHeirarchy(rho);
    }

    void printStrip(fas_heirarchy_t out_h, IDX_T depth)
    {
      IDX_T i;
      IDX_T n = _2toPwr(depth);
      fas_grid_t const out = out_h[_dIdx(depth)];
      std::cout << std::fixed << "Values: { ";
      for(i=0; i<n; i++)
      {
        IDX_T idx = _gIdx(i,n/4,n/4,n);
        std::cout << out[idx];
        std::cout << ", ";
      }
      std::cout << "}\n";
    }

};

#endif