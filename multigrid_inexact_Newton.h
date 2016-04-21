#ifndef FAS_MULTIGRID_H
#define FAS_MULTIGRID_H

//#include <omp.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdio>

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
                    u_h, // field seeking a solution for
                    appx_u_h, // field containing an approximate solution
                    rho_h, // matter source in elliptic pde
                    damping_tmp_h, // storing _lap (u) - f, using for calculate F(u + \lambda v)
                    lap_v_h, // storing _lap(v), using for caculate F(u + \lambda v)
                    damping_v_h, //storing v, using for update u through u_{n+1} = u{n} + v in interation
                    jac_rhs_h; //stroing - F(u) which is rhs of Jacob Linear function
                    
					
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

      fas_grid_t const u = u_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];

      IDX_T idx = _gIdx(i, j, k, n);
      return _laplacian(u, i, j, k, n) + rho[idx] * std::pow(u[idx], 5.0);
    }

    /**
     * @brief Find value of A that gives a root of the function:
     * f(A) = \integral (\rho * (\u + A)^5 - coarse_src  ) d V
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
      _shiftGridVals(u_h[depth_idx], shift, _Pwr3(n));
    }

    REAL_T _monopoleConstraintTotal(IDX_T depth, REAL_T shift)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const u = u_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += rho[idx] * std::pow(u[idx] + shift, 5.0) - coarse_src[idx];
      }

      return total;
    }

    REAL_T _monopoleConstraintDerivativeTotal(IDX_T depth, REAL_T shift)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const u = u_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += rho[idx] * std::pow(u[idx] + shift, 4.0);
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

      fas_grid_t const u = u_h[depth_idx];
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
     * @brief      Compute coarse_src and u on a coarser grid
     * using tmp_h for some computations
     *
     * @param[in]  fine_depth  depth of grid to coarsen
     */
    void _computeCoarseRestrictions(IDX_T fine_depth)
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

	
	REAL_T _getLambda(IDX_T depth, REAL_T norm)
    {
		//linearly inumerate corresponding damping parameter \lambda
        IDX_T i, j, k, s;
        IDX_T depth_idx = _dIdx(depth);
        IDX_T n = _2toPwr(depth);
        REAL_T lambda = 0.0, temp, sum = 0.0;
        fas_grid_t const damping_tmp = damping_tmp_h[depth_idx];
        fas_grid_t const lap_v = lap_v_h[depth_idx];
		fas_grid_t const damping_v = damping_v_h[depth_idx];
		fas_grid_t const rho = rho_h[depth_idx];
		fas_grid_t const u = u_h[depth_idx];
		
        for( s = 0; s < 100; s++)
        {
            lambda = 1.0 - (REAL_T)s * 0.01; //shoule always start with \lambda = 1
            sum = 0.0;
            FAS_LOOP3_N(i,j,k,n)
            {
                IDX_T idx = _gIdx(i, j, k, n);
                temp = damping_tmp[idx] + lambda * lap_v[idx] 
                       + rho[idx] * std::pow(u[idx] + lambda * damping_v[idx], 5.0);
                sum += temp * temp;
            }
            if(sum <= norm)  // when | F(u + \lambda v) | < | F(u) | stop
                return lambda;
        }
        return lambda;
    }
	
	REAL_T _dampingConstraintTotal(IDX_T depth, REAL_T shift)
    {
       //calculating total  of Jacob equation after shift
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);
	  fas_grid_t const rho = rho_h[depth_idx];
	  fas_grid_t const u = u_h[depth_idx];
	  fas_grid_t const damping_v = damping_v_h[depth_idx];
	  fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];

      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += 5.0 * rho[idx] * std::pow(u[idx] , 4.0)
                *(damping_v[idx] + shift)
                - jac_rhs[idx];
      }

      return total;
    }
	
	REAL_T _dampingConstraintDerivativeTotal(IDX_T depth, REAL_T shift)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);
	  fas_grid_t const rho = rho_h[depth_idx];
	  fas_grid_t const u = u_h[depth_idx];


      REAL_T total = 0.0;
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        total += 5.0 * rho[idx] * std::pow(u[idx] , 4.0);
      }

      return total;
    }
	
	void _shiftConstrainedDamping(IDX_T depth) 
    {
		
	  //shift the Jacob equation to satisy constraint, shometime does not helpful
      REAL_T eps = 0.0, shift = 0.0;
      REAL_T num, den, cnt=0;
	  
	  num = _dampingConstraintTotal(depth, 0.0);
	  den = _dampingConstraintDerivativeTotal(depth, 0.0);
      shift = -num / den;
      IDX_T n = _2toPwr(depth);
      IDX_T depth_idx = _dIdx(depth);
      _shiftGridVals(damping_v_h[depth_idx], shift, _Pwr3(n));
    }
	
	bool jac_relax(IDX_T depth, REAL_T norm, REAL_T C, IDX_T p)
    {
		//relax Jocob equation, currently without using constrain to speedup, so will work very slow
		// C and p set the convergent speed of interation
        IDX_T i, j, k, s;
        IDX_T depth_idx = _dIdx(depth);
        IDX_T n = _2toPwr(depth), cnt = 0;
        REAL_T   norm_r = 1e100, dx = grid_length/n, norm_pre, temp;
	    fas_grid_t const rho = rho_h[depth_idx];
	    fas_grid_t const u = u_h[depth_idx];
	    fas_grid_t const damping_v = damping_v_h[depth_idx];
	    fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];
        fas_grid_t const coarse_src = coarse_src_h[depth_idx];
		
		FAS_LOOP3_N(i, j, k, n)
		{
			damping_v[_gIdx(i,j,k,n)] = 0.0;
		}
		
        while( norm_r >= std::min(pow(norm, (REAL_T)(p+1)) * C, norm)) 
        {
			
			//relax until the convergent condition got satisfy 
            norm_r = 0.0;
            norm_pre = 0.0;
            FAS_LOOP3_N(i,j,k,n)
            {
                IDX_T idx = _gIdx(i, j, k, n);
                damping_v[idx] = (damping_v[_gIdx(i+1, j, k, n)] + damping_v[_gIdx(i-1, j, k, n)] +
											damping_v[_gIdx(i, j+1, k, n)] + damping_v[_gIdx(i, j-1, k, n)] +
											damping_v[_gIdx(i, j, k+1, n)] + damping_v[_gIdx(i, j, k-1, n)]
											- jac_rhs[idx] * dx *dx) /
											( 6.0 - 5.0*rho[idx]*std::pow(u[idx], 4.0) *dx *dx);
				/*damping_v[depth_idx][idx]  -= (_laplacian(damping_v[depth_idx], i, j, k, n) 
												+ 5.0*rho_h[depth_idx][idx]*std::pow(u_h[depth_idx][idx], 4.0) * damping_v[depth_idx][idx] 
												- jac_rhs[depth_idx][idx])/(-6.0/ dx/dx + 5.0*rho_h[depth_idx][idx]*std::pow(u_h[depth_idx][idx], 4.0));*/
            }
            
            FAS_LOOP3_N(i,j,k,n)
            {
				IDX_T idx = _gIdx(i, j, k, n);
                temp = _laplacian(damping_v, i, j, k, n) 
											+ 5.0*rho[idx]*std::pow(u[idx], 4.0) * damping_v[idx]
											- jac_rhs[idx];
				
                norm_r += temp * temp;
            }
			
            //_shiftConstrainedDamping(depth); 
			
			cnt++;
			if(cnt > 500 && norm_r > norm_pre) 
            {
                //cannot solve Jacobian equation to precision needed
                std::cout<<"cannot get enough precision!\n";
                return 0;
            }
        }
        return 1;
		
    }
	
	void _relaxSolution_GaussSeidel_damp(IDX_T depth, IDX_T iterations)
    {
      // relax u using the inexact Newton iteration
	  
      IDX_T i, j, k, s;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);
      REAL_T temp, lambda, norm;

      fas_grid_t const rho = rho_h[depth_idx];
	  fas_grid_t const u = u_h[depth_idx];
	  fas_grid_t const damping_v = damping_v_h[depth_idx];
	  fas_grid_t const jac_rhs = jac_rhs_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];
	  fas_grid_t const damping_tmp = damping_tmp_h[depth_idx];
	  fas_grid_t const lap_v = lap_v_h[depth_idx];
      
	  
      // run some # iterations
      
      for(s=0; s<iterations; ++s)
      {
        
        //#pragma omp parallel for default(shared) private(i,j,k)
        norm = 0.0;
        FAS_LOOP3_N(i,j,k,n)
        {
          IDX_T idx = _gIdx(i, j, k, n);

          temp  =  _laplacian(u, i, j, k, n)
            + rho[idx]*std::pow(u[idx], 5.0) - coarse_src[idx];
          norm += temp * temp; // updating norm of F(u)
          //updating _lap(u) - f
          damping_tmp[idx] = _laplacian(u, i, j, k, n) - coarse_src[idx];
          //evalue jac_source at right hand side of Jacobian linear equation
           jac_rhs[idx] = -_evaluateEllipticEquationPt(depth, i, j, k) + coarse_src[idx];  
        }
        

        if( jac_relax(depth, norm, 1, 0) ==0) //relax to solve Jacobian linear equation
            break;
        FAS_LOOP3_N(i,j,k,n)
        {
            //updating _lap(v) which help to calculate F(u + \lambda v)
            IDX_T idx = _gIdx(i,j,k,n);
            lap_v[idx] = _laplacian(damping_v, i, j, k, n);
        }
        //get lambda
        lambda = _getLambda(depth, norm);

        FAS_LOOP3_N(i,j,k,n)
        {
            IDX_T idx = _gIdx(i, j, k, n);

          // update u according to lambda and v
            u[idx] += damping_v[idx] * lambda;
        }
        
        temp = _getMaxResidual(depth);
        
        
        if(temp < 1e-6) //set presision
			break;
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

      u_h = new fas_grid_t[total_depths];
      rho_h = new fas_grid_t[total_depths];
      coarse_src_h = new fas_grid_t[total_depths];
      tmp_h = new fas_grid_t[total_depths];
	  
	    damping_tmp_h = new fas_grid_t[total_depths];
		lap_v_h = new fas_grid_t[total_depths];
		damping_v_h = new fas_grid_t[total_depths];

		
		jac_rhs_h = new fas_grid_t[total_depths];
	  
      for(IDX_T depth = min_depth; depth <= max_depth; ++depth)
      {
        IDX_T depth_idx = _dIdx(depth);
        IDX_T n = _2toPwr(depth);
        IDX_T points = _Pwr3(n);

        u_h[depth_idx] = new REAL_T[points];
        rho_h[depth_idx] = new REAL_T[points];
        coarse_src_h[depth_idx] = new REAL_T[points];
        tmp_h[depth_idx] = new REAL_T[points];
		
		damping_tmp_h[depth_idx] = new REAL_T[points];
		lap_v_h[depth_idx] = new REAL_T[points];
		damping_v_h[depth_idx] = new REAL_T[points];
		
		jac_rhs_h[depth_idx] = new REAL_T[points];
		
        _zeroGrid(u_h[depth_idx], points);
        _zeroGrid(rho_h[depth_idx], points);
        _zeroGrid(coarse_src_h[depth_idx], points);
        _zeroGrid(tmp_h[depth_idx], points);
		_zeroGrid(damping_tmp_h[depth_idx], points);
		_zeroGrid(lap_v_h[depth_idx], points);
		_zeroGrid(damping_v_h[depth_idx], points);
		
		_zeroGrid(jac_rhs_h[depth_idx], points);
      }
    }; // constructor
 
    /**
     * @brief Initialize u on finest grid
     * 
     * @param u_guess guess for \u on the finest grid
     */
    void initializeFineU(REAL_T *u)
    {
      IDX_T points = _Pwr3(_2toPwr(max_depth));
      std::copy(u, u + points, u_h[max_depth_idx]);
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
      IDX_T relax_iters = 5;

      // initial residual
      _relaxSolution_GaussSeidel_damp(max_depth, relax_iters);
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
        _relaxSolution_GaussSeidel_damp(coarse_depth, relax_iters);

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
      _relaxSolution_GaussSeidel_damp(max_depth, relax_iters);
      std::cout << "  Final max. residual on fine grid is: "
                << _getMaxResidual(max_depth) << ".\n" << std::flush;
      // std::cout << "  u (fine) slice is: "; printStrip(u_h, max_depth);
    }

    void VCycles(IDX_T num_cycles)
    {
      for(int cycle=0; cycle < num_cycles; ++cycle)
      {
        VCycle();
        //std::cout<"dfsfdsaf\n";
      }
      _relaxSolution_GaussSeidel_damp(max_depth, 10);
      std::cout << "  Final solution residual is: "
                << _getMaxResidual(max_depth) << ".\n" << std::flush;
	   freopen("res.txt","w", stdout);
		print_all(u_h, max_depth);
    }


    /* TODO: split initialization into child class */
    void setTrialSolution()
    {
      IDX_T i, j, k;
      IDX_T n = _2toPwr(max_depth);

      fas_grid_t const u = u_h[max_depth_idx];
      fas_grid_t const rho = rho_h[max_depth_idx];

      // frequency and phase of waves
      REAL_T n1 = 1.0, n2 = 1.0, n3 = 1.0;
      REAL_T phi1 = 0, phi2 = 0, phi3 = 0;

      // generate trial solution u
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);

        // generate trial solution
        u[idx] = 1.0 - std::sin( 2.0 * 3.14159265 * n1 * (REAL_T)i/ (n) + phi1)
                         * std::sin( 2.0 * 3.14159265 * n2 * (REAL_T)j/ (n) + phi2)
                         * std::sin( 2.0 * 3.14159265 * n3 * (REAL_T)k/ (n) + phi3)/10.0;
      }

      // reconstruct rho from u
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);
        // generate rho according to trial solution
        rho[idx] = -_laplacian(u, i, j, k, n) / std::pow(u[idx], 5.0);
      }

      // change u to a "worse" guess
      FAS_LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i,j,k,n);
        // add a random component ("worse" guess)
        u[idx] = u[idx] + ( ((REAL_T) std::rand())/RAND_MAX - 0.5)/100.0;
      }

      initializeFineU(u);
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
	void print_all(fas_heirarchy_t out_h, IDX_T depth)
	{
		fas_grid_t const m = out_h[_dIdx(depth)];
		IDX_T n = _2toPwr(depth);
		std::cout<<"{";
		
		for(int i = 0; i < n; i++)
		{
			std::cout<<"{";
			for(int j = 0; j < n; j++)
			{
				std::cout<<"{";
				std::cout<<std::fixed<<m[_gIdx(i,j,0,n)];
				for(int k = 1; k < n; k++)
				{
					
					std::cout<<std::fixed<<","<<m[_gIdx(i,j,k,n)];
				}
				std::cout<<"}";
				if(j != n-1)
					std::cout<<",";
			}
			std::cout<<"}";
			if(i != n-1)
				std::cout<<',';
			//std::cout<<"\n";
        
		}
		std::cout<<"}";
	}

};

#endif
