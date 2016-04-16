
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
                    psi_h, // field weeking a solution for
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
    void _zero_grid(fas_grid_t grid, IDX_T points)
    {
      for(IDX_T i=0; i < points; i++)
        grid[i] = 0;
    }

    /**
     * @brief raise 2 to a power
     * 
     * @param pwr power to raise 2 to
     * @return 2^pwr
     */
    IDX_T _2toPwr(IDX_T pwr)
    {
      return 1<<pwr;
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

      IDX_T n_fine = _2toN(fine_depth);
      IDX_T n_coarse = _2toN(fine_depth-1);

      fas_array_t const fine_grid = grid_heirarchy[fine_depth];
      fas_array_t const coarse_grid = grid_heirarchy[fine_depth-1];

      IDX_T i, j, k; // coarse grid iterator
      IDX_T fi, fj, fk; // fine grid indexes

      #pragma omp parallel for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n_coarse)
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
      IDX_T n_coarse = _2toN(coarse_depth);
      IDX_T n_fine = _2toN(coarse_depth+1);

      fas_grid_t const coarse_grid = grid_heirarchy[coarse_depth];
      fas_grid_t const fine_grid = grid_heirarchy[coarse_depth+1];

      IDX_T i, j, k;
      IDX_T fi, fj, fk;

      zero_array(fine_grid, PW3(n_fine));

      #pragma omp parallel for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n_coarse)
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
    void _computeEllipticOperator(fas_heirarchy_t result_h, IDX_T depth)
    {
      IDX_T i, j, k;
      IDX_T depth_idx = _dIdx(depth);
      IDX_T n = _2toPwr(depth);

      fas_grid_t const phi = phi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const result = result_h[depth_idx];

      #pragma omp for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        result[idx] = _laplacian(phi, i, j, k, n) + rho[idx] * PW5(phi[idx]);
      }
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

      fas_grid_t const phi = phi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];
      fas_grid_t const residual = residual_h[depth_idx];

      // intermediately store elliptic operator result on residual grid
      _computeEllipticOperator(residual_h, depth);
      // residual is coarse_src - elliptic operator
      #pragma omp for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        residual[idx] = coarse_source[idx]
          - residual[idx];
      }
    }

    /**
     * @brief      Compute coarse_src and psi on a coarser grid
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
      _computeEllipticOperator(coarse_src_h, fine_depth-1);

      // add in restricted residual to coarse source;
      // coarse source is then set.
      IDX_T n_coarse = _2toN(fine_depth-1);
      #pragma omp parallel for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n_coarse)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        coarse_src[idx] += tmp;
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
      LOOP3_N(i,j,k,n)
      {
        IDX_T idx = _gIdx(i, j, k, n);
        appx_to_err[idx] = exact_soln[idx] - appx_to_err[idx];
      }
    }

    void _correctFineFromCoarseErr(fas_heirarchy_t err_h,
      fas_heirarchy_t appx_soln_h, IDX_T fine_depth)
    {
      IDX_T i, j, k;
      IDX_T coarse_depth = fine_depth-1;

      IDX_T fine_depth_idx = _dIdx(fine_depth);
      IDX_T n_fine = _2toPwr(fine_depth);

      _interpolateCoarse2fine(err_h, coarse_depth);

      fas_grid_t const appx_corr = err_h[fine_depth_idx];
      fas_grid_t const appx_soln = appx_soln_h[fine_depth_idx];

      #pragma omp parallel for default(shared) private(i,j,k)
      LOOP3_N(i,j,k,n_fine)
      {
        IDX_T idx = _gIdx(i, j, k, n_fine);
        appx_soln[idx] += appx_corr[idx];
      }
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
      REAL_T dx = (H_LEN_FRAC/n);

      fas_grid_t const phi = phi_h[depth_idx];
      fas_grid_t const rho = rho_h[depth_idx];
      fas_grid_t const coarse_src = coarse_src_h[depth_idx];

      // run some # iterations
      for(s=0; s<iterations; ++s)
      {
        // Note that this has a race condition
        // TODO: fix.
        #pragma omp parallel for default(shared) private(i,j,k)
        LOOP3_N(i,j,k,n)
        {
          IDX_T idx = _gIdx(i, j, k, n);
          // Gauss-Seidel step for equation of interest
          phi[idx] -= (
              _laplacian(phi, i, j, k, n) + rho[idx]*PW5(phi[idx])
              - coarse_source[idx]
            )/(
              -6/(dx*dx) + 5 * rho[idx] * PW4(phi[idx])
            );
        }
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

      psi_h = new fas_array_t[total_depths];
      rho_h = new fas_array_t[total_depths];
      coarse_src_h = new fas_array_t[total_depths];

      for(IDX_T depth = min_depth; depth <= max_depth; ++depth)
      {
        depth_idx = _dIdx(depth);
        std::cout << "Allocating depth: " << depth << " with index: " << depth_idx << "\n";
        n = _2toPwr(depth);

        psi_h[depth_idx] = new REAL_T[PW3(n)];
        rho_h[depth_idx] = new REAL_T[PW3(n)];
        coarse_src_h[depth_idx] = new REAL_T[PW3(n)];
        tmp_h[depth_idx] = new REAL_T[PW3(n)];

        _zero_grid(psi_h[depth_idx]);
        _zero_grid(rho_h[depth_idx]);
        _zero_grid(coarse_src_h[depth_idx]);
        _zero_grid(tmp_h[depth_idx]);
      }

    }; // constructor
 
    /**
     * @brief Initialize phi on finest grid
     * 
     * @param psi_guess guess for \psi on the finest grid
     */
    void initializeFinePsi(REAL_T *psi_guess)
    {
      // initialize values on fine grid
      std::copy(std::begin(psi), std::end(psi),
        std::begin(psi_h[max_depth_idx]));
    }

    /**
     * @brief      Initialize matter source on all grids
     *
     * @param      rho   matter source on finest grid
     */
    void initializeRhoHeirarchy(REAL_T * rho)
    {
      // initialize values on fine grid
      std::copy(std::begin(rho), std::end(rho),
        std::begin(rho_h[max_depth_idx]));

      // restrict supplied rho to coarser grids
      for(IDX_T depth = max_depth; depth > min_depth; --depth)
      {
        _restrictFine2coarse(rho_h, depth);
      }
    }

};
