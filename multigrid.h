class FASMultigrid
{
  public:

    /**
     * @brief Constructor
     * @details Initialize internal variables, allocate memory
     * 
     * @param max_depth_in "depth" of finest grid (size is n = 2^max_depth)
     * @param min_depth_in "depth" of coarsest grid (size is n = 2^min_depth)
     */
    FASMultigrid(idx_t max_depth_in, idx_t min_depth_in)
    {
      max_depth = max_depth_in;
      min_depth = min_depth_in;
      max_depth_idx = D_INDEX(max_depth);
      min_depth_idx = D_INDEX(min_depth);

      total_depths = max_depth - min_depth + 1;

      *rho_h = new *real_t[total_depths];
      *err_h = new *real_t[total_depths];
      *phi_h = new *real_t[total_depths];
      *tau_h = new *real_t[total_depths];
      *tmp_h = new *real_t[total_depths];

      for(idx_t depth = min_depth; depth <= max_depth; ++depth)
      {
        depth_idx = D_INDEX(depth);
        std::cout << "Allocating depth: " << depth << " with index: " << depth_idx << "\n";
        n = PWROF2(depth);

        rho_h[depth_idx] = new real_t[PW3(n)];
        err_h[depth_idx] = new real_t[PW3(n)];
        phi_h[depth_idx] = new real_t[PW3(n)];
        tau_h[depth_idx] = new real_t[PW3(n)];
        tmp_h[depth_idx] = new real_t[PW3(n)];
      }

    }; // constructor
 
    // initialize 
    void initialize(real_t *rho)
    {
      rho_h[max_depth_idx] = rho;
      zero_array(err_h, );
    }

  private:
    // define a heirarchy of references to grids
    real_t **irhs, **itau, **iu,
           **irho, **itemp;

    real_t tr_err, res;

    idx_t max_depth, max_depth_idx;
    idx_t min_depth, min_depth_idx;
    idx_t total_depths;
};