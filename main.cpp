#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <omp.h>

#define PI (4.0*atan(1.0))
// box size in hubble units
#define H_LEN_FRAC 0.5
#define dx (H_LEN_FRAC/(1.0*N))

/*-----------NEW DEFINITION----------------------------------------------*/

// TODO: rename depth to something accurate
// grids have sizes from 2^MIN_DEPTH to 2^MAX_DEPTH
#define MIN_DEPTH 3 // coarsest grid depth, cannot be larger than the depth of finest grids
#define MAX_DEPTH 5 // max possible interation depth
#define TOTAL_DEPTHS (1 + MAX_DEPTH - MIN_DEPTH)
// indexing scheme for selecting a grid from heirarchy of grids
#define D_INDEX(depth) ((depth) - MIN_DEPTH)

// total grid points on the finest grid:
#define N      PWROF2(MAX_DEPTH)
#define POINTS PW3(N)

#define NPRE_RELAX_STEPS 100  //number of pre-cycle relaxation steps
#define NPOST_RELAX_STEPS 1000 //number of post-cycle relaxation steps
#define NCOARSE_RELAX_STEPS 1000 //numer of relaxation on coarse grids

#define LOOP3_N(i,j,k,n) \
  for(i=0; i<n; ++i) \
    for(j=0; j<n; ++j) \
      for(k=0; k<n; ++k)

#define LOOP_N3(i, n) \
  for(i=0; i<PW3(n); ++i) \

#define ALPHA 0.33 //coefficient of estimated truncation error

#define COARSE_EPS 1e-9
#define EPS 1e-6
#define IT_EPS 1e-20
#define CON_EPS 1e-9 //EPS used when finding shift to speed up interation
#define INF 1e100
#define NEWTON_INI 1.0

#define G_INDEX(i, j, k, n) (((i+n)%(n))*(n)*(n) + ((j+n)%(n))*(n) + (k+n)%(n))
#define LAP(mat, i, j, k, n, h) \
  (( mat[G_INDEX(i+1, j, k, n)] + mat[G_INDEX(i-1, j, k, n)] + \
     mat[G_INDEX(i, j+1, k, n)] + mat[G_INDEX(i, j-1, k, n)] + \
     mat[G_INDEX(i, j, k+1, n)] + mat[G_INDEX(i, j, k-1, n)] \
     - 6.0*mat[G_INDEX(i, j, k, n)])/(h*h))

#define PW5(a) ((a)*(a)*(a)*(a)*(a))
#define PW4(a) ((a)*(a)*(a)*(a))
#define PW3(a) ((a)*(a)*(a))
#define PW2(a) ((a)*(a))
#define PWROF2(n) (1<<(n))

#define MAX(a, b) ((a>b)? a: b )

typedef double real_t;
typedef long int idx_t;

#include "array_math.h"
#include "io.h"

real_t solve_volume_constraint(real_t *psi, real_t *rho, real_t *source, idx_t n)
{
  // Find value of A that gives a root of the function:
  //   f(A) = \integral (\rho * (\psi + A)^5 - f  ) d V 

  idx_t i, j, k;
  real_t eps = -INF, residual = NEWTON_INI;
  real_t num, den;

  do
  {
    num = integrate_conformal_rho(psi, rho, residual, n)
      - integrate_array(source, n);
    den = integrate_conformal_rho_derivative(psi, rho, residual, n);

    residual -= num/den;
    if( fabs(fabs(num/den) - eps) < CON_EPS )
      break;
    eps = fabs(num/den);
  } while(1);

  return residual;
}

void compute_constraint_residual(real_t *psi, real_t *rho, real_t *residual, idx_t n)
{
  // compute residual of Hamiltonian constraint equation

  idx_t i, j, k;
  real_t h = (H_LEN_FRAC/(1.0*n));

  LOOP3_N(i,j,k,n)
  {
    residual[G_INDEX(i, j, k, n)] = LAP(psi, i, j, k, n, h)
      + rho[G_INDEX(i, j, k, n)] * PW5(psi[G_INDEX(i, j, k, n)]);    
  }

  return;
}

real_t max_constraint_residual(real_t *phi, real_t *rho,
  real_t *truncation_source, idx_t n)
{
  // Calculate max. differnce between hamiltonian constraint and truncation_source

  int i, j, k;
  real_t h = (H_LEN_FRAC/(1.0*n)), max_difference = -INF;
  real_t constraint;

  LOOP3_N(i,j,k,n)
  {
    constraint = LAP(phi, i, j, k, n, h) + 
      rho[G_INDEX(i, j, k, n)] * PW5(phi[G_INDEX(i, j, k, n)]);

    max_difference = MAX( max_difference,
      fabs(constraint - truncation_source[G_INDEX(i, j, k, n)]) );
  }

  return max_difference;
}

real_t relax(real_t *phi, real_t *source, real_t *rho, idx_t n)
{
  // relax psi to a solution of the Hamiltonian constraint

  idx_t i, j, k, ipass, isw = 0, jsw, ksw;
  real_t temp, h = (H_LEN_FRAC/(1.0*n)), residual = INF;
  
  // "Red-black" Newton Gauss-Seidel relaxation
  // See, eg, Numerical Recipes P. 886.
  /*
  for(ipass = 0; ipass < 2; ipass++, isw = 1 - isw)
  {
    jsw = isw; 
    for(i = 0; i < n; i++, jsw = 1 - jsw)
    {
      ksw = jsw;
      for(j = 0; j < n; j++, ksw = 1 - ksw)
      {
        for(k = ksw; k < n; k+=2)
        {
          phi[G_INDEX(i, j, k, n)] -= (
              LAP(phi, i, j, k, n, h) - source[G_INDEX(i, j, k, n)]
              + rho[G_INDEX(i, j, k, n)] * PW5(phi[G_INDEX(i, j, k, n)])
            )/(
              -6/(h*h) + 
              5 * rho[G_INDEX(i, j, k, n)] * PW4(phi[G_INDEX(i, j, k, n)])
            );
        }
      }
    }
  }*/
  
  // Regular Gauss-Seidel relaxation as alternative
  // Loop has race condition; parallelization is unsafe
  #pragma omp parallel for default(shared) private(i,j,k)
  LOOP3_N(i,j,k,n)
  {
    phi[G_INDEX(i, j, k, n)] -= (
        LAP(phi, i, j, k, n, h) + rho[G_INDEX(i, j, k, n)] * PW5(phi[G_INDEX(i, j, k, n)])
        - source[G_INDEX(i, j, k, n)]
      )/(
        -6/(h*h) + 5 * rho[G_INDEX(i, j, k, n)] * PW4(phi[G_INDEX(i, j, k, n)])
      );

    // residual = MAX(res, fabs(temp));
  }

  return residual;
}

void coarse_grid_solve_constrained(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
  // solve the first order appproximation equation:
  //   \nabla \xi + 2 \pi \rho * 5 \xi = irhs - 2\pi \rho
  // and take u = 1 + xi as our initial value.
  // with speed up scheme

  idx_t i,j,k;
  real_t  h = H_LEN_FRAC/(1<<MIN_DEPTH), eps = INF, temp;
  idx_t nn = NCOARSE_RELAX_STEPS;

  initialize_random_array(u, (1<<MIN_DEPTH));

  while(nn--)
  {
    relax(u, irhs, irho, 1<<MIN_DEPTH);

    temp = solve_volume_constraint(u, irho, irhs, 1<<MIN_DEPTH);
    shift_array_values(u, temp, 1<<MIN_DEPTH);
    
    eps = max_constraint_residual(u, irho, irhs, 1<<MIN_DEPTH);
  }
}

void coarse_grid_solve(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
  // solve the first order appproximation equation:
  //   \nabla \ksi + 2 \pi \rho * 5 \ksi = irhs - 2\pi \rho
  // and take u = 1 + ksi as our initial value.
  // with out coarse scheme

  idx_t i,j,k;
  real_t  h = H_LEN_FRAC/(1<<MIN_DEPTH), eps = INF, temp;
  idx_t nn = NCOARSE_RELAX_STEPS;

  initialize_random_array(u, (1<<MIN_DEPTH));

  while(nn--)
  {
    relax(u, irhs, irho, 1<<MIN_DEPTH);
    eps = max_constraint_residual(u, irho, irhs, 1<<MIN_DEPTH);
    std::cout << "Coarse Grid Solution Max. Residual: " << eps << "\r" << std::flush;
  }
  std::cout << "\n";
}

void restrict_fine2coarse(real_t *u_coarse, real_t *u_fine, idx_t n_coarse) 
{
  // restrict scheme: 1*(1/8) + 6 * (1/16) + 12 * (1/32) + 8 * (1/64)

  idx_t i, j, k, fn = n_coarse*2, fi, fj, fk;
  #pragma omp parallel for default(shared) private(i,j,k)
  LOOP3_N(i,j,k,n_coarse)
  {
    fi = i*2;
    fj = j*2;
    fk = k*2;

    u_coarse[G_INDEX(i,j,k,n_coarse)] = 0.125 * u_fine[G_INDEX(fi,fj,fk,fn)]     
      + 0.0625 * (
        u_fine[G_INDEX(fi+1,fj,fk,fn)] +
        u_fine[G_INDEX(fi,fj+1,fk,fn)] +
        u_fine[G_INDEX(fi,fj,fk+1,fn)] +
        u_fine[G_INDEX(fi-1,fj,fk,fn)] +
        u_fine[G_INDEX(fi,fj-1,fk,fn)] +
        u_fine[G_INDEX(fi,fj,fk-1,fn)]
      )
      + 0.03125 * (
        u_fine[G_INDEX(fi+1,fj+1,fk,fn)] +
        u_fine[G_INDEX(fi+1,fj-1,fk,fn)] +
        u_fine[G_INDEX(fi-1,fj+1,fk,fn)] +
        u_fine[G_INDEX(fi-1,fj-1,fk,fn)] +
        u_fine[G_INDEX(fi+1,fj,fk+1,fn)] +
        u_fine[G_INDEX(fi+1,fj,fk-1,fn)] +
        u_fine[G_INDEX(fi-1,fj,fk+1,fn)] +
        u_fine[G_INDEX(fi-1,fj,fk-1,fn)] +
        u_fine[G_INDEX(fi,fj+1,fk+1,fn)] +
        u_fine[G_INDEX(fi,fj+1,fk-1,fn)] +
        u_fine[G_INDEX(fi,fj-1,fk+1,fn)] +
        u_fine[G_INDEX(fi,fj-1,fk-1,fn)] 
      )
      + 0.015625 * (
        u_fine[G_INDEX(fi+1,fj+1,fk+1,fn)] +
        u_fine[G_INDEX(fi+1,fj+1,fk-1,fn)] +
        u_fine[G_INDEX(fi+1,fj-1,fk+1,fn)] +
        u_fine[G_INDEX(fi-1,fj+1,fk+1,fn)] +
        u_fine[G_INDEX(fi+1,fj-1,fk-1,fn)] +
        u_fine[G_INDEX(fi-1,fj+1,fk-1,fn)] +
        u_fine[G_INDEX(fi-1,fj-1,fk+1,fn)] +
        u_fine[G_INDEX(fi-1,fj-1,fk-1,fn)]
      );
  }
}

void interpolate_coarse2fine(real_t *u_fine, real_t *u_coarse, idx_t fine_n) 
{
  // TODO: document interpolation scheme

  idx_t i, j, k;
  int coarse_n = fine_n/2;
  int fi, fj, fk, fn = fine_n;

  zero_array(u_fine, PW3(fine_n));

  #pragma omp parallel for default(shared) private(i,j,k)
  LOOP3_N(i,j,k,coarse_n)
  {
    fi = i*2;
    fj = j*2;
    fk = k*2;

    real_t cc = u_coarse[G_INDEX(i,j,k,coarse_n)];
    u_fine[G_INDEX(fi,fj,fk,fine_n)] += cc;
    
    u_fine[G_INDEX(fi+1,fj,fk,fn)] += cc/2.0;
    u_fine[G_INDEX(fi,fj+1,fk,fn)] += cc/2.0; 
    u_fine[G_INDEX(fi,fj,fk+1,fn)] += cc/2.0;
    u_fine[G_INDEX(fi-1,fj,fk,fn)] += cc/2.0; 
    u_fine[G_INDEX(fi,fj-1,fk,fn)] += cc/2.0; 
    u_fine[G_INDEX(fi,fj,fk-1,fn)] += cc/2.0;

    u_fine[G_INDEX(fi+1,fj+1,fk,fn)] += cc/4.0;
    u_fine[G_INDEX(fi+1,fj-1,fk,fn)] += cc/4.0;
    u_fine[G_INDEX(fi-1,fj+1,fk,fn)] += cc/4.0;
    u_fine[G_INDEX(fi-1,fj-1,fk,fn)] += cc/4.0;
    u_fine[G_INDEX(fi+1,fj,fk+1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi+1,fj,fk-1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi-1,fj,fk+1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi-1,fj,fk-1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi,fj+1,fk+1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi,fj+1,fk-1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi,fj-1,fk+1,fn)] += cc/4.0;
    u_fine[G_INDEX(fi,fj-1,fk-1,fn)] += cc/4.0;
            
    u_fine[G_INDEX(fi+1,fj+1,fk+1,fn)] += cc/8.0; 
    u_fine[G_INDEX(fi+1,fj+1,fk-1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi+1,fj-1,fk+1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi-1,fj+1,fk+1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi+1,fj-1,fk-1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi-1,fj+1,fk-1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi-1,fj-1,fk+1,fn)] += cc/8.0;
    u_fine[G_INDEX(fi-1,fj-1,fk-1,fn)] += cc/8.0;
  }
}

void fas_multigrid(real_t *psi, real_t *source, real_t *rho, idx_t n_cycles, real_t eps)
{
  // Solve non-linear constraint equation \nabla^2 \psi = -2pi \rho \psi 
  // using Full multigrid in FAS scheme
  // `source` is the right hand side of Numerical Recipes 2nd ed., eq. 19.6.23.

  std::cout << "MAX_DEPTH: " << MAX_DEPTH << " | ";
  std::cout << "MIN_DEPTH: " << MIN_DEPTH << " | ";
  std::cout << "TOTAL_DEPTHS: " << TOTAL_DEPTHS << "\n";

  // iterators
  idx_t depth, cycle, cycle_depth, nf;
  idx_t pre_relax_step, post_relax_step;

  idx_t n, depth_idx;
  idx_t max_depth_idx, min_depth_idx;
  max_depth_idx = D_INDEX(MAX_DEPTH);
  min_depth_idx = D_INDEX(MIN_DEPTH);

  // define a heirarchy of references to grids
  real_t *irhs[TOTAL_DEPTHS], *itau[TOTAL_DEPTHS], *iu[TOTAL_DEPTHS],
         *irho[TOTAL_DEPTHS], *itemp[TOTAL_DEPTHS], tr_err, res; 
  
  // initialize arrays at different grid levels / "depths";
  for(depth = MIN_DEPTH; depth <= MAX_DEPTH; ++depth)
  {
    depth_idx = D_INDEX(depth);
    std::cout << "Allocating depth: " << depth << " with index: " << depth_idx << "\n";
    n = PWROF2(depth);

    irho[depth_idx] = new real_t[PW3(n)];
    irhs[depth_idx] = new real_t[PW3(n)];
    iu[depth_idx] = new real_t[PW3(n)];
    itau[depth_idx] = new real_t[PW3(n)];
    itemp[depth_idx] = new real_t[PW3(n)];
  }

  // the "finest" grid is supplied to this function
  memcpy(irho[max_depth_idx], rho, sizeof(real_t)*PW3(PWROF2(MAX_DEPTH)));
  memcpy(irhs[max_depth_idx], source, sizeof(real_t)*PW3(PWROF2(MAX_DEPTH)));
  memcpy(iu[max_depth_idx], psi, sizeof(real_t)*PW3(PWROF2(MAX_DEPTH)));

  // restrict rho and "rhs" from finer grids to coarser grids:
  std::cout << "Restricting solution to coarser grids...\n" << std::flush;
  for(depth = MAX_DEPTH-1; depth >= MIN_DEPTH; --depth)
  { // depth = finest;  depth > coarsest; 
    depth_idx = D_INDEX(depth);
    n = PWROF2(depth);
    restrict_fine2coarse(irho[depth_idx], irho[depth_idx+1], n);
    restrict_fine2coarse(irhs[depth_idx], irhs[depth_idx+1], n);
  }

  // obtain a solution on the coarsest grid
  // Currently using multigrid without applying constraint of integral in speed up
  std::cout << "Obraining coarse grid solution...\n" << std::flush;
  coarse_grid_solve(iu[min_depth_idx], irhs[min_depth_idx],
    irho[min_depth_idx], PW3(PWROF2(MIN_DEPTH)));

  // Perform V-cycles at increasing depths
  for (depth = MIN_DEPTH + 1; depth <= MAX_DEPTH; ++depth)
  {
    n = PWROF2(depth);
    depth_idx = D_INDEX(depth);

    interpolate_coarse2fine(iu[depth_idx], iu[depth_idx-1], n);

    // Perform n_cycles V-cycles between MIN_DEPTH and depth
    for(cycle = 0; cycle < n_cycles; cycle++)
    {
      // perform fine->coarse steps of a V-cycle
      for(cycle_depth = depth; cycle_depth > MIN_DEPTH; --cycle_depth)
      {
        idx_t cycle_depth_idx = D_INDEX(cycle_depth);
        nf = PWROF2(cycle_depth);

        // perform pre-cycle relaxation
        for(pre_relax_step = 0; pre_relax_step < NPRE_RELAX_STEPS; pre_relax_step++)
        {
          real_t eps = relax(iu[cycle_depth_idx], irhs[cycle_depth_idx],
            irho[cycle_depth_idx], nf);
          if(eps < IT_EPS)
            break;

          // enforce volume constraint
          // real_t volume_constraint_shift = solve_volume_constraint(
          //   iu[cycle_depth_idx], irho[cycle_depth_idx], irhs[cycle_depth_idx], nf);
          // shift_array_values(iu[cycle_depth_idx], volume_constraint_shift, nf);
        }

        compute_constraint_residual(iu[cycle_depth_idx], irho[cycle_depth_idx], itemp[cycle_depth_idx], nf);

        nf /= 2;
        restrict_fine2coarse(itemp[cycle_depth_idx-1], itemp[cycle_depth_idx], nf);
        restrict_fine2coarse(iu[cycle_depth_idx-1], iu[cycle_depth_idx], nf);
        compute_constraint_residual(iu[cycle_depth_idx-1], irho[cycle_depth_idx-1], itau[cycle_depth_idx-1], nf);
        matrix_subtract(itau[cycle_depth_idx-1], itemp[cycle_depth_idx-1], itau[cycle_depth_idx-1], nf);
        
        if(cycle_depth_idx == depth)
          tr_err = ALPHA * norm(itau[cycle_depth_idx-1], nf);

        restrict_fine2coarse(irhs[cycle_depth_idx-1], irhs[cycle_depth_idx], nf);
        matrix_add(irhs[cycle_depth_idx-1], itau[cycle_depth_idx-1], irhs[cycle_depth_idx-1], nf);
      } // end V-cycle

      // determine solution on coarse grid given truncation error source
      coarse_grid_solve(iu[min_depth_idx], irhs[min_depth_idx],
        irho[min_depth_idx], PW3(PWROF2(MIN_DEPTH)));

      // perform coarse->fine steps in V-cycle
      nf = 1<<MIN_DEPTH;
      for(cycle_depth = MIN_DEPTH + 1; cycle_depth <= depth; cycle_depth++)
      {
        idx_t cycle_depth_idx = D_INDEX(cycle_depth);
        restrict_fine2coarse(itemp[cycle_depth_idx - 1], iu[cycle_depth_idx], nf);
        matrix_subtract(iu[cycle_depth_idx - 1], itemp[cycle_depth_idx-1], itemp[cycle_depth_idx-1], nf);
        nf *= 2;
        interpolate_coarse2fine(itau[cycle_depth_idx], itemp[cycle_depth_idx-1], nf);
        
        matrix_add(iu[cycle_depth_idx], itau[cycle_depth_idx], iu[cycle_depth_idx], nf);
        for(post_relax_step = 0; post_relax_step < NPOST_RELAX_STEPS; post_relax_step++)
        {
          if(relax(iu[cycle_depth_idx], irhs[cycle_depth_idx], irho[cycle_depth_idx], nf) < IT_EPS)
            break;
          //shift_array_values(iu[cycle_depth_idx], solve_volume_constraint(iu[cycle_depth_idx], irho[cycle_depth_idx], irhs[cycle_depth_idx], nf), nf);
        }
      }

      // Check residual on finest grid
      compute_constraint_residual(iu[depth_idx], irho[depth_idx], itemp[depth_idx], nf);
      matrix_subtract(itemp[depth_idx], irhs[depth_idx], itemp[depth_idx], nf);
      res = norm(itemp[depth_idx], nf);
      std::cout << "max_constraint_residual at depth " << depth << " is:\n";
      std::cout << std::fixed << max_constraint_residual(iu[depth_idx], irho[depth_idx], irhs[depth_idx], 1<<depth)<<"\n\n";
      
      if(res < tr_err) break;
    } // end loop for V-cycles
  }
  
  std::cout << "Residual on coarse grid\n" << std::fixed
            << max_constraint_residual(iu[D_INDEX(MIN_DEPTH)],
                irho[D_INDEX(MIN_DEPTH)], irhs[D_INDEX(MIN_DEPTH)], PWROF2(MIN_DEPTH))
            << "\n";

  memcpy(psi, iu[max_depth_idx], sizeof(real_t)*PW3(PWROF2(MAX_DEPTH)));

  for(depth = MIN_DEPTH; depth <= MAX_DEPTH; ++depth)
  {
    depth_idx = D_INDEX(depth);
    delete [] iu[depth_idx];
    delete [] irho[depth_idx];
    delete [] irhs[depth_idx];
    delete [] itemp[depth_idx];
    delete [] itau[depth_idx];
  }
}

int main(int argc, char **argv)
{  
  //Solving equations Lap (psi) + rho * psi^5 = source

  int i, j, k;
  real_t *psi_solution, *psi_trial, *rho, *res, *source;
  real_t h = H_LEN_FRAC/(real_t)(N);
  
  std::cout.precision(15);
  omp_set_num_threads(4);
  srand(129);

  psi_solution = new real_t[PW3(N)];
  source = new real_t[PW3(N)];
  rho = new real_t[PW3(N)];
  psi_trial = new real_t[PW3(N)];
  res = new real_t[PW3(N)];

  // frequency and phase of waves
  real_t n1 = 1, n2 = 1, n3 = 1;
  real_t phi1 = 0, phi2 = 0, phi3 = 0;

  LOOP3_N(i,j,k,N)
  {
    idx_t p = G_INDEX(i, j, k, N);

    source[p] = 0;
    // generating standard solution
    psi_trial[p] = 1.0 - sin( 2.0 * PI * n1 * (real_t)i/ (N) + phi1)
                 * sin( 2.0 * PI * n2 * (real_t)j/ (N) + phi2)
                 * sin( 2.0 * PI * n3 * (real_t)k/ (N) + phi3)/20.0;
  }

  LOOP3_N(i,j,k,N)
  {
    idx_t p = G_INDEX(i, j, k, N);
    // generating rho according to standard solution
    rho[p] = (-LAP(psi_trial, i, j, k, N, h) + source[p]) / PW5(psi_trial[p]);
  }

  fas_multigrid(psi_solution, source, rho, 5, 1e-7);

  freopen("rho.txt","w", stdout);
  print_mathematica_array(rho, (N));
  freopen("source.txt", "w", stdout);
  print_mathematica_array(source, N);
  freopen("psi_solution.txt", "w", stdout);
  print_mathematica_array(psi_solution, (N));
  freopen("psi_trial.txt", "w", stdout);
  print_mathematica_array(psi_trial, N);
  
  delete [] psi_solution;
  delete [] source;
  delete [] rho;
  delete [] psi_trial;
  delete [] res;
  
  return 0;
}
