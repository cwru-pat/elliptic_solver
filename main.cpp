#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>

#define PI (4.0*atan(1.0))
// box size in hubble units
#define H_LEN_FRAC 0.5
#define dx (H_LEN_FRAC/(1.0*N))

/*-----------NEW DEFINITION----------------------------------------------*/

#define N 16  // finest used grid # points
#define MIN_DEPTH 4 //coarsest grids depth, cannot be larger than the depth of finest grids
#define MAX_DEPTH 10 // max possible interation depth

#define NPRE 100  //number of pre-cycle relaxation steps
#define NPOST 100 //number of post-cycle relaxation steps
#define NCOARSE 1000 //numer of relaxation on coarse grids


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

void shift_array_values(real_t *array, real_t shift, idx_t n)
{
  // shift array values; eg. to satisfy constraint
  idx_t i;
  LOOP_N3(i, n)
    array[i] += shift;
}

void constraint_residual(real_t *psi, real_t *rho, real_t *residual, idx_t n)
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
  real_t temp, h = (H_LEN_FRAC/(1.0*n)), res =  INF;
  
  // "Red-black" Newton Gauss-Seidel relaxation
  // See, eg, Numerical Recipes P. 886.
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
  }
  
  // Regular Gauss-Seidel relaxation as alternative
  /*
  LOOP3_N(i,j,k,n)
  {
    phi[G_INDEX(i, j, k, n)] -= (
        LAP(phi, i, j, k, n, h) + rho[G_INDEX(i, j, k, n)] * PW5(phi[G_INDEX(i, j, k, n)])
        - source[G_INDEX(i, j, k, n)]
      )/(
        -6/(h*h) + 5 * rho[G_INDEX(i, j, k, n)] * PW4(phi[G_INDEX(i, j, k, n)])
      );

    res = MAX(res, fabs(temp));
  }
  */

  return res;
}

void coarse_grid_solve_constrained(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
  // solve the first order appproximation equation:
  //   \nabla \xi + 2 \pi \rho * 5 \xi = irhs - 2\pi \rho
  // and take u = 1 + xi as our initial value.
  // with speed up scheme

  idx_t i,j,k;
  real_t  h = H_LEN_FRAC/(1<<MIN_DEPTH), eps = INF, temp;
  idx_t nn = NCOARSE;

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
  initialize_random_array(u, (1<<MIN_DEPTH));
  idx_t nn = NCOARSE;

  while(nn--)
  {
    relax(u, irhs, irho, 1<<MIN_DEPTH);
    eps = max_constraint_residual(u, irho, irhs, 1<<MIN_DEPTH);
    std::cout << "Residual: " << eps << "\n";
  }
}

void restrict_fine2coarse(real_t *u_coarse, real_t *u_fine, idx_t n_coarse) 
{
  // restrict scheme: 1*(1/8) + 6 * (1/16) + 12 * (1/32) + 8 * (1/64)

  idx_t i, j, k, fn = n_coarse*2, fi, fj, fk;
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
  // interpolation scheme: 1*(1/8) + 6 * (1/16) + 12 * (1/32) + 8 * (1/64)

  idx_t i, j, k;
  int coarse_n = fine_n/2;
  int fi, fj, fk, fn = fine_n;

  zero_array(u_fine, PW3(fine_n));

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

void fas_multigrid(real_t *psi, real_t *source, real_t *rho, idx_t ncycle, real_t eps)
{
  // Solve non-linear constraint equation \nabla^2 \psi = -2pi \rho \psi 
  // using Full multigrid in FAS scheme
  // `source` is the right hand side of Numerical Recipes 2nd ed., eq. 19.6.23.

  // "depth" of finest grid;
  // finest grid is (2^DEPTH)^3
  idx_t DEPTH = 0;

  // iterators
  idx_t i, j, k, jc, nf, jj, jpre, jpost;
  idx_t n = (1<<MIN_DEPTH);

  // define arrays that will be used in the iteration
  real_t *irhs[MAX_DEPTH], *itau[MAX_DEPTH], *iu[MAX_DEPTH],
         *irho[MAX_DEPTH], *itemp[MAX_DEPTH], tr_err, res; 
  
  // log_2(N) = DEPTH
  while ((1<<DEPTH) != N) 
    DEPTH++;

  // 1<<DEPTH = 2^DEPTH
  irho[DEPTH] = new real_t[PW3(1<<DEPTH)];
  irhs[DEPTH] = new real_t[PW3(1<<DEPTH)];

  memcpy(irho[DEPTH], rho, sizeof(real_t)*PW3(1<<DEPTH));
  memcpy(irhs[DEPTH], source, sizeof(real_t)*PW3(1<<DEPTH));
  i = DEPTH;

  while(i > MIN_DEPTH)
  {
    irho[--i] = new real_t[PW3(1<<i)];
    irhs[i] = new real_t[PW3(1<<i)];
    restrict_fine2coarse(irho[i], irho[i+1], 1<<i);
    restrict_fine2coarse(irhs[i], irhs[i+1], 1<<i);
  }
  
  iu[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
  itau[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
  itemp[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
  initialize_random_array(iu[MIN_DEPTH], (1<<MIN_DEPTH));
  //Currently using multigrid without applying constraint of integral in speed up
  coarse_grid_solve(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));

  for (j = MIN_DEPTH + 1; j <= DEPTH; j++)
  {
    n *= 2;
    iu[j] = new real_t[PW3(n)];
    itau[j] = new real_t[PW3(n)];
    itemp[j] = new real_t[PW3(n)]; 
    
    interpolate_coarse2fine(iu[j], iu[j-1], n);
    
    // loop for V-cycles
    for(jc = 0; jc < ncycle; jc++)
    {
      nf = n;
      
      for(jj = j; jj > MIN_DEPTH; jj--)
      {
        for(jpre =0; jpre < NPRE; jpre++)
        {
          if(relax(iu[jj], irhs[jj], irho[jj], nf) < IT_EPS)
            break;
          //shift_array_values(iu[jj], solve_volume_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
        }
        constraint_residual(iu[jj], irho[jj], itemp[jj], nf);
        nf = nf>>1;
        restrict_fine2coarse(itemp[jj-1], itemp[jj], nf);
        //print_vector(itemp[jj-1], PW3(nf));;
        restrict_fine2coarse(iu[jj-1], iu[jj], nf);
        constraint_residual(iu[jj-1], irho[jj-1], itau[jj-1], nf);
        matrix_subtract(itau[jj-1], itemp[jj-1], itau[jj-1], nf);
        if(jj == j)
          tr_err = ALPHA * norm(itau[jj-1], nf);
        restrict_fine2coarse(irhs[jj-1], irhs[jj], nf);
        matrix_add(irhs[jj-1], itau[jj-1], irhs[jj-1], nf);
      }
      
      coarse_grid_solve(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));
      
      nf = 1<<MIN_DEPTH;
      for(jj = MIN_DEPTH + 1; jj <= j; jj++)
      {
        
        restrict_fine2coarse(itemp[jj - 1], iu[jj], nf);
        matrix_subtract(iu[jj - 1], itemp[jj-1], itemp[jj-1], nf);
        nf *=2;
        interpolate_coarse2fine(itau[jj], itemp[jj-1], nf);
        
        matrix_add(iu[jj], itau[jj], iu[jj], nf);
        for(jpost = 0; jpost < NPOST; jpost++)
        {
          if(relax(iu[jj], irhs[jj], irho[jj], nf) < IT_EPS)
            break;
          //shift_array_values(iu[jj], solve_volume_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
        }
      }
      constraint_residual(iu[j], irho[j], itemp[j], nf);
      matrix_subtract(itemp[j], irhs[j], itemp[j], nf);
      res = norm(itemp[j], nf);
      std::cout << "max_constraint_residual at depth " << j << ":\n";
      std::cout << std::fixed << max_constraint_residual(iu[j], irho[j], irhs[j], 1<<j)<<"\n\n";
      
      if(res < tr_err) break;
    } // end loop for V-cycles
  }
  
  std::cout << "Residual on coarse grid\n" << std::fixed
            << max_constraint_residual(iu[DEPTH],rho,source,N)<<"\n";

  memcpy(psi, iu[DEPTH], sizeof(real_t) * PW3((1<<DEPTH)));

  for(idx_t i = MIN_DEPTH; i <= DEPTH; i++)
  {
    delete [] iu[i];
    delete [] irho[i];
    delete [] irhs[i];
    delete [] itemp[i];
    delete [] itau[i];
  }
}

int main(int argc, char **argv)
{  
  //Solving equations Lap (psi) + rho * psi^5 = source

  int i, j, k;
  real_t *psi_solution, *psi_trial, *rho, *res, *source;
  real_t h = H_LEN_FRAC/(real_t)(N);
  
  std::cout.precision(15);

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

  fas_multigrid(psi_solution, source, rho, 20, 1e-7);

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
