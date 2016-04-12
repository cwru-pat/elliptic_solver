#include <iostream>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>

#define N 4
#define NX N
#define NY N
#define NZ N
#define POINTS ((NX)*(NY)*(NZ))
#define PI (4.0*atan(1.0))
// box size in hubble units
#define H_LEN_FRAC 0.5
#define dx (H_LEN_FRAC/(1.0*N))

#define LOOP3(i,j,k) \
  for(i=0; i<NX; ++i) \
    for(j=0; j<NY; ++j) \
      for(k=0; k<NZ; ++k)

#define AREA_LOOP(j,k) \
  for(j=0; j<NY; ++j) \
    for(k=0; k<NZ; ++k)

#define INTERNAL_LOOP3(i,j,k) \
  for(i=1; i<NX-1; ++i) \
    for(j=1; j<NY-1; ++j) \
      for(k=1; k<NZ-1; ++k)

/*-----------NEW DEFINITION----------------------------------------------*/

#define LOOP3_N(i,j,k,n) \
  for(i=0; i<n; ++i) \
    for(j=0; j<n; ++j) \
      for(k=0; k<n; ++k)

#define LOOP_N3(i, n) \
  for(i=0; i<PW3(n); ++i) \

#define NPRE 100 //number of pre relaxation
#define NPOST 100 //number of post relaxation
#define NCOARSE 1005 //numer of relaxation on coarse grids
#define ALPHA 0.33 //coefficient of estimated truncation error
#define MAX_DEPTH 10 //max interation depth
#define MIN_DEPTH 2 //coarsest grids depth, cannot be larger than the depth of finest grids

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

void zero_array(real_t *m, idx_t n)
{
  for(idx_t i=0; i < n; i++)
    m[i] = 0;

  return;
}

void alloc(real_t *m, idx_t n)
{
  m = new real_t[n*n*n];

  return;
}

void print_vector(real_t *m, idx_t n)
{
  for(int i = 0; i < n; i++)
    std::cout << std::fixed <<std::setprecision(7)<<m[i]<<" ";

  std::cout << "\n";
}

void print_mathematica_array(real_t *m, idx_t n)
{
  std::cout << "{";
  for(int i = 0; i < n; i++)
  {
    std::cout << "{";
    for(int j = 0; j < n; j++)
    {
      std::cout << "{";
      std::cout << std::fixed << m[G_INDEX(i, j, 0, n)];
      for(int k = 1; k < n; k++)
      {
        std::cout << std::fixed << ","<<m[G_INDEX(i, j, k, n)];
      }
      std::cout << "}";
      if(j != n-1)
        std::cout << ",";
    }
    std::cout << "}";
    if(i != n-1)
      std::cout << ',';
  }
  std::cout << "}";
}


real_t integrate_array(real_t *array, idx_t n) 
{
  // integrate array values
  idx_t i;
  real_t total = 0.0;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP_N3(i, n)
    total += array[i];

  return total*PW3(h);
}

real_t integrate_conformal_rho(real_t *psi, real_t *rho, real_t shift, idx_t n)
{
  // integrate rho * (\psi + shift)^5
  idx_t i;
  real_t res = 0.0;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP_N3(i, n)
    res += rho[i] * PW5(psi[i] + shift);

  return res*PW3(h);
}

real_t integrate_conformal_rho_derivative(real_t *psi, real_t *rho,
  real_t shift, idx_t n)
{
  // derivative of f(A) = \integral (\rho * 5 * (\psi + A)^4 - f  ) d V
  // used in Newton iteration to find root

  idx_t i;
  real_t res = 0.0;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP_N3(i, n)
  {
    res += rho[i] * PW4(psi[i] + shift);
  }
  
  return 5*res*PW3(h);
}

real_t exam_constraint( real_t *u, real_t *irho, real_t *irhs, real_t shift, idx_t n)
{
  // compute residual?

  idx_t i, j, k;
  real_t res = 0.0;

  LOOP3_N(i,j,k,n)
  {
    res += integrate_conformal_rho(u, irho, shift, n) - integrate_array(irhs, n);
  }

  return res;
}

real_t solve_constraint(real_t *u, real_t *irho, real_t *irhs, idx_t n)
{
  //Find root of function f(A) = \integral (\rho * (\psi + A)^5 - f  ) d V 
  idx_t i, j, k;
  real_t eps = -INF, res = NEWTON_INI
  real_t num, den;

  do
  {
    num = integrate_conformal_rho(u, irho, res, n) - integrate_array(irhs, n);
    den = integrate_conformal_rho_derivative(u, irho, res, n);

    res -= num/den;
    if( fabs(fabs(num/den) - eps) < CON_EPS )
      break;
    eps = fabs(num/den);
  } while(1);

  return res;
}

void shift_array_values(real_t *array, real_t shift, idx_t n)
{
  // shift array values; eg. to satisfy constraint
  idx_t i;
  LOOP_N3(i, n)
    array[i] += shift;
}

void residual(real_t *u, real_t *irho, real_t *res, idx_t n)
{
  // compute residual of constraint equation

  idx_t i, j, k;
  real_t h = (H_LEN_FRAC/(1.0*n));

  LOOP3_N(i,j,k,n)
  {
    res[G_INDEX(i, j, k, n)] = LAP(u, i, j, k, n, h);
    res[G_INDEX(i, j, k, n)] += irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]);    
  }

  return;
}

real_t relax(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
  // relax hamiltonian constraint?

  idx_t i, j, k, ipass, isw = 0, jsw, ksw;
  real_t temp, h = (H_LEN_FRAC/(1.0*n)), res =  INF;
  
  // white red grids relaxation
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
          temp = LAP(u, i, j, k, n, h) + irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]) 
          - irhs[G_INDEX(i, j, k, n)] ;
          u[G_INDEX(i, j, k, n)] -= temp/(-6/(h*h) + 
          5 * irho[G_INDEX(i, j, k, n)] * PW4(u[G_INDEX(i, j, k, n)]));
           
        }
      }
    }
  }
  
  // Regular Gauss-Seidel relaxation as alternative
  /*
  for(int i = 0; i < n; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n; k++)
      {
        temp = (LAP(u, i, j, k, n, h) + irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]) 
          - irhs[G_INDEX(i, j, k, n)]) /(-6/(h*h) + 
          5 * irho[G_INDEX(i, j, k, n)] * PW4(u[G_INDEX(i, j, k, n)]));
          u[G_INDEX(i, j, k, n)] -= temp;
          res = MAX(res, fabs(temp));
          //std::cout << temp<<"\n";
      }
    */
  return res;
}

real_t coarse_constraint_residual(real_t *phi, real_t *rho,
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

void initialize_random_array(real_t *array, idx_t n)
{
  // set random initial values
  int i, j, k;
  LOOP3_N(i,j,k,n)
  {
    array[G_INDEX(i, j, k, n)] = (real_t) rand()/1000;
  }
}

void solve_coarse_1(real_t *u, real_t *irhs, real_t *irho, idx_t n)
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

    temp = solve_constraint(u, irho, irhs, 1<<MIN_DEPTH);
    shift_array_values(u, temp, 1<<MIN_DEPTH);
    
    eps = coarse_constraint_residual(u, irho, irhs, 1<<MIN_DEPTH);
  }
}

void solve_coarse_0(real_t *u, real_t *irhs, real_t *irho, idx_t n)
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
    eps = coarse_constraint_residual(u, irho, irhs, 1<<MIN_DEPTH);
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

    real_t cc = u_coarse[G_INDEX(i,j,k,cn)];
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

void lop(real_t *u, real_t *irho, real_t *out, idx_t n)
{
  //calculating L_h(\tilde{u}_h) used in FMG
  idx_t i, j, k;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP3_N(i,j,k,n)
    out[G_INDEX(i, j, k, n)] = LAP(u, i, j, k, n, h)
          + irho[G_INDEX(i,j,k,n)] * PW5(u[G_INDEX(i,j,k,n)]);
  
  return;
}

void matrix_subtract(real_t *a, real_t *b, real_t *c, idx_t n)
{
  idx_t i;
  LOOP_N3(i,n)
    c[i] = a[i] - b[i];
}

void matrix_add(real_t *a, real_t *b, real_t *c, idx_t n)
{
  idx_t i;
  LOOP_N3(i,n)
    c[i] = a[i] + b[i];
}

real_t norm(real_t *u, idx_t n)
{
  idx_t i;
  real_t sum = 0.0;
  real_t h = (H_LEN_FRAC/(real_t)n);

  LOOP_N3(i,n)
    sum += u[i]*u[i];

  return sqrt(sum)/h;
}

void fas_multigrid(real_t *u, real_t *rhs, real_t *rho, idx_t ncycle, real_t eps)  
{
  // Solve non-linear constraint equation \nebla^2 \psi = -2pi \rho \psi with using Full multigrid in FAS scheme
  // u is the right hand side of eq. 19.6.23, which u=0, rho = rho for this equation.
  
  idx_t DEPTH = 0;

  // iterators
  idx_t i, j, k, jc, nf, jj, jpre, jpost;
  idx_t n = (1<<MIN_DEPTH);

  // define arrays that will be used in the iteration
  real_t *irhs[MAX_DEPTH], *itau[MAX_DEPTH], *iu[MAX_DEPTH], *irho[MAX_DEPTH], *itemp[MAX_DEPTH], tr_err, res; 
  
  while ((DEPTH/2) != N) 
    DEPTH++;

  irho[DEPTH] = new real_t[PW3(1<<DEPTH)];
  irhs[DEPTH] = new real_t[PW3(1<<DEPTH)];
  
  memcpy(irho[DEPTH], rho, sizeof(real_t)*PW3(1<<DEPTH));
  memcpy(irhs[DEPTH], rhs, sizeof(real_t)*PW3(1<<DEPTH));
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
  solve_coarse_0(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));
  
  for (j = MIN_DEPTH + 1; j <= DEPTH; j++)
  {
    n *= 2;
    iu[j] = new real_t[PW3(n)];
    itau[j] = new real_t[PW3(n)];
    itemp[j] = new real_t[PW3(n)]; 
    
    interpolate_coarse2fine(iu[j], iu[j-1], n);
    
    for(jc = 0; jc < ncycle; jc++)
    {
      //std::cout << ncycle;
      nf = n;
      
      for(jj = j; jj > MIN_DEPTH; jj--)
      {
        for(jpre =0; jpre < NPRE; jpre++)
        {
          if(relax(iu[jj], irhs[jj], irho[jj], nf) < IT_EPS)
            break;
          //shift_array_values(iu[jj], solve_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
        }
        lop(iu[jj], irho[jj], itemp[jj], nf);
        nf = nf>>1;
        restrict_fine2coarse(itemp[jj-1], itemp[jj], nf);
        //print_vector(itemp[jj-1], PW3(nf));;
        restrict_fine2coarse(iu[jj-1], iu[jj], nf);
        lop(iu[jj-1], irho[jj-1], itau[jj-1], nf);
        matrix_subtract(itau[jj-1], itemp[jj-1], itau[jj-1], nf);
        if(jj == j)
          tr_err = ALPHA * norm(itau[jj-1], nf);
        restrict_fine2coarse(irhs[jj-1], irhs[jj], nf);
        matrix_add(irhs[jj-1], itau[jj-1], irhs[jj-1], nf);
      }
      
      solve_coarse_0(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));
      
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
          //shift_array_values(iu[jj], solve_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
        }
      }
      lop(iu[j], irho[j], itemp[j], nf);
      matrix_subtract(itemp[j], irhs[j], itemp[j], nf);
      res = norm(itemp[j], nf);
      std::cout << "Depth "<<"Difference\n";
      std::cout << j<<" "<<std::fixed<< coarse_constraint_residual(iu[j], irho[j], irhs[j], 1<<j)<<"\n\n";
      
      if(res < tr_err) break;
    }
  }
  std::cout << "Final Difference\n";
  std::cout << std::fixed;
  std::cout <<   coarse_constraint_residual(iu[DEPTH],rho,rhs,N)<<"\n";
  memcpy(u, iu[DEPTH], sizeof(real_t) * PW3((1<<DEPTH)));
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
  real_t *psi_solution, *psi_trial, *rho, *res, *source
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

  LOOP3(i,j,k)
  {
    idx_t p = G_INDEX(i, j, k, N);

    source[p] = 0;
    // generating standard solution
    psi_trial[p] = 1.0 - sin( 2.0 * PI * n1 * (real_t)i/ (N) + phi1)
                 * sin( 2.0 * PI * n2 * (real_t)j/ (N) + phi2)
                 * sin( 2.0 * PI * n3 * (real_t)k/ (N) + phi3)/20.0;
  }

  LOOP3(i,j,k)
  {
    idx_t p = G_INDEX(i, j, k, N);
    // generating rho according to standard solution
    rho[p] = (-LAP(psi_trial, i, j, k, N, h) + source[p]) / PW5(psi_trial[p]);
  }

  fas_multigrid(psi_solution, source, rho, 3, 1e-7);

  freopen("rho.txt","w", stdout);
  print_mathematica_array(rho, (N));
  freopen("rhs.txt", "w", stdout);
  print_mathematica_array(rhs, N);
  freopen("res.txt", "w", stdout);
  print_mathematica_array(psi_solution, (N));
  freopen("std.txt", "w", stdout);
  print_mathematica_array(psi_trial, N);
  
  delete [] psi_solution;
  delete [] source;
  delete [] rho;
  delete [] psi_trial;
  delete [] res;
  
  return 0;
}
