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


#define NPRE 100//number of pre relaxation
#define NPOST 100 //number of post relaxation
#define NCOARSE 1005 //numer of relaxation on coarse grids
#define ALPHA 0.33 //coefficient of estimated truncation error
#define MAX_DEPTH 10 //max interation depth
#define MIN_DEPTH 2 //coarsest grids depth, cannot be larger than the depth of finest grids
#define G_INDEX(i, j, k, n) (((i+n)%(n))*(n)*(n) + ((j+n)%(n))*(n) + (k+n)%(n))
#define LAP(mat, i, j, k, n, h) ((mat[G_INDEX(i+1, j, k, n)]+mat[G_INDEX(i-1, j, k, n)]+mat[G_INDEX(i, j+1, k, n)]+mat[G_INDEX(i, j-1, k, n)]+mat[G_INDEX(i, j, k+1, n)]+mat[G_INDEX(i, j, k-1, n)]-6.0*mat[G_INDEX(i, j, k, n)])/(h*h))
#define PW5(a) ( (a)*(a)*(a)*(a)*(a))
#define PW4(a) ((a)*(a)*(a)*(a))
#define PW3(a) ((a)*(a)*(a))
#define PW2(a) ((a)*(a))
#define MAX(a, b) ((a>b)? a: b )
#define COARSE_EPS 1e-9
#define EPS 1e-6
#define IT_EPS 1e-20
#define CON_EPS 1e-9 //EPS used when finding shift to speed up interation
#define INF 1e100
#define PW3(a) ((a)*(a)*(a))
#define NEWTON_INI 1.0
typedef double real_t;
typedef long int idx_t;

void fill0(real_t *m, idx_t n)
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

void v_out(real_t *m, idx_t n)
{
    for(int i = 0; i < n; i++)
        std::cout<<std::fixed<<std::setprecision(7)<<m[i]<<" ";
    std::cout<<"\n";
}

void v_out_mathematica(real_t *m, idx_t n)
{
    std::cout<<"{";
    for(int i = 0; i < n; i++)
    {
        std::cout<<"{";
        for(int j = 0; j < n; j++)
        {
            std::cout<<"{";
            std::cout<<std::fixed<<std::setprecision(15)<<m[G_INDEX(i, j, 0, n)];
            for(int k = 1; k < n; k++)
            {
                std::cout<<std::fixed<<std::setprecision(15)<<","<<m[G_INDEX(i, j, k, n)];
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


real_t integrate_rhs(real_t *irhs, idx_t n) 
{
    //integrate rhs
    idx_t i, j, k;
    real_t res = 0.0, h = H_LEN_FRAC/(real_t)(n);
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
            {
                res += irhs[G_INDEX(i, j, k, n)];
            }
    return res*PW3(h);
}

real_t integrate_rho(real_t *u, real_t *irho, real_t shift, idx_t n)
{
    //integrate rho * \psi^2
    idx_t i, j, k;
    real_t res = 0.0, h = H_LEN_FRAC/(real_t)(n);
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
            {
                res += irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)] + shift);
                
            }
    return res*PW3(h);
}

real_t integrate_der_rho(real_t *u, real_t *irho, real_t shift, idx_t n)
{
    // derivative of f(A) = \integral (\rho * (\psi + A)^5 - f  ) d V , used in Newton interation to find root
    idx_t i, j, k;
    real_t res = 0.0, h = H_LEN_FRAC/(real_t)(n);
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
            {
                res += irho[G_INDEX(i, j, k, n)] * PW4(u[G_INDEX(i, j, k, n)] + shift);
            }
    return 5*res*PW3(h);
}

real_t exam_constraint( real_t *u, real_t *irho, real_t *irhs, real_t shift, idx_t n)
{
    //exam the difference of f(A) from 0 after solving constraint
    idx_t i, j, k;
    real_t res = 0.0;
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
            {
                res += integrate_rho(u, irho, shift, n) - integrate_rhs(irhs, n);
            }
    return res;
}

real_t solve_constraint(real_t *u, real_t *irho, real_t *irhs, idx_t n)
{
    //Find root of function f(A) = \integral (\rho * (\psi + A)^5 - f  ) d V 
    idx_t i, j, k;
    real_t eps = -INF, res = NEWTON_INI, num, den;
    do
    {
        num = integrate_rho(u, irho, res, n) - integrate_rhs(irhs, n);
        den = integrate_der_rho(u, irho, res, n);
        res -= num/den;
        if( fabs(fabs(num/den) - eps) < CON_EPS )
            break;
        eps = fabs(num/den);
    }while(1);
    return res;
}

void make_shift(real_t *u, real_t shift, idx_t n)
{
    //make a shift of solution to satisfy constraint
    for(idx_t i = 0; i < PW3(n); i++)
        u[i] += shift;
}

void residue(real_t *u, real_t *irho, real_t *res, idx_t n)
{
    
    idx_t i, j, k;
    real_t h = (H_LEN_FRAC/(1.0*n));
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
            {
                res[G_INDEX(i, j, k, n)] = LAP(u, i, j, k, n, h);
                
                res[G_INDEX(i, j, k, n)] += irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]);
                
            }               
    return;
}

real_t relax(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
    idx_t i, j, k, ipass, isw = 0, jsw, ksw;
    real_t temp, h = (H_LEN_FRAC/(1.0*n)), res =  INF;
    
    //white red grids relaxation
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
    
    //Gegular Gauss-Seidel relaxation as alternative
    
   /*for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
            {
                temp = (LAP(u, i, j, k, n, h) + irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]) 
                    - irhs[G_INDEX(i, j, k, n)]) /(-6/(h*h) + 
                    5 * irho[G_INDEX(i, j, k, n)] * PW4(u[G_INDEX(i, j, k, n)]));
                    u[G_INDEX(i, j, k, n)] -= temp;
                    res = MAX(res, fabs(temp));
                    //std::cout<<temp<<"\n";
            }*/
    return res;
}

real_t dif(real_t *u, real_t *irho, real_t *irhs, idx_t n)
{
    //Calculating differnet to rhs
    real_t res, h = (H_LEN_FRAC/(1.0*n)), eps = -INF;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
            {
                
                res = LAP(u, i, j, k, n, h) + 
                irho[G_INDEX(i, j, k, n)] * PW5(u[G_INDEX(i, j, k, n)]); 
                

                eps = MAX(eps, fabs(res - irhs[G_INDEX(i, j, k, n)]));
            }
    return eps;
}

void set_init(real_t *m, idx_t n)
{
    //set the initial value before relaxation
	for(int i =0; i < (n); i++)
        for(int j = 0; j < (n); j++)
            for(int k=0; k < (n); k++)
            {
				
				m[G_INDEX(i, j, k, n)] = (real_t) rand()/1000;
			}
}
void solve_coarse_1(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
    //solve the first order appproximation equation: \nabla \ksi + 2 \pi \rho * 5 \ksi = irhs - 2\pi \rho
    //and take u = 1+ ksi as our initial value.
    //with speed up scheme
    idx_t i,j,k;
    real_t  h = H_LEN_FRAC/(1<<MIN_DEPTH), eps = INF, temp;
    set_init(u, (1<<MIN_DEPTH));
    idx_t nn = NCOARSE, cnt=0;
    //std::cout<<"dsfsd";
    while(nn--)
    {
        cnt++;
        
        relax(u, irhs, irho, 1<<MIN_DEPTH);
        
        temp = solve_constraint(u, irho, irhs, 1<<MIN_DEPTH);
        
        make_shift(u, temp, 1<<MIN_DEPTH);
        
        eps = dif(u, irho, irhs, 1<<MIN_DEPTH);
        
    }
}

void solve_coarse_0(real_t *u, real_t *irhs, real_t *irho, idx_t n)
{
    //solve the first order appproximation equation: \nabla \ksi + 2 \pi \rho * 5 \ksi = irhs - 2\pi \rho
    //and take u = 1+ ksi as our initial value.
    //with out coarse scheme
    idx_t i,j,k;
    real_t  h = H_LEN_FRAC/(1<<MIN_DEPTH), eps = INF, temp;
    set_init(u, (1<<MIN_DEPTH));
    idx_t nn = NCOARSE;
    //std::cout<<"dsfsd";
    while(nn--)
    {
        relax(u, irhs, irho, 1<<MIN_DEPTH);
        
        
        eps = dif(u, irho, irhs, 1<<MIN_DEPTH);
    
    }
}

void restrict(real_t *u_coarse, real_t *u_fine, idx_t n_coarse) 
{
    //restrict scheme: 1*(1/8) + 6 * (1/16) + 12 * (1/32) + 8 * (1/64)
    idx_t i, j, k, fn = n_coarse*2, fi, fj, fk;
    for(i = 0; i < n_coarse; i++)
        for(j=0; j < n_coarse; j++)
            for(k=0; k< n_coarse; k++)
            {
                fi = i<<1;
                fj = j<<1;
                fk = k<<1;
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


void interpolat(real_t *u_fine, real_t *u_coarse, idx_t n_fine) 
{
    //restrict scheme: 1*(1/8) + 6 * (1/16) + 12 * (1/32) + 8 * (1/64)
    idx_t i, j, k, cn = n_fine/2, fi, fj, fk, fn = n_fine;
    fill0(u_fine, n_fine*n_fine*n_fine);
    for(i = 0; i < cn; i++)
        for(j=0; j < cn; j++)
            for(k=0; k< cn; k++)
            {
                fi = i<<1;
                fj = j<<1;
                fk = k<<1;
                real_t cc = u_coarse[G_INDEX(i,j,k,cn)];
                u_fine[G_INDEX(fi,fj,fk,n_fine)] += cc;
                
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
    for(i = 0; i < n; i++)
        for(j = 0; j < n; j++)
            for(k = 0; k < n; k++)
                out[G_INDEX(i, j, k, n)] = LAP(u, i, j, k, n, h) + irho[G_INDEX(i,j,k,n)] * PW5(u[G_INDEX(i,j,k,n)]);
    return;
}

void mat_sub(real_t *a, real_t *b, real_t *c, idx_t n)
{
    idx_t i;
    for(i = 0; i < PW3(n); i++)
        c[i] = a[i] - b[i];
}

void mat_add(real_t *a, real_t *b, real_t *c, idx_t n)
{
    idx_t i;
    for(i = 0; i < PW3(n); i++)
        c[i] = a[i] + b[i];
}

real_t norm(real_t *u, idx_t n)
{
    real_t sum = 0.0;
    for(int i = 0; i < PW3(n); i++)
        sum += (u[i] * u[i])/(H_LEN_FRAC/(real_t)n)/(H_LEN_FRAC/(real_t)n);
    return sqrt(sum);
}



void fas_multigrid(real_t *u, real_t *rhs, real_t *rho, idx_t ncycle, real_t eps)    
{
    //Solve non-linear constraint equation \nebla^2 \psi = -2pi \rho \psi with using Full multigrid in FAS scheme
    //u is the right hand side of eq. 19.6.23, which u =0, rho = rho for this equation.
    
    idx_t DEPTH =0, i, j, k, jc, n = (1<<MIN_DEPTH), nf, jj, jpre, jpost;
    //define arrays that will be used in the iteration
    real_t *irhs[MAX_DEPTH], *itau[MAX_DEPTH], *iu[MAX_DEPTH], *irho[MAX_DEPTH], *itemp[MAX_DEPTH], tr_err, res; 
    
    while ((1<<DEPTH)!=N) 
        DEPTH++;
    irho[DEPTH] = new real_t[PW3(1<<DEPTH)];
    //alloc(irhs[DEPTH], 1<<(DEPTH));
    irhs[DEPTH] = new real_t[PW3(1<<DEPTH)];
    
    memcpy(irho[DEPTH], rho, sizeof(real_t)*PW3(1<<DEPTH));
    memcpy(irhs[DEPTH], rhs, sizeof(real_t)*PW3(1<<DEPTH));
    i = DEPTH;
    while(i>MIN_DEPTH)
    {
        irho[--i] = new real_t[PW3(1<<i)];
        irhs[i] = new real_t[PW3(1<<i)];
        restrict(irho[i], irho[i+1], 1<<i);
        restrict(irhs[i], irhs[i+1], 1<<i);
    }
    
    iu[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
    itau[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
    itemp[MIN_DEPTH] = new real_t[PW3((1<<MIN_DEPTH))];
    set_init(iu[MIN_DEPTH], (1<<MIN_DEPTH));
    //Currently using multigrid without applying constraint of integral in speed up
    solve_coarse_0(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));
    
	
    for (j = MIN_DEPTH + 1; j <= DEPTH; j++)
    {
        n *= 2;
        iu[j] = new real_t[PW3(n)];
        itau[j] = new real_t[PW3(n)];
        itemp[j] = new real_t[PW3(n)]; 
        
        interpolat(iu[j], iu[j-1], n);
        
        for(jc = 0; jc < ncycle; jc++)
        {
            //std::cout<<ncycle;
            nf = n;
            
            for(jj = j; jj > MIN_DEPTH; jj--)
            {
                for(jpre =0; jpre < NPRE; jpre++)
                {
                    if(relax(iu[jj], irhs[jj], irho[jj], nf) < IT_EPS)
                        break;
                    //make_shift(iu[jj], solve_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
                }
                lop(iu[jj], irho[jj], itemp[jj], nf);
                nf = nf>>1;
                restrict(itemp[jj-1], itemp[jj], nf);
                //v_out(itemp[jj-1], PW3(nf));;
                restrict(iu[jj-1], iu[jj], nf);
                lop(iu[jj-1], irho[jj-1], itau[jj-1], nf);
                mat_sub(itau[jj-1], itemp[jj-1], itau[jj-1], nf);
                if(jj == j)
                    tr_err = ALPHA * norm(itau[jj-1], nf);
                restrict(irhs[jj-1], irhs[jj], nf);
                mat_add(irhs[jj-1], itau[jj-1], irhs[jj-1], nf);
            }
            
            solve_coarse_0(iu[MIN_DEPTH], irhs[MIN_DEPTH], irho[MIN_DEPTH], PW3(1<<MIN_DEPTH));
            
            
            nf = 1<<MIN_DEPTH;
            for(jj = MIN_DEPTH + 1; jj <= j; jj++)
            {
                
                restrict(itemp[jj - 1], iu[jj], nf);
                mat_sub(iu[jj - 1], itemp[jj-1], itemp[jj-1], nf);
                nf *=2;
                interpolat(itau[jj], itemp[jj-1], nf);
                
                mat_add(iu[jj], itau[jj], iu[jj], nf);
                for(jpost = 0; jpost < NPOST; jpost++)
                {
                    if(relax(iu[jj], irhs[jj], irho[jj], nf) < IT_EPS)
                        break;
                    //make_shift(iu[jj], solve_constraint(iu[jj], irho[jj], irhs[jj], nf), nf);
                }
            }
            lop(iu[j], irho[j], itemp[j], nf);
            mat_sub(itemp[j], irhs[j], itemp[j], nf);
            res = norm(itemp[j], nf);
			std::cout<<"Depth "<<"Difference\n";
            std::cout<<j<<" "<<std::fixed<<std::setprecision(15)<<dif(iu[j], irho[j], irhs[j], 1<<j)<<"\n\n";
            
            if(res < tr_err) break;
        }
    }
    std::cout<<"Final Difference\n";
    std::cout << std::fixed;
    std::cout<< std::setprecision(15)<<dif(iu[DEPTH],rho,rhs,N)<<"\n";
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
    
    //Solving equations Lap (u) + rho * u = rhs

    int gn = 0, n=8;
	real_t **m, **mm, **mmm, *u, *f, *rho, *res, *rhs,  h = H_LEN_FRAC/(real_t)(N);
    
    srand(129);
    
    u = new real_t[PW3(N)];
    rhs = new real_t[PW3(N)];
    rho = new real_t[PW3(N)];
    f = new real_t[PW3(N)];
	res =new real_t[PW3(N)];

    real_t n1= 3, n2 = 5, n3 = 7, phi1 = 0, phi2 = 0, phi3 = 0;
    for(int i =0; i < (N); i++)
        for(int j = 0; j < (N); j++)
            for(int k=0; k < (N); k++)
            {
                idx_t p = G_INDEX(i, j, k, N);
        
                rhs[p] = 0;
                //generating standard solution
				f[p] = 1.0 - sin( 2.0 * PI * n1 * (real_t)i/ (N) + phi1) * sin( 2.0 * PI * n2 * (real_t)j/ (N) + phi2) * sin( 2.0 * PI * n3 * (real_t)k/ (N) + phi3)/20.0;
                
            }
	for(int i =0; i < (N); i++)
        for(int j = 0; j < (N); j++)
            for(int k=0; k < (N); k++)
            {
                idx_t p = G_INDEX(i, j, k, N);
                //gnerating rho according to standard solution
                rho[p] = (-LAP(f, i, j, k, N, h)+ rhs[p])/PW5(1.0 - sin( 2.0 * PI * n1 * (real_t)i/ (N) + phi1) * sin( 2.0 * PI * n2 * (real_t)j/ (N) + phi2) * sin( 2.0 * PI * n3 * (real_t)k/ (N) + phi3)/20.0);
			}
	fas_multigrid(u, rhs, rho, 3, 1e-7);
	
	freopen ("rho.txt","w",stdout);
	v_out_mathematica(rho, (N));
    freopen ("rhs.txt","w",stdout);
    v_out_mathematica(rhs, N);
    freopen ("res.txt","w",stdout);
    v_out_mathematica(u, (N));
    freopen ("std.txt","w",stdout);
	v_out_mathematica(f, N);
	
    delete [] u ;
    delete [] rhs;
    delete [] rho;
    
	return 0;
}
