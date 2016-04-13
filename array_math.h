
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

void zero_array(real_t *m, idx_t n)
{
  for(idx_t i=0; i < n; i++)
    m[i] = 0;

  return;
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
  real_t total = 0.0;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP_N3(i, n)
    total += rho[i] * PW5(psi[i] + shift);

  return total*PW3(h);
}

real_t integrate_conformal_rho_derivative(real_t *psi, real_t *rho,
  real_t shift, idx_t n)
{
  // derivative of f(A) = \integral (\rho * 5 * (\psi + A)^4 - f  ) d V
  // used in Newton iteration to find root

  idx_t i;
  real_t total = 0.0;
  real_t h = H_LEN_FRAC/(real_t)(n);

  LOOP_N3(i, n)
  {
    total += rho[i] * PW4(psi[i] + shift);
  }
  
  return 5*total*PW3(h);
}

void initialize_random_array(real_t *array, idx_t n)
{
  // set random initial values
  int i, j, k;
  LOOP3_N(i,j,k,n)
  {
    array[G_INDEX(i, j, k, n)] = 1.0 + (real_t) rand()/RAND_MAX/1000;
  }
}

void shift_array_values(real_t *array, real_t shift, idx_t n)
{
  // shift array values; eg. to satisfy constraint
  idx_t i;
  LOOP_N3(i, n)
    array[i] += shift;
}

