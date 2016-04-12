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
