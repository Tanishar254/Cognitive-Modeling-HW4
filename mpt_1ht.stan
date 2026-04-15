data {
  int<lower=1> N_old;
  int<lower=1> N_new;

  array[2] int<lower=0> y_old;
  array[2] int<lower=0> y_new;
}

parameters {
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> g;
}

transformed parameters {
  simplex[2] p_old;
  simplex[2] p_new;

  // OLD items
  p_old[1] = d + (1 - d) * g;
  p_old[2] = (1 - d) * (1 - g);

  // NEW items (1HT assumption)
  p_new[1] = g;
  p_new[2] = 1 - g;
}

model {
  d ~ beta(1, 1);
  g ~ beta(1, 1);

  y_old ~ multinomial(p_old);
  y_new ~ multinomial(p_new);
}

generated quantities {
  array[2] int yrep_old;
  array[2] int yrep_new;

  yrep_old = multinomial_rng(p_old, N_old);
  yrep_new = multinomial_rng(p_new, N_new);
}