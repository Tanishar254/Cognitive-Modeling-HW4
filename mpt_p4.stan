data {
    int<lower=1> N;
    int<lower=1> K;
    matrix[N, K] X;
    array[N] int<lower=0, upper=1> y;

    int<lower=1> M;
    matrix[M, K] X_test;
}
parameters {
    real alpha;
    vector[K] beta;
}
model {
    alpha ~ normal(0, 2.5);
    beta ~ normal(0, 2.5);

    y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {
    vector[N] p;
    vector[N] log_lik;
    vector[M] p_test;
    array[M] int<lower=0, upper=1> y_test_rep;

    for (n in 1:N) {
        p[n] = inv_logit(alpha + X[n] * beta);
        log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + X[n] * beta);
    }

    for (m in 1:M) {
        p_test[m] = inv_logit(alpha + X_test[m] * beta);
        y_test_rep[m] = bernoulli_rng(p_test[m]);
    }
}