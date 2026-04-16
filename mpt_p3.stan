data {
    int<lower=1> N;
    int<lower=1> K;
    matrix[N, K] X;
    array[N] int<lower=0, upper=1> y;
}
parameters {
    real alpha;
    vector[K] beta;
}
model {
    // Weakly informative priors
    alpha ~ normal(0, 2.5);
    beta ~ normal(0, 2.5);

    // Logistic regression likelihood
    y ~ bernoulli_logit(alpha + X * beta);
}
generated quantities {
    vector[N] p;
    vector[N] log_lik;

    for (n in 1:N) {
        p[n] = inv_logit(alpha + X[n] * beta);
        log_lik[n] = bernoulli_logit_lpmf(y[n] | alpha + X[n] * beta);
    }
}