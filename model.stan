
    data {
        int<lower=1> N;
        int<lower=1> q;
        matrix[N, q] X;
        vector[N] Y;
        real hyperparam_alpha_mu;
        real<lower=0> hyperparam_alpha_sigma;
        real hyperparam_beta_mu;
        real<lower=0> hyperparam_beta_sigma;
        real<lower=0> hyperparam_sigma_beta;
        int<lower=0, upper=1> use_likelihood;
    }
    parameters {
        real alpha;
        vector[q] beta;
        real<lower=0> sigma;
    }
    transformed parameters {
        vector[N] mu = alpha + X * beta;
    }
    model {
        alpha ~ normal(hyperparam_alpha_mu, hyperparam_alpha_sigma);
        beta ~ normal(hyperparam_beta_mu, hyperparam_beta_sigma);
        sigma ~ exponential(hyperparam_sigma_beta);
        if (use_likelihood) {
            Y ~ normal(mu, sigma);
        }
    }
    generated quantities {
        array[N] real Y_rep = normal_rng(mu, sigma);
    }
    