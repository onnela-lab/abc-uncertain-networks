data {
  int<lower=1> N;
  int<lower=1> times;
  int<lower=0> X_flat[times,N];
  int<lower=0> N_zero[times];
}

parameters {
  positive_ordered[2] n_0;
  positive_ordered[2] n_1;
  positive_ordered[2] n_2;
  positive_ordered[2] n_3;
  positive_ordered[2] n_4;
  real<lower=0> p[times];
  real<lower=0, upper=1> rho[times];
}

model {
  for (t in 1:times){
    n_0[1] ~ gamma(2,4);
    n_0[2] ~ gamma(2,4);
    n_1[1] ~ gamma(2,4);
    n_1[2] ~ gamma(2,4);
    n_2[1] ~ gamma(2,4);
    n_2[2] ~ gamma(2,4);
    n_3[1] ~ gamma(2,4);
    n_3[2] ~ gamma(2,4);
    n_4[1] ~ gamma(2,4);
    n_4[2] ~ gamma(2,4);
    p[t] ~ gamma(2,4);
    #p[2] ~ beta(1,4);
    rho[t] ~ beta(1,20);
  }
  for (t in 1:times){
    # Handle all the zero-ed out elements first.
    vector[2] curr_n;
    if (t==1){
      curr_n = n_0;
    }
    if (t==2){
      curr_n = n_1;
    }
    if (t==3){
      curr_n = n_2;
    }
    if (t==4){
      curr_n = n_3;
    }
    if (t==5){
      curr_n = n_4;
    }
    real zero_log_mu_ij_0 = neg_binomial_lpmf(0|curr_n[1],1); 
    real zero_log_mu_ij_1 = neg_binomial_lpmf(0|curr_n[2],p[t]); 
    real log_nu_ij_0 = bernoulli_lpmf(0|rho[t]);
    real log_nu_ij_1 = bernoulli_lpmf(1|rho[t]);
    real zero_z_ij_0 = zero_log_mu_ij_0 + log_nu_ij_0;
    real zero_z_ij_1 = zero_log_mu_ij_1 + log_nu_ij_1;
    if (zero_z_ij_0 > zero_z_ij_1) {target += N_zero[t]*(zero_z_ij_0 + log1p_exp(zero_z_ij_1 - zero_z_ij_0));}
    else {target += N_zero[t]*(zero_z_ij_1 + log1p_exp(zero_z_ij_0 - zero_z_ij_1));}

    for (i in 1:N){
        if (X_flat[t,i] > 0){
          real log_mu_ij_0 = neg_binomial_lpmf(X_flat[t,i]|curr_n[1], 1);
          real log_mu_ij_1 = neg_binomial_lpmf(X_flat[t,i]|curr_n[2], p[t]); 

          log_nu_ij_0 = bernoulli_lpmf(0 | rho[t]);
          log_nu_ij_1 = bernoulli_lpmf(1 | rho[t]);

          real z_ij_0 = log_mu_ij_0 + log_nu_ij_0;
          real z_ij_1 = log_mu_ij_1 + log_nu_ij_1;
          if (z_ij_0 > z_ij_1) {target += z_ij_0 + log1p_exp(z_ij_1 - z_ij_0);}
          else {target += z_ij_1 + log1p_exp(z_ij_0 - z_ij_1);}
      }
      }
  }
}
