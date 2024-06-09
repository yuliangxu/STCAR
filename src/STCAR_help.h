#ifndef STCAR_HELP_H
#define STCAR_HELP_H

#include <RcppArmadillo.h>

bool geweke_test(arma::vec& x,  double p_thresh, 
                 double frac1, double frac2);

arma::uvec complement(arma::uword start, arma::uword end, arma::uword n);
arma::vec High_to_low_vec(arma::vec& High_vec, int L, Rcpp::List& Phi_Q,
                          Rcpp::List& region_idx, Rcpp::List& L_idx);

arma::mat High_to_low(const arma::mat& High_mat, int L, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx);

arma::mat Low_to_high(arma::mat& Low_mat, int p, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx);

arma::colvec Low_to_high_vec(const arma::colvec& Low_vec, int p,
                             const Rcpp::List& Phi_Q,
                             const Rcpp::List& region_idx, 
                             const Rcpp::List& L_idx);

arma::mat get_mix_prob(arma::vec v1, arma::vec v2, arma::vec v3);

arma::vec sample_truncated_normal_vec(const arma::vec& mu, const arma::vec& sigma, double lambda,
                                      int delta );

arma::mat truncated_normal_stats_vec(const arma::vec& mu, const arma::vec& sigma, 
                                     double lambda,
                                     int delta , int output);

#endif