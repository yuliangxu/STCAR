#include <RcppArmadillo.h>
#include "STCAR_help.h"
using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::export]]
bool geweke_test(arma::vec& x,  double p_thresh = 0.05, 
                 double frac1=0.1, double frac2=0.5) {
  // Obtaining namespace of Matrix package
  Environment pkg = Environment::namespace_env("coda");
  
  // Picking up Matrix() function from Matrix package
  Function f = pkg["geweke.diag"];
  List f_out = f(x,Named("frac1")=frac1, _["frac2"]=frac2);
  double z_score = f_out["z"];
  if(z_score<0){
    z_score = -1.0*z_score;
  }
  bool test = (1.0-arma::normcdf(z_score))*2 <= p_thresh; 
  
  return test;
  
}

// [[Rcpp::export]]
arma::uvec complement(arma::uword start, arma::uword end, arma::uword n) {
  arma::uvec y1 = arma::linspace<arma::uvec>(0, start-1, start);
  arma::uvec y2 = arma::linspace<arma::uvec>(end+1, n-1, n-1-end);
  arma::uvec y = arma::join_cols(y1,y2);
  return y;
}
// [[Rcpp::export]]
arma::vec High_to_low_vec(arma::vec& High_vec, int L, Rcpp::List& Phi_Q,
                          Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec Low_vec(L,1);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    Low_vec(L_range) = Q.t()*High_vec(p_idx);
  }
  return Low_vec;
  
}
// [[Rcpp::export]]
arma::mat High_to_low(const arma::mat& High_mat, int L, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = High_mat.n_cols;
  arma::mat Low_mat = arma::zeros(L,n);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    // Rcout<<"High_to_low..1"<<std::endl;
    // Rcout<<"L_range = "<<std::endl;
    // Rcout<<L_range<<std::endl;
    // Rcout<<"size(Q) = "<<size(Q)<<std::endl;
    // Rcout<<"range of p_idx = "<<min(p_idx)<<","<<max(p_idx)<<std::endl;
    // Rcout<<"length of p_idx = "<<size(p_idx)<<std::endl;
    // Rcout<<"size of High_mat = "<<size(High_mat)<<std::endl;
    Low_mat.rows(L_range) = Q.t()*High_mat.rows(p_idx);
    // Rcout<<"High_to_low..2"<<std::endl;
  }
  return Low_mat;
  
}
// [[Rcpp::export]]
arma::mat Low_to_high(arma::mat& Low_mat, int p, Rcpp::List& Phi_Q,
                      Rcpp::List& region_idx, Rcpp::List& L_idx){
  int num_region = region_idx.size();
  int n = Low_mat.n_cols;
  arma::mat High_mat(p,n);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_mat.rows(p_idx) = Q*Low_mat.rows(L_range);
  }
  return High_mat;
}
// [[Rcpp::export]]
arma::colvec Low_to_high_vec(const arma::colvec& Low_vec, int p,
                             const Rcpp::List& Phi_Q,
                             const Rcpp::List& region_idx, 
                             const Rcpp::List& L_idx){
  int num_region = region_idx.size();
  arma::colvec High_vec(p,1);
  for(arma::uword r=0; r<num_region; r++){
    arma::uvec p_idx = region_idx[r];
    arma::uvec L_range = L_idx[r];
    arma::mat Q = Phi_Q[r];
    High_vec(p_idx) = Q*Low_vec(L_range);
  }
  return High_vec;
}


arma::mat get_mix_prob(arma::vec v1, arma::vec v2, arma::vec v3) {
  arma::mat log_mix_prob = join_horiz(v1, v2, v3);
  log_mix_prob.each_col() -= max(log_mix_prob,1);
  arma::mat mix_prob = exp(log_mix_prob);
  mix_prob.each_col() /= sum(mix_prob,1);
  return mix_prob;
}


// [[Rcpp::export]]
arma::vec sample_truncated_normal_vec(const arma::vec& mu, const arma::vec& sigma, double lambda,
                                      int delta ){
  
  arma::vec a_cdf = arma::normcdf((-lambda-mu)/sigma);
  arma::vec b_cdf = arma::normcdf((lambda-mu)/sigma);
  arma::vec u = arma::randu(size(mu)); 
  arma::vec u_; 
  
  // arma::uvec pos = find(delta==1);
  // arma::uvec zero = find(delta==0);
  // arma::uvec neg = find(delta==-1);
  u_.zeros(size(mu));
  if(delta==1){
    u_ = u%(1-b_cdf) + b_cdf;
  }
  if(delta==0){
    u_= u%(b_cdf-a_cdf) + a_cdf;
  }
  if(delta== -1){
    u_ = u%a_cdf;
  }
  
  // avoid INF in alpha
  u_(arma::find(u_== -1)).fill(-1+1e-12);
  u_(arma::find(u_==1)).fill(1-1e-12);
  u_(arma::find(u_==0)).fill(1e-12);
  Rcpp::NumericVector u_Rcpp = Rcpp::NumericVector(Rcpp::wrap(u_));
  Rcpp::NumericVector x_Rcpp = Rcpp::qnorm(u_Rcpp, 0.0, 1.0);
  arma::vec x_ = Rcpp::as<arma::vec>(x_Rcpp);
  return sigma%x_ + mu;
}

// [[Rcpp::export]]
arma::mat truncated_normal_stats_vec(const arma::vec& mu, const arma::vec& sigma, 
                                     double lambda,
                                     int delta , int output){
  arma::vec alpha; arma::vec beta; arma::vec Z;
  arma::vec tn_mean; arma::vec tn_var; arma::vec tn_2moment;
  arma::vec sigma_sq = sigma%sigma;
  if(delta == 1){
    alpha = (lambda - mu)/sigma;
    Z = 1-normcdf(alpha);
    if(output == 3){return Z;}
    arma::vec phi_alpha_over_z = normpdf(alpha)/Z;
    tn_mean = mu + sigma%phi_alpha_over_z;
    if(output == 0){return tn_mean;}
    tn_var = sigma_sq%(1+phi_alpha_over_z - phi_alpha_over_z%phi_alpha_over_z );
    if(output == 1){return tn_var;}
    tn_2moment = tn_var + tn_mean%tn_mean;
    if(output == 2){return tn_2moment;}
    if(output == 4){return join_horiz(tn_mean, tn_var, tn_2moment, Z);}
  }else if(delta == 0){
    alpha = (-lambda - mu)/sigma;
    beta = (lambda - mu)/sigma;
    Z = normcdf(beta)-normcdf(alpha);
    if(output == 3){return Z;}
    arma::vec phi_over_z = ( normpdf(beta) - normpdf(alpha) ) /Z;
    tn_mean = mu - sigma%phi_over_z;
    if(output == 0){return tn_mean;}
    tn_var = sigma_sq%(1 - (beta%normpdf(beta) - alpha%normpdf(alpha))/Z - phi_over_z%phi_over_z );
    if(output == 1){return tn_var;}
    tn_2moment = tn_var + tn_mean%tn_mean;
    if(output == 2){return tn_2moment;}
    if(output == 4){return join_horiz(tn_mean, tn_var, tn_2moment, Z);}
  }else if(delta == -1){
    beta = (-lambda - mu)/sigma;
    Z = normcdf(beta);
    if(output == 3){return Z;}
    arma::vec phi_over_z = normpdf(beta)/Z;
    tn_mean = mu - sigma%phi_over_z;
    if(output == 0){return tn_mean;}
    tn_var = sigma_sq%(1 - beta%phi_over_z - phi_over_z%phi_over_z );
    if(output == 1){return tn_var;}
    tn_2moment = tn_var + tn_mean%tn_mean;
    if(output == 2){return tn_2moment;}
    if(output == 4){return join_horiz(tn_mean, tn_var, tn_2moment, Z);}
  }else{
    Rcpp::Rcout<<"Error: delta must be -1,0,1"<<std::endl;
    return arma::zeros(1,1);
  }
  return arma::zeros(1,1);
}

