#include "STCAR_help.h"
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::depends(BH)]]
#include <RcppArmadillo.h>


using namespace Rcpp;
// using namespace arma;

// #include <progress.hpp>
// #include <progress_bar.hpp>
// // [[Rcpp::depends(RcppProgress)]]


class SonI_standalone{
private:
  struct LinearRegData{
    int n;
    int p;
    int m;
    arma::vec y;
    arma::mat M; // n by p
    arma::rowvec M_l2;
    arma::rowvec X_l2;
    arma::mat X; // n by m
    arma::mat XtX_eigenval;
    arma::mat XtX_eigenvec;
    arma::mat MtM_eigenval;
    double lambda;
  } dat, sub_dat;
  
  struct SVD_on_M{
    arma::mat U; // n by n
    arma::vec D; // n by 1
    arma::vec D2;
    arma::mat V; // p by n
    arma::mat M_star; // n by p
    arma::mat D_Vt;
  }svd_m, svd_x;
  
  
  
  
  
  struct VBLinearRegParas{
    arma::vec E_beta;
    arma::vec E_mu;
    arma::vec E_IP;
    arma::vec E_f_mu;
    arma::vec Var_beta; // marginal
    arma::vec E_gamma;
    arma::mat Var_gamma; //joint
    arma::vec var_gamma_marg; //marginal
    arma::vec E_neighbor_mean;
    arma::vec E_mu_neighbor_2moment;
    arma::vec E_mu2;
    arma::vec E_f_mu2;
    arma::vec Var_f_mu;
    arma::mat mix_prob; 

    arma::vec E_delta_rho;
    arma::vec E_delta_rho_prior;
    
    arma::vec E_inv_sigma_mu_sq;
    
    double E_inv_sigma_sq;
    double E_inv_sigma_sq_beta;
    double E_inv_tau_sq_mu;
    double E_inv_sigma_sq_gamma;
    double E_inv_a;
    double E_inv_a_beta;
    double E_inv_a_gamma;
    double E_SSE;
    double E_SSE_beta;
    double E_SSE_gamma;
    double ELBO;
  } vb_paras;

  struct GibbsLinearRegParas{
    arma::vec beta;
    arma::vec IP;
    arma::vec f_mu;
    arma::vec gamma;
    arma::vec delta_rho;
    arma::vec delta_rho_prior;
    arma::mat mix_prob;

    arma::vec neighbor_mean;

    double inv_sigma_sq;
    double inv_sigma_sq_beta;
    double inv_tau_sq_mu;
    double inv_sigma_sq_gamma;
    double inv_a;
    double inv_a_beta;
    double inv_a_gamma;
    double SSE;

    double loglik;

    

  }paras;
  
  struct CARParas{
    double rho;
    double E_mu_quad;
    arma::vec D;
    arma::sp_mat B;
    arma::sp_mat W;
    arma::sp_mat B_t;
    arma::sp_mat inv_Sigma;
    arma::vec neighbor_mean;
  } CAR;
  
  struct ELBOType{
    double overall;
    double loglik;
    double beta;
    double gamma;
    double sigma;
  } ELBO;
  
  struct ELBOTrace{
    arma::vec loglik;
    arma::vec beta;
    arma::vec gamma;
    arma::vec sigma;
  } ELBO_trace;
  
  
  struct VBProfile{
    arma::vec ELBO;
    arma::vec E_inv_sigma_sq;
    arma::mat E_gamma;
    arma::mat E_beta;
    arma::mat E_delta_rho;
    arma::mat E_mu;
    arma::mat E_f_mu;
    arma::mat E_IP;
    arma::vec E_inv_sigma_sq_beta;
    arma::vec E_inv_tau_sq_mu;
    arma::vec E_inv_sigma_sq_gamma;
  } vb_profile;

  struct MCMCsample{
    arma::mat beta;
    arma::mat gamma;
    arma::mat delta_rho;
    arma::mat f_mu;
    arma::mat IP;
    arma::vec sigma_sq;
    arma::vec sigma_sq_beta;
    arma::vec sigma_sq_gamma;
    arma::vec tau_sq_mu;
    arma::vec loglik;
  } paras_sample;

  struct MCMCcontrol{
    int total_iter;
    int burnin;
    int mcmc_sample;
    int thinning;
    int verbose;
    int save_profile;
    int total_profile;
    int begin_f_beta;
  } gibbs_control;
  
  
  struct VBControl{
    int max_iter;
    double para_diff_tol;
    int ELBO_stop;
    double ELBO_diff_tol;
    bool trace_all_ELBO;
    bool update_beta;
    bool update_gamma;
    bool update_delta_rho;
    bool update_tau_sq_mu;
    int verbose;
    int save_profile;
    int total_profile;
    int begin_f_mu;
  } vb_control;

  struct AnnealingControl{
    double a;
    double b;
    double gamma;
  }sigmasq_beta_control;

  struct SGDControl{
    double step;
    int subsample_size;
    arma::uvec subsample_idx;
    int Geweke_interval;
    double Geweke_pvalue;
    double protect_percent;
  }sgd_control;
  
  int iter;
  
public:
  
  List test_output;
  double elapsed_sgd = 0;
  bool display_progress = true;
  int method;
  bool include_Confounder = true;

  void set_method(CharacterVector in_method){
    if(in_method(0)=="CAVI"){
      std::cout << "Scalar on image: Coordinate Ascent Mean Field Variational Bayes" << std::endl;
      method = 1;
    } else if(in_method(0)=="SGD"){
      std::cout << "Scalar on image: Stochastic gradient descent on global variables VI" << std::endl;
      method = 2;
    }else if(in_method(0) == "Gibbs"){
      std::cout << "Scalar on image: Gibbs sampling" << std::endl;
      method = 0;
    }else{
      method = -1;
      Rcpp::stop( "Scalar on image: Method not implemented. Must be CAVI, SGD, or Gibbs" );
    }
  };
  
  
  void load_data(const arma::vec& in_y, const arma::mat& in_X, const  arma::mat& in_M, 
                 const double& in_lambda){
    Rcout<<"Scalar on image: Loading data...."<<std::endl;
    dat.y = in_y;
    dat.X = in_X;
    dat.M = in_M;
    dat.M_l2 = sum(dat.M%dat.M,0);
    dat.X_l2 = sum(dat.X%dat.X,0);
    dat.lambda = in_lambda;
    
    
    dat.n = dat.y.n_elem;
    dat.p = dat.M.n_cols;
    dat.m = dat.X.n_cols;
    
    if(include_Confounder){
      arma::mat XtX = dat.X.t()*dat.X;
      arma::mat eigvec; arma::vec eigval;
      eig_sym(eigval, eigvec, XtX);
      dat.XtX_eigenval = eigval;
      dat.XtX_eigenvec = eigvec;
      
    }
    
    // svd on m
    arma::svd_econ(svd_m.U,svd_m.D,svd_m.V,dat.M);
    svd_m.M_star = svd_m.U.t() * dat.M;
    svd_m.D_Vt = arma::diagmat(svd_m.D)*svd_m.V.t();
    svd_m.D2 = svd_m.D % svd_m.D;
    
    // svd on x
    arma::svd_econ(svd_x.U,svd_x.D,svd_x.V,dat.X);
    svd_x.D2 = svd_x.D % svd_x.D;
    
    Rcout<<"Scalar on image: Load data successful"<<std::endl;
  };
  
  void load_CAR(const double& in_rho, 
                const arma::sp_mat& in_B, 
                const arma::sp_mat& in_W, 
                const arma::sp_mat& in_Sigma_inv,
                const arma::vec& in_D){
    Rcout<<"Sparse mean: Loading CAR..."<<std::endl;
    CAR.rho = in_rho; 
    
    CAR.B = in_B;
    CAR.B_t = in_B.t();
    CAR.D = in_D;
    CAR.W = in_W;
    
    CAR.inv_Sigma = in_Sigma_inv;
    Rcout<<"Sparse mean: Load CAR successful"<<std::endl;
  }
  
  
  void set_paras_initial_values(const Rcpp::List& init,
                                const arma::vec& in_delta_rho,
                                const double& in_sigma_sq,
                                const double& in_sigma_sq_beta,
                                const double& in_sigma_sq_gamma,
                                const double& in_tau_mu_sq){
    vb_paras.E_delta_rho = in_delta_rho;
    vb_paras.E_delta_rho_prior = arma::ones(dat.p,1);
    vb_paras.E_delta_rho_prior *= 0.5;

    arma::vec beta = init["beta"];
    arma::vec f_mu = init["f_beta"];
    arma::vec gamma = init["gamma"];
    if(arma::max(arma::abs(beta - f_mu)) < 1e-3){
      Rcout<<"Warning: initial value for beta and f_beta cannot be too close."<<std::endl;
      Rcout<<" Automatically adding error N(0,1) to the initial value of beta."<<std::endl;
      beta = beta + arma::randn(dat.p,1);
    }

    arma::vec mu = f_mu + arma::randn( dat.p, arma::distr_param(0.0, 0.01) );
    if(method > 0){
      vb_paras.E_beta = beta;
      vb_paras.E_f_mu = f_mu;
      vb_paras.E_mu = mu;
      vb_paras.Var_beta = arma::ones(dat.p,1); // marginal variance
      vb_paras.E_gamma =  gamma;
      vb_paras.Var_gamma = arma::ones(dat.m, dat.m); // joint mat
      vb_paras.var_gamma_marg = arma::ones(dat.m,1);
      vb_paras.E_mu_neighbor_2moment = arma::ones(dat.p,1);
      vb_paras.E_mu2 = arma::ones(dat.p,1);
      vb_paras.Var_f_mu =  arma::ones(dat.p,1);
      vb_paras.E_f_mu2 = arma::ones(dat.p,1);
      vb_paras.E_inv_sigma_mu_sq = arma::ones(dat.p,1)*in_tau_mu_sq;
      
      vb_paras.mix_prob.zeros(dat.p,3);
      vb_paras.mix_prob.col(0).ones();
      
      vb_paras.E_neighbor_mean = CAR.rho * CAR.B*vb_paras.E_mu;
      vb_paras.E_neighbor_mean %= vb_paras.E_delta_rho;
      
      vb_paras.E_inv_sigma_sq = 1.0/in_sigma_sq;
      vb_paras.E_inv_sigma_sq_beta = 1.0/in_sigma_sq_beta;
      vb_paras.E_inv_sigma_sq_gamma = 1.0/in_sigma_sq_gamma;
      vb_paras.E_inv_tau_sq_mu = 1.0/in_tau_mu_sq;
      vb_paras.E_inv_a_beta = 1.0;
      vb_paras.E_inv_a = 1.0;
      vb_paras.E_inv_a_gamma = 1.0;
      vb_paras.E_SSE = 1;
      vb_paras.E_IP = arma::zeros(dat.p,1);
      ELBO.beta = 0;
      ELBO.gamma = 0;
      ELBO.overall = 0;
      ELBO.loglik = 0;
      ELBO.sigma = 0;
      vb_paras.ELBO = 0;

      arma::vec rho_W_mu = CAR.W*vb_paras.E_mu;
      rho_W_mu %= vb_paras.E_delta_rho*CAR.rho;
      double b_mid = arma::accu(vb_paras.E_mu%vb_paras.E_mu%CAR.D) - arma::dot(vb_paras.E_mu, rho_W_mu);
      b_mid += b_mid + dat.p;
      CAR.E_mu_quad = b_mid;
    }else{
      paras.beta = beta;
      paras.f_mu = f_mu;
      paras.gamma = gamma;
      paras.delta_rho = in_delta_rho;
      paras.delta_rho_prior = arma::ones(dat.p,1);
      paras.delta_rho_prior *= 0.5;
      paras.IP = arma::zeros(dat.p,1);
      paras.inv_sigma_sq = 1.0/in_sigma_sq;
      paras.inv_sigma_sq_beta = 1.0/in_sigma_sq_beta;
      paras.inv_sigma_sq_gamma = 1.0/in_sigma_sq_gamma;
      paras.inv_tau_sq_mu = 1.0/in_tau_mu_sq;
      paras.inv_a_beta = 1.0;
      paras.inv_a = 1.0;
      paras.inv_a_gamma = 1.0;
      paras.loglik = 0;
      paras.mix_prob.zeros(dat.p,3);
      paras.mix_prob.col(0).ones();

      paras.neighbor_mean = CAR.rho * CAR.B*paras.f_mu;
      paras.neighbor_mean %= paras.delta_rho;

      arma::vec rho_W_mu = CAR.W*mu;
      rho_W_mu %= paras.delta_rho*CAR.rho;
      double b_mid = arma::accu(mu%mu%CAR.D) - arma::dot(mu, rho_W_mu);
      b_mid += b_mid + dat.p;
      CAR.E_mu_quad = b_mid;

    }
    
    
    Rcout<<"Scalar on image: set initial values successful"<<std::endl;
  };
  
  
  void set_vb_control(int in_max_iter, 
                      double in_para_diff_tol, 
                      int in_ELBO_stop,
                      double in_ELBO_diff_tol,
                      bool trace_all_ELBO,
                      int in_verbose,
                      int in_save_profile,
                      int begin_f_mu,
                      bool update_beta,
                      bool update_gamma,
                      bool update_delta_rho){
    vb_control.update_beta = update_beta;
    vb_control.update_gamma = update_gamma;
    vb_control.begin_f_mu = begin_f_mu;
    vb_control.max_iter = in_max_iter;
    vb_control.para_diff_tol = in_para_diff_tol;
    vb_control.ELBO_stop = in_ELBO_stop;
    vb_control.ELBO_diff_tol = in_ELBO_diff_tol;
    vb_control.trace_all_ELBO = trace_all_ELBO;
    vb_control.verbose = in_verbose;
    vb_control.save_profile = in_save_profile;
    vb_control.update_delta_rho = update_delta_rho;
    if(vb_control.save_profile > 0){
      vb_control.total_profile = vb_control.max_iter/vb_control.save_profile;
    } else{
      vb_control.total_profile = 0;
    }
    
  };

  void set_sigmasq_beta_annealing_control(List& sigmasq_beta_controls){
    double a = sigmasq_beta_controls["a"];
    double b = sigmasq_beta_controls["b"];
    double gamma = sigmasq_beta_controls["gamma"];
    sigmasq_beta_control.a = a;
    sigmasq_beta_control.b = b;
    sigmasq_beta_control.gamma = gamma;
  };
  
  
  void update_E_SSE(){
    
    arma::vec E_res = dat.y - dat.X*vb_paras.E_gamma - dat.M*vb_paras.E_beta;
    
    
    // old efficient approach
    double Var_beta_M = arma::accu(svd_m.D2 / (vb_paras.E_inv_sigma_sq_beta + vb_paras.E_inv_sigma_sq*svd_m.D2));
    double Var_gamma_X = 0;
    if(include_Confounder){
      Var_gamma_X = arma::accu(svd_x.D2 /(vb_paras.E_inv_sigma_sq_gamma +  vb_paras.E_inv_sigma_sq*svd_x.D2));
    }
    
    if(include_Confounder){
      vb_paras.E_SSE = dot(E_res,E_res) + Var_beta_M + Var_gamma_X; // joint: works for CAVI
    }else{
      vb_paras.E_SSE = dot(E_res,E_res) + Var_beta_M;
    }
    
    
    // Rcout<<"vb_paras.E_SSE = "<<vb_paras.E_SSE<<std::endl;
  };
  
  void update_E_beta(){
    
    arma::vec res = dat.y - dat.X*vb_paras.E_gamma;
    
    arma::mat Y = svd_m.U.t()*res - svd_m.M_star*vb_paras.E_f_mu;
    double tau2 = vb_paras.E_inv_sigma_sq/vb_paras.E_inv_sigma_sq_beta;
    vb_paras.E_beta = svd_m.V*arma::diagmat(svd_m.D/(1/tau2 + svd_m.D2))*Y;
    vb_paras.E_beta += vb_paras.E_f_mu;
    vb_paras.Var_beta = 1.0/(vb_paras.E_inv_sigma_sq*dat.M_l2.t() + vb_paras.E_inv_sigma_sq_beta);
    
    
    // update E_inv_sigma_sq_beta
    
    arma::vec bias = vb_paras.E_beta - vb_paras.E_f_mu;
    vb_paras.E_SSE_beta = accu(vb_paras.Var_beta + bias%bias);
    double b_q_sigma_beta = 0.5*vb_paras.E_SSE_beta*vb_paras.E_inv_sigma_sq + vb_paras.E_inv_a_beta;
    // b_q_sigma_beta += CAR.E_mu_quad*vb_paras.E_inv_tau_sq_mu*0.5; // include tau_mu term
    
    
    // link to sigma_Y
    // double E_inv_tau2_beta = (dat.p + 1.0)/2 / b_q_sigma_beta;
    // vb_paras.E_inv_sigma_sq_beta = E_inv_tau2_beta * vb_paras.E_inv_sigma_sq;

    vb_paras.E_inv_sigma_sq_beta = 1.0/(sigmasq_beta_control.a*pow(sigmasq_beta_control.b+iter,sigmasq_beta_control.gamma));
    
    
    // update E_inv_a_beta
    // vb_paras.E_inv_a_beta = 1/(1+E_inv_tau2_beta);
    
    // update ELBO
    // ELBO.beta = -0.5*vb_paras.E_inv_sigma_sq_beta*vb_paras.E_SSE_beta;
    // ELBO.beta += -(dat.p+1)/2*log(b_q_sigma_beta) + 0.5*accu(log(vb_paras.Var_beta));
    // ELBO.beta += -b_q_sigma_beta*vb_paras.E_inv_sigma_sq_beta - log(1+vb_paras.E_inv_sigma_sq_beta);
    
    
    
  };
  
  void update_E_mu(){
    
    // double E_inv_tau_sq_beta = vb_paras.E_inv_sigma_sq_beta / vb_paras.E_inv_sigma_sq;
    // double E_inv_sigma_sq_mu = E_inv_tau_sq_beta*vb_paras.E_inv_tau_sq_mu;
    double E_inv_sigma_sq_mu = vb_paras.E_inv_tau_sq_mu;
    
    // arma::vec D_inv_sig_mu2 = vb_paras.E_inv_sigma_mu_sq % CAR.D;
    
    arma::vec Vi = 1./(vb_paras.E_inv_sigma_sq_beta + E_inv_sigma_sq_mu*CAR.D);
    // arma::vec Vi = 1./(vb_paras.E_inv_sigma_sq_beta + D_inv_sig_mu2);
    
    arma::vec Vi_sqrt = sqrt(Vi);
    // arma::vec V0 = 1./D_inv_sig_mu2;
    arma::vec V0 = 1./(E_inv_sigma_sq_mu*CAR.D);
    arma::vec V0_inv = 1/V0;
    arma::vec V0_sqrt = sqrt(V0);
    
    arma::vec y_plus_lambda = vb_paras.E_beta + dat.lambda;
    arma::vec y_plus_lambda2 = y_plus_lambda%y_plus_lambda;
    arma::vec y_minus_lambda = vb_paras.E_beta - dat.lambda;
    arma::vec y_minus_lambda2 = y_minus_lambda%y_minus_lambda;
    
    arma::vec y2 = vb_paras.E_beta%vb_paras.E_beta;
    
    arma::vec mu_pos = Vi % (vb_paras.E_inv_sigma_sq_beta*(y_plus_lambda) + 
      V0_inv%vb_paras.E_neighbor_mean);
    arma::vec mu_neg = Vi % (vb_paras.E_inv_sigma_sq_beta*(y_minus_lambda) + 
      V0_inv%vb_paras.E_neighbor_mean);
    arma::vec mu_zero = vb_paras.E_neighbor_mean;
    
    // tn_mean, tn_var, tn_2moment, Z
    arma::mat pos_stats = truncated_normal_stats_vec(mu_pos, Vi_sqrt, dat.lambda, 1, 4);
    arma::mat neg_stats = truncated_normal_stats_vec(mu_neg,  Vi_sqrt, dat.lambda, -1, 4);
    arma::mat zero_stats = truncated_normal_stats_vec(mu_zero, V0_sqrt, dat.lambda,0,4);
    
    // get mixing prob
    arma::vec C_pos = V0_inv%vb_paras.E_mu_neighbor_2moment;
    arma::vec C_neg = C_pos; arma::vec C_zero = C_pos;
    
    C_pos += vb_paras.E_inv_sigma_sq_beta*y_plus_lambda2;
    C_neg += vb_paras.E_inv_sigma_sq_beta*y_minus_lambda2;
    C_zero += vb_paras.E_inv_sigma_sq_beta*y2;
    
    arma::vec log_Z_pos = -0.5*C_pos + 0.5*mu_pos%mu_pos/Vi + 0.5*log(Vi) + log(pos_stats.col(3));
    arma::vec log_Z_neg = -0.5*C_neg + 0.5*mu_neg%mu_neg/Vi + 0.5*log(Vi) + log(neg_stats.col(3));
    arma::vec log_Z_zero = -0.5*C_zero + 0.5*mu_zero%mu_zero/V0 + 0.5*log(V0) + log(zero_stats.col(3));
    
    
    
    vb_paras.mix_prob = get_mix_prob(log_Z_neg, log_Z_zero, log_Z_pos);
    // Rcout<<"iter = "<<iter<<std::endl;
    // Rcout<<"vb_paras.mix_prob.row(229) = "<<vb_paras.mix_prob.row(229)<<iter<<std::endl;
    
    
    arma::mat E_beta_mat = vb_paras.mix_prob % 
      join_horiz(neg_stats.col(0),zero_stats.col(0),pos_stats.col(0));
    E_beta_mat.elem(find(vb_paras.mix_prob==0)).zeros(); 
    vb_paras.E_mu = sum(E_beta_mat,1);
    
    E_beta_mat = vb_paras.mix_prob % 
      join_horiz(neg_stats.col(2),zero_stats.col(2),pos_stats.col(2));
    E_beta_mat.elem(find(vb_paras.mix_prob==0)).zeros();
    vb_paras.E_mu2 = sum(E_beta_mat,1);  
    
    E_beta_mat = vb_paras.mix_prob % 
      join_horiz(neg_stats.col(0)+dat.lambda,
                 arma::zeros(dat.p,1),
                 pos_stats.col(0)-dat.lambda);
    E_beta_mat.elem(find(vb_paras.mix_prob==0)).zeros();
    vb_paras.E_f_mu = sum(E_beta_mat,1);  
    
    
    if(vb_paras.E_beta.has_nan()){
      Rcout<<"Error!!! E_beta has nan"<<std::endl;
      test_output = List::create(Named("zero_stats")=zero_stats,
                                 Named("pos_stats")=pos_stats,
                                 Named("neg_stats")=neg_stats,
                                 Named("mix_prob")=vb_paras.mix_prob,
                                 Named("mu_pos")= mu_pos,
                                 Named("Vi_sqrt")= Vi_sqrt);
    }
    if(vb_paras.E_mu2.has_nan()){
      Rcout<<"Error!!! E_beta2 has nan"<<std::endl;
    }
    if(vb_paras.E_f_mu.has_nan()){
      Rcout<<"Error!!! E_f_beta has nan"<<std::endl;
    }
    
    // Rcout<<"..............4"<<std::endl;
    arma::vec bias_neg = neg_stats.col(0)+dat.lambda;
    arma::vec bias_pos = pos_stats.col(0)-dat.lambda;
    
    vb_paras.E_f_mu2 = sum(vb_paras.mix_prob % 
      join_horiz(neg_stats.col(1) + bias_neg%bias_neg,
                 arma::zeros(dat.p,1),
                 pos_stats.col(1) + bias_pos%bias_pos),1);
    
    vb_paras.Var_f_mu = vb_paras.E_f_mu2 - 
      vb_paras.E_f_mu % vb_paras.E_f_mu;
    
    
    // update other summary stats
    vb_paras.E_neighbor_mean = CAR.rho*CAR.B*vb_paras.E_mu;
    vb_paras.E_neighbor_mean %= vb_paras.E_delta_rho;
    
    arma::mat Cov_q_full = vb_paras.E_beta * vb_paras.E_beta.t();
    Cov_q_full.diag() = vb_paras.E_mu2;
    vb_paras.E_mu_neighbor_2moment = CAR.rho*CAR.rho*diagvec(CAR.B*Cov_q_full*CAR.B_t);
    vb_paras.E_mu_neighbor_2moment %= vb_paras.E_delta_rho;
    
    // update E_inv_tau_sq_mu
    
    arma::vec rho_W_mu = CAR.W*vb_paras.E_mu;
    rho_W_mu %= vb_paras.E_delta_rho*CAR.rho;
    double b_mid = arma::accu(vb_paras.E_mu%vb_paras.E_mu%CAR.D) - arma::dot(vb_paras.E_mu, rho_W_mu);
    b_mid += dat.p;
    CAR.E_mu_quad = b_mid;
    // b_mid = b_mid* E_inv_tau_sq_beta;
    
    // fixed at 1 for better performance
    if(method == 2){
      vb_paras.E_inv_tau_sq_mu = 0.5*(dat.p+1)/(0.5*b_mid + 1/2); // need to change sigma_beta accordingly
    }
    
    // update IP
    vb_paras.E_IP = 1 - vb_paras.mix_prob.col(1);
    
    // update delta_rho
    // E_inv_sigma_sq_mu = E_inv_tau_sq_beta*vb_paras.E_inv_tau_sq_mu;
    
    E_inv_sigma_sq_mu = vb_paras.E_inv_tau_sq_mu;
    
    if(vb_control.update_delta_rho){
      arma::vec Var_mu = vb_paras.E_mu2 - vb_paras.E_mu % vb_paras.E_mu;
      arma::vec bias = vb_paras.E_mu - vb_paras.E_neighbor_mean;
      arma::vec mse = bias%bias + Var_mu;
      arma::vec log_L1 = -0.5*(E_inv_sigma_sq_mu*CAR.D) % mse + log(vb_paras.E_delta_rho_prior);
      arma::vec log_L0 = -0.5*(E_inv_sigma_sq_mu*CAR.D) % vb_paras.E_mu2 + log(1-vb_paras.E_delta_rho_prior);
      arma::vec log_Z = log_L1 - log_L0;
      vb_paras.E_delta_rho = 1/(1+exp(-log_Z));
    }
    
    
  };
  
  
  
  
  
  void update_E_gamma(){
    arma::vec res = dat.y - dat.M*vb_paras.E_beta;
    
    arma::vec Y = svd_x.U.t()*res;
    arma::vec var_gamma = 1.0/(vb_paras.E_inv_sigma_sq_gamma + vb_paras.E_inv_sigma_sq*svd_x.D2);
    arma::vec E_gamma_star = var_gamma%(vb_paras.E_inv_sigma_sq*svd_x.D%Y);
    vb_paras.E_gamma = svd_x.V*E_gamma_star;
    
    vb_paras.E_SSE_gamma = arma::accu(var_gamma + E_gamma_star%E_gamma_star);
    
    
    
    // link to sigma_Y
    // double E_inv_tau2_gamma =  0.5*(dat.m+1) /( vb_paras.E_inv_a_gamma + vb_paras.E_SSE_gamma);
    // vb_paras.E_inv_sigma_sq_gamma = E_inv_tau2_gamma*vb_paras.E_inv_sigma_sq;
    
    vb_paras.E_inv_sigma_sq_gamma =  0.5*(dat.m+1) /( vb_paras.E_inv_a_gamma + vb_paras.E_SSE_gamma);
    // vb_paras.E_inv_sigma_sq_gamma = E_inv_tau2_gamma*vb_paras.E_inv_sigma_sq;
    
    vb_paras.E_inv_a_gamma = 1/(1+vb_paras.E_inv_sigma_sq_gamma);
    
    
    // Rcout<<"vb_paras.E_inv_sigma_sq_gamma = "<<vb_paras.E_inv_sigma_sq_gamma<<std::endl;
    // Rcout<<"vb_paras.E_SSE_gamma = "<<vb_paras.E_SSE_gamma<<std::endl;
    
    
    
    
    // // update elbo - gamma
    // ELBO.gamma = -0.5*vb_paras.E_inv_sigma_sq_gamma*E_gamma_l2 - 0.5*(dat.m+1)*log(b_sigma_gamma) +
    //   0.5*accu(log(Sigma.diag()));
    // ELBO.gamma += b_sigma_gamma*vb_paras.E_inv_sigma_sq_gamma - log(1+vb_paras.E_inv_sigma_sq_gamma);
    
  }
  
  
  void update_E_inv_sigma_sq(){
    double b_sigma = 0.5*vb_paras.E_SSE + vb_paras.E_inv_a ; 
    // double inv_tau_beta2 = vb_paras.E_inv_sigma_sq_beta / vb_paras.E_inv_sigma_sq;
    // double inv_tau_gamma2 = vb_paras.E_inv_sigma_sq_gamma / vb_paras.E_inv_sigma_sq;
    // b_sigma += 0.5* inv_tau_beta2 * vb_paras.E_SSE_beta;
    // b_sigma += 0.5* inv_tau_gamma2 * vb_paras.E_SSE_gamma;
    // vb_paras.E_inv_sigma_sq = 0.5 * (dat.n + dat.p + dat.m + 1.0)/b_sigma;
    // vb_paras.E_inv_sigma_sq = 0.5 * (dat.n + dat.m + 1.0)/b_sigma;
    vb_paras.E_inv_sigma_sq = 0.5 * (dat.n + 1.0)/b_sigma;
    
    // Rcout<<"update_E_inv_sigma_sq ----  test"<<std::endl;
    // Rcout<<"b_sigma = "<<b_sigma<<std::endl;
    
    vb_paras.E_inv_a = 1/(1+vb_paras.E_inv_sigma_sq);
    
    // update elbo sigma
    // ELBO.sigma = (dat.n)/2*b_sigma - (dat.n+1)/2*log(b_sigma) + 
    //   b_sigma*vb_paras.E_inv_sigma_sq - log(1+vb_paras.E_inv_sigma_sq);
  };
  
  void update_ELBO(){
    ELBO.loglik = dat.n/2*log(vb_paras.E_inv_sigma_sq) - 0.5*vb_paras.E_inv_sigma_sq*vb_paras.E_SSE;
    ELBO.overall = ELBO.loglik + ELBO.beta + ELBO.gamma + ELBO.sigma;
    vb_paras.ELBO = ELBO.overall;
    
  };
  
  double compute_paras_diff(arma::vec& beta, arma::vec& beta_prev){
    arma::vec temp = beta - beta_prev;
    return accu(temp%temp)/beta.n_elem;
  };
  
  void initialize_vb_profile(){
    if(vb_control.save_profile>0){
      vb_profile.ELBO.zeros(vb_control.total_profile);
      vb_profile.E_inv_sigma_sq.zeros(vb_control.total_profile);
      vb_profile.E_inv_sigma_sq_beta.zeros(vb_control.total_profile);
      vb_profile.E_inv_sigma_sq_gamma.zeros(vb_control.total_profile);
      vb_profile.E_inv_tau_sq_mu.zeros(vb_control.total_profile);
      vb_profile.E_gamma.zeros(dat.m,vb_control.total_profile);
      vb_profile.E_beta.zeros(dat.p, vb_control.total_profile);
      vb_profile.E_delta_rho.zeros(dat.p, vb_control.total_profile);
      vb_profile.E_f_mu.zeros(dat.p, vb_control.total_profile);
      vb_profile.E_IP.zeros(dat.p, vb_control.total_profile);
    }
    if(vb_control.trace_all_ELBO){
      ELBO_trace.loglik.zeros(vb_control.max_iter);
      ELBO_trace.beta.zeros(vb_control.max_iter);
      ELBO_trace.gamma.zeros(vb_control.max_iter);
      ELBO_trace.sigma.zeros(vb_control.max_iter);
    }
  }
  
  void save_vb_profile(){
    if(vb_control.save_profile > 0){
      if(iter%vb_control.save_profile==0){
        int profile_iter = iter/vb_control.save_profile;
        if(vb_control.ELBO_stop==0){
          update_ELBO();
        }
        vb_profile.ELBO(profile_iter) = ELBO.overall;
        vb_profile.E_inv_sigma_sq(profile_iter) = vb_paras.E_inv_sigma_sq;
        
        vb_profile.E_gamma.col(profile_iter) = vb_paras.E_gamma;
        vb_profile.E_inv_sigma_sq_beta(profile_iter) = vb_paras.E_inv_sigma_sq_beta;
        vb_profile.E_inv_sigma_sq_gamma(profile_iter) = vb_paras.E_inv_sigma_sq_gamma;
        vb_profile.E_inv_tau_sq_mu(profile_iter) = vb_paras.E_inv_tau_sq_mu;
        vb_profile.E_beta.col(profile_iter) = vb_paras.E_beta;
        vb_profile.E_delta_rho.col(profile_iter) = vb_paras.E_delta_rho;
        vb_profile.E_f_mu.col(profile_iter) = vb_paras.E_f_mu;
        vb_profile.E_IP.col(profile_iter) = vb_paras.E_IP;
      }
    }
    
    if(vb_control.trace_all_ELBO){
      ELBO_trace.loglik(iter) = ELBO.loglik;
      ELBO_trace.beta(iter) = ELBO.beta;
      ELBO_trace.gamma(iter) = ELBO.gamma;
      ELBO_trace.sigma(iter) = ELBO.sigma;
    }
  }
  
  void monitor_vb(){
    if(vb_control.verbose > 0){
      if((iter%vb_control.verbose==0) & (!display_progress)){
        if(vb_control.ELBO_stop==0){
          update_ELBO();
        }
        // std::cout << "Scalar on image: iter: " << iter <<  " ELBO: "<< ELBO.overall << std::endl;
        std::cout << "Scalar on image: iter: " << iter <<  " sigma_sq: "<< 1.0/vb_paras.E_inv_sigma_sq;
        std::cout <<" mu_quad = "<<CAR.E_mu_quad<< std::endl;
      }
    }
  }
  
  void run_CAVI(int f_beta_interval){
    initialize_vb_profile();
    std::cout << "Scalar on image: running CAVI " <<std::endl;
    // Progress prog(vb_control.max_iter, display_progress);
    for(iter=0; iter<vb_control.max_iter; iter++){
      // prog.increment();
      arma::vec E_beta_prev = vb_paras.E_beta;
      if((iter % f_beta_interval == 0) & (iter>vb_control.begin_f_mu)){
        update_E_mu();
      }
      
      if(vb_control.update_beta){
        update_E_beta();
      }
      
      if(vb_control.update_gamma && include_Confounder){
        update_E_gamma();
      }
      
      update_E_SSE();
      
      update_E_inv_sigma_sq();
      
      double ELBO_prev = ELBO.overall;
      update_ELBO();
      save_vb_profile();
      monitor_vb();
      if(vb_control.ELBO_stop == 0){
        if(compute_paras_diff(vb_paras.E_beta,E_beta_prev) < vb_control.para_diff_tol){
          Rcout<<"Scalar on image: CAVI converged at iter: "<< iter<<"; paras_diff="<<compute_paras_diff(vb_paras.E_beta,E_beta_prev) << std::endl;
          save_vb_profile();
          monitor_vb();
          break;
        }
      } else {
        double ELBO_overall = ELBO.overall;
        if(std::abs(ELBO_overall - ELBO_prev) < vb_control.ELBO_diff_tol){
          save_vb_profile();
          monitor_vb();
          break;
        }
      }
      
      
    }
    
  };
  
  
  
  List get_vb_post_mean(){
    return List::create(Named("beta") = vb_paras.E_beta,
                        Named("IP") = vb_paras.E_IP,
                        Named("f_mu") = vb_paras.E_f_mu,
                        Named("Var_beta") = vb_paras.Var_beta,
                        Named("gamma") = vb_paras.E_gamma,
                        Named("sigma_sq") = 1/vb_paras.E_inv_sigma_sq,
                        Named("sigma_beta_sq") = 1/vb_paras.E_inv_sigma_sq_beta,
                        Named("sigma_gamma_sq") = 1/vb_paras.E_inv_sigma_sq_gamma);
  };
  
  List get_vb_trace(){
    int actual_profile_iter = 1;
    if(iter == 0){
      iter = 1;
    }
    if(vb_control.save_profile>0){
      actual_profile_iter = iter/vb_control.save_profile;
    }
    arma::uvec iters = arma::linspace<arma::uvec>(1,iter,actual_profile_iter);
    List all_ELBO;
    if(vb_control.trace_all_ELBO){
      all_ELBO = List::create(Named("loglik") = ELBO_trace.loglik,
                              Named("beta") = ELBO_trace.beta,
                              Named("gamma") = ELBO_trace.gamma,
                              Named("sigma") = ELBO_trace.sigma);
    }else{
      all_ELBO = List::create(Named("trace_all_ELBO") = vb_control.trace_all_ELBO);
    }
    
    return List::create(Named("iters") = iters,
                        Named("ELBO") = vb_profile.ELBO.rows(0,actual_profile_iter-1),
                        Named("all_ELBO") = all_ELBO,
                        Named("E_sigma_sq") = 1/vb_profile.E_inv_sigma_sq.rows(0,actual_profile_iter-1),
                        Named("E_gamma") = vb_profile.E_gamma.cols(0,actual_profile_iter-1),
                        Named("E_sigma_sq_beta") = 1/vb_profile.E_inv_sigma_sq_beta.rows(0,actual_profile_iter-1),
                        Named("E_tau_sq_mu") = 1/vb_profile.E_inv_tau_sq_mu.rows(0,actual_profile_iter-1),
                        Named("E_sigma_sq_gamma") = 1/vb_profile.E_inv_sigma_sq_gamma.rows(0,actual_profile_iter-1),
                        Named("E_beta") = vb_profile.E_beta.cols(0,actual_profile_iter-1),
                        Named("E_delta_rho") = vb_profile.E_delta_rho.cols(0,actual_profile_iter-1),
                        Named("E_f_mu") = vb_profile.E_f_mu.cols(0,actual_profile_iter-1),
                        Named("E_IP") = vb_profile.E_IP.cols(0,actual_profile_iter-1));
  }
  
  List get_test_output(){
    return List::create(Named("Scalar_on_Image") = test_output);
  }
  
  List get_vb_control(){
    return List::create(Named("max_iter")= vb_control.max_iter,
                        Named("para_diff_tol") = vb_control.para_diff_tol,
                        Named("ELBO_stop") = vb_control.ELBO_stop,
                        Named("ELBO_diff_tol") = vb_control.ELBO_diff_tol,
                        Named("verbose") = vb_control.verbose,
                        Named("save_profile") = vb_control.save_profile,
                        Named("total_profile") = vb_control.total_profile);
  };
  
  int get_iter(){
    return iter;
  };


  void set_SGD_controls(List SGD_controls,
                        List Geweke_controls){
    double step = SGD_controls["step"];
    sgd_control.step = step;
    int subsample_size = SGD_controls["subsample_size"];
    sgd_control.subsample_size = subsample_size;
    double protect_percent = SGD_controls["protect_percent"];
    sgd_control.protect_percent = protect_percent;

    int Geweke_interval = Geweke_controls["interval"];
    double Geweke_pvalue = Geweke_controls["pvalue"];
    sgd_control.Geweke_interval = Geweke_interval;
    sgd_control.Geweke_pvalue = Geweke_pvalue;
  };

  
  void SGD_initialize_subsample(const arma::uvec& sub_sample){
    sub_dat.n = sub_sample.n_elem;
    sub_dat.y = dat.y(sub_sample);
    sub_dat.M = dat.M.rows(sub_sample);
    sub_dat.X = dat.X.rows(sub_sample);
    
  }

  void SGD_update_E_beta(){
    // update E_beta
    arma::vec res_sub = sub_dat.y - sub_dat.X*vb_paras.E_gamma;
    arma::vec E_jM_j = sub_dat.M*vb_paras.E_beta;
    arma::vec temp = E_jM_j - res_sub;
    
    vb_paras.Var_beta = trans(1/(vb_paras.E_inv_sigma_sq * dat.M_l2 + 
      vb_paras.E_inv_sigma_sq_beta));
    arma::vec grad_E_beta = - vb_paras.E_inv_sigma_sq*trans(sub_dat.M)*temp;
    arma::vec post_grad = grad_E_beta;
    arma::vec prior_grad = -vb_paras.E_inv_sigma_sq_beta*(vb_paras.E_beta - 
      vb_paras.E_f_mu);
    // // for testing only
    // sgd_control.step = 1.0/vb_paras.E_inv_sigma_sq_beta;
    vb_paras.E_beta += sgd_control.step * (dat.n/sub_dat.n*post_grad + prior_grad);
    // try to add injected noise, langevin dynamic
    vb_paras.E_beta += sqrt(sgd_control.step*2)*arma::randn<arma::vec>(dat.p);
    
    arma::vec grad = dat.n/sub_dat.n*post_grad + prior_grad;
    
    if(iter%vb_control.verbose==0){
      Rcout<<"iter = "<<iter<<"min(grad) = "<<min(grad)
            <<"; max(grad) = "<<max(grad)<<"; step = "<<sgd_control.step<<std::endl;
    }
    
    // annealing sigma_beta
    vb_paras.E_inv_sigma_sq_beta = 1.0/(sigmasq_beta_control.a*pow(sigmasq_beta_control.b+iter,sigmasq_beta_control.gamma));
    
    // arma::vec bias = vb_paras.E_beta - vb_paras.E_f_mu;
    // double E_SSE_beta = accu(vb_paras.Var_beta + bias%bias);
    // double b_q_sigma_beta = 0.5*E_SSE_beta + vb_paras.E_inv_a_beta;
    
    // // update E_inv_a_beta
    // vb_paras.E_inv_a_beta = 1/(1+vb_paras.E_inv_sigma_sq_beta);

    // // get ELBO
    // ELBO.beta = -0.5*vb_paras.E_inv_sigma_sq_beta*E_SSE_beta;
    // ELBO.beta += -(dat.p+1)/2*log(b_q_sigma_beta) + 0.5*accu(log(vb_paras.Var_beta));
    // ELBO.beta += -b_q_sigma_beta - log(1+vb_paras.E_inv_sigma_sq_beta);
    
  }


  void run_SGD(int f_beta_interval){
    arma::wall_clock timer;
    
    initialize_vb_profile();
    std::cout << "Scalar on image: running SGD " <<std::endl;

    arma::vec Geweke_SSE = arma::zeros(sgd_control.Geweke_interval);
    
    // Progress prog(vb_control.max_iter, display_progress);
    for(iter=0; iter<vb_control.max_iter; iter++){
      // prog.increment();
      arma::vec E_beta_prev = vb_paras.E_beta;
      if((iter % f_beta_interval == 0) & (iter>vb_control.begin_f_mu)){
        update_E_mu();
      }
      sgd_control.subsample_idx = arma::randperm(dat.n, sgd_control.subsample_size);
      SGD_initialize_subsample(sgd_control.subsample_idx);
      SGD_update_E_beta();// Use SGD to update all locations on a subsample
      
      if(vb_control.update_gamma && include_Confounder){
        update_E_gamma();
      }
      
      update_E_SSE();
      update_E_inv_sigma_sq();
      
      update_ELBO();
      save_vb_profile();
      monitor_vb();
      
      // replace this by Geweke test on SSE
      if(iter > sgd_control.Geweke_interval-1){
        Geweke_SSE = arma::shift(Geweke_SSE,-1);
        Geweke_SSE(sgd_control.Geweke_interval-1) = vb_paras.E_SSE;
        bool Geweke_fail = geweke_test(Geweke_SSE, sgd_control.Geweke_pvalue, 0.1,0.5);
        if(!Geweke_fail && iter> sgd_control.protect_percent*vb_control.max_iter){
          break;
        }
      }else{
        Geweke_SSE(iter) = vb_paras.E_SSE;
      }
      
    }// end of for-loop
    
  }

  // begin gibbs functions
  void set_gibbs_control(int in_mcmc_sample, int in_burnin, int in_thinning, 
                         int in_verbose, int in_save_profile, int in_begin_f_beta){
    gibbs_control.mcmc_sample = in_mcmc_sample;
    gibbs_control.burnin = in_burnin;
    gibbs_control.thinning = in_thinning;
    gibbs_control.total_iter = gibbs_control.burnin;
    gibbs_control.total_iter += gibbs_control.mcmc_sample*gibbs_control.thinning; 
    gibbs_control.verbose = in_verbose;
    gibbs_control.save_profile = in_save_profile;
    gibbs_control.begin_f_beta = in_begin_f_beta;
    if(gibbs_control.save_profile > 0){
      gibbs_control.total_profile = gibbs_control.total_iter/gibbs_control.save_profile;
    } else{
      gibbs_control.total_profile = 0;
    }
    Rcout<<"Scalar on image: set gibbs control successful"<<std::endl;
  };

  void initialize_gibbs_sample(){
    paras_sample.beta.zeros(paras.beta.n_elem,gibbs_control.mcmc_sample);
    paras_sample.gamma.zeros(paras.gamma.n_elem,gibbs_control.mcmc_sample);
    paras_sample.delta_rho.zeros(paras.delta_rho.n_elem,gibbs_control.mcmc_sample);
    paras_sample.f_mu.zeros(paras.f_mu.n_elem,gibbs_control.mcmc_sample);
    paras_sample.IP.zeros(paras.IP.n_elem,gibbs_control.mcmc_sample);
    paras_sample.sigma_sq.zeros(gibbs_control.mcmc_sample);
    paras_sample.sigma_sq_beta.zeros(gibbs_control.mcmc_sample);
    paras_sample.sigma_sq_gamma.zeros(gibbs_control.mcmc_sample);
    paras_sample.tau_sq_mu.zeros(gibbs_control.mcmc_sample);

    paras_sample.loglik.zeros(gibbs_control.total_iter);
  };

  void update_beta(){
    arma::vec res = dat.y - dat.X*paras.gamma;
    arma::mat Y = svd_m.U.t()*res - svd_m.M_star*paras.f_mu;
    double sigma_beta = 1.0/sqrt(paras.inv_sigma_sq_beta);
    double sigma_Y = 1.0/sqrt(paras.inv_sigma_sq);
    double tau = sigma_beta/sigma_Y, tau2 = tau*tau;
    arma::vec alpha1 = arma::randn(dat.p) * sigma_beta;
    arma::vec alpha2 = arma::randn(dat.n) * sigma_Y;
    
    paras.beta = alpha1 + tau2 * svd_m.V*diagmat(svd_m.D/(1+tau2*svd_m.D%svd_m.D)) *
      (Y - svd_m.D_Vt*alpha1-alpha2) + paras.f_mu;

    // annealing on sigma_beta
    paras.inv_sigma_sq_beta = 1.0/(sigmasq_beta_control.a*pow(sigmasq_beta_control.b+iter,sigmasq_beta_control.gamma));
  };
  void update_mu(){
    double inv_sigma_sq_mu = paras.inv_tau_sq_mu;
    
    arma::vec Vi = 1./(paras.inv_sigma_sq_beta + inv_sigma_sq_mu*CAR.D);
    
    arma::vec Vi_sqrt = sqrt(Vi);
    arma::vec V0 = 1./(inv_sigma_sq_mu*CAR.D);
    arma::vec V0_inv = 1/V0;
    arma::vec V0_sqrt = sqrt(V0);
    
    arma::vec y_plus_lambda = paras.beta + dat.lambda;
    arma::vec y_plus_lambda2 = y_plus_lambda%y_plus_lambda;
    arma::vec y_minus_lambda = paras.beta - dat.lambda;
    arma::vec y_minus_lambda2 = y_minus_lambda%y_minus_lambda;
    
    arma::vec y2 = paras.beta%paras.beta;
    
    arma::vec mu_pos = Vi % (paras.inv_sigma_sq_beta*(y_plus_lambda) + 
      V0_inv%paras.neighbor_mean);
    arma::vec mu_neg = Vi % (paras.inv_sigma_sq_beta*(y_minus_lambda) + 
      V0_inv%paras.neighbor_mean);
    arma::vec mu_zero = paras.neighbor_mean;

    
    // tn_mean, tn_var, tn_2moment, Z
    arma::mat pos_stats = truncated_normal_stats_vec(mu_pos, Vi_sqrt, dat.lambda, 1, 4);
    arma::mat neg_stats = truncated_normal_stats_vec(mu_neg,  Vi_sqrt, dat.lambda, -1, 4);
    arma::mat zero_stats = truncated_normal_stats_vec(mu_zero, V0_sqrt, dat.lambda,0,4);
    
    // get mixing prob
    arma::vec C_pos = V0_inv%paras.neighbor_mean%paras.neighbor_mean;
    arma::vec C_neg = C_pos; arma::vec C_zero = C_pos;
    
    C_pos += paras.inv_sigma_sq_beta*y_plus_lambda2;
    C_neg += paras.inv_sigma_sq_beta*y_minus_lambda2;
    C_zero += paras.inv_sigma_sq_beta*y2;

    
    arma::vec log_Z_pos = -0.5*C_pos + 0.5*mu_pos%mu_pos/Vi + 0.5*log(Vi) + log(pos_stats.col(3));
    arma::vec log_Z_neg = -0.5*C_neg + 0.5*mu_neg%mu_neg/Vi + 0.5*log(Vi) + log(neg_stats.col(3));
    arma::vec log_Z_zero = -0.5*C_zero + 0.5*mu_zero%mu_zero/V0 + 0.5*log(V0) + log(zero_stats.col(3));

    arma::mat mix_prob = get_mix_prob(log_Z_neg, log_Z_zero, log_Z_pos);
    
    
    arma::vec u = arma::randu(mix_prob.n_rows,1);
    arma::uvec neg_idx = arma::find(u<mix_prob.col(0));
    arma::uvec zero_idx = arma::find(u>=mix_prob.col(0) && u<mix_prob.col(0)+mix_prob.col(1));
    arma::uvec pos_idx = arma::find(u>=mix_prob.col(0)+mix_prob.col(1));
    

    arma::vec mu_sample = arma::zeros(mix_prob.n_rows);
    if(neg_idx.n_elem>0){
      mu_sample(neg_idx) = sample_truncated_normal_vec(mu_neg(neg_idx), Vi_sqrt(neg_idx),dat.lambda, -1);
      paras.f_mu(neg_idx) = mu_sample(neg_idx) + dat.lambda;
    }
    if(zero_idx.n_elem>0){
      mu_sample(zero_idx) = sample_truncated_normal_vec(mu_zero(zero_idx), V0_sqrt(zero_idx), dat.lambda, 0);
      paras.f_mu(zero_idx).zeros();
    }
    if(pos_idx.n_elem>0){
      mu_sample(pos_idx) = sample_truncated_normal_vec(mu_pos(pos_idx), Vi_sqrt(pos_idx), dat.lambda, 1);
      paras.f_mu(pos_idx) = mu_sample(pos_idx) - dat.lambda;
    }
    // update IP
    paras.IP = 1 - mix_prob.col(1);

    // update neighbor_mean
    paras.neighbor_mean = CAR.rho * CAR.B* mu_sample;
    paras.neighbor_mean %= paras.delta_rho;

    // update tau_sq_mu
    arma::vec rho_W_mu = CAR.W*mu_sample;
    rho_W_mu %= paras.delta_rho*CAR.rho;
    double b_mid = arma::accu(mu_sample%mu_sample%CAR.D) - arma::dot(mu_sample, rho_W_mu);
    b_mid += dat.p;
    CAR.E_mu_quad = b_mid;
    paras.inv_tau_sq_mu = arma::randg(arma::distr_param(0.5*(dat.p+1),1.0/(0.5*b_mid + 1/2)));
    
    // update delta_rho
    arma::vec diff = mu_sample - paras.neighbor_mean;
    arma::vec mse = diff%diff;
    arma::vec log_L1 = -0.5*(paras.inv_tau_sq_mu*CAR.D) % mse + log(paras.delta_rho_prior);
    arma::vec log_L0 = -0.5*(paras.inv_tau_sq_mu*CAR.D) % mu_sample%mu_sample + log(1-paras.delta_rho_prior);
    arma::vec log_Z = log_L1 - log_L0;
    arma::vec prob1 = 1/(1+exp(-log_Z));
    arma::vec u1 = arma::randu(prob1.n_elem);
    arma::uvec delta_pos_idx = arma::find(u1<prob1);
    paras.delta_rho.zeros();
    if(delta_pos_idx.n_elem>0){
      paras.delta_rho(delta_pos_idx).ones();
    }
          


  };
  void update_gamma(){
    arma::vec res = dat.y - dat.M*paras.beta;
    arma::vec Y = svd_x.U.t()*res;
    arma::vec var_gamma = 1.0/(paras.inv_sigma_sq_gamma + paras.inv_sigma_sq*svd_x.D2);
    arma::vec E_gamma_star = var_gamma%(paras.inv_sigma_sq*svd_x.D%Y);
    arma::vec gamma_star = E_gamma_star + arma::randn(dat.m)%sqrt(var_gamma);
    paras.gamma = svd_x.V*gamma_star;
    
    double SSE_gamma = arma::accu(var_gamma + gamma_star%gamma_star);
    paras.inv_sigma_sq_gamma = arma::randg(arma::distr_param(0.5*(dat.m+1),1.0/(0.5*SSE_gamma + paras.inv_a_gamma)));
    paras.inv_a_gamma = arma::randg(arma::distr_param(1.0,1.0/(paras.inv_sigma_sq_gamma + 1)));
    
  };
  void update_SSE(){
    arma::vec res = dat.y - dat.M*paras.beta - dat.X*paras.gamma;
    paras.SSE = arma::dot(res,res);
  };
  void update_inv_sigma_sq(){
    double b_sigma = 0.5*paras.SSE + paras.inv_a ; 
    paras.inv_sigma_sq = arma::randg(arma::distr_param(0.5*(dat.n+1),1.0/b_sigma));
    paras.inv_a = arma::randg(arma::distr_param(1.0,1.0/(paras.inv_sigma_sq + 1)));

    paras.loglik = dat.n/2*log(paras.inv_sigma_sq) - 0.5*paras.inv_sigma_sq*paras.SSE;
  };
  void save_gibbs_sample(){
    if(iter > gibbs_control.burnin){
      if((iter - gibbs_control.burnin)%gibbs_control.thinning==0){
        int mcmc_iter = (iter - gibbs_control.burnin)/gibbs_control.thinning;
        paras_sample.beta.col(mcmc_iter) = paras.beta;
        paras_sample.gamma.col(mcmc_iter) = paras.gamma;
        paras_sample.delta_rho.col(mcmc_iter) = paras.delta_rho;
        paras_sample.f_mu.col(mcmc_iter) = paras.f_mu;
        paras_sample.IP.col(mcmc_iter) = paras.IP;
        paras_sample.sigma_sq(mcmc_iter) = 1.0/paras.inv_sigma_sq;
        paras_sample.sigma_sq_beta(mcmc_iter) = 1.0/paras.inv_sigma_sq_beta;
        paras_sample.sigma_sq_gamma(mcmc_iter) = 1.0/paras.inv_sigma_sq_gamma;
        paras_sample.tau_sq_mu(mcmc_iter) = 1.0/paras.inv_tau_sq_mu;
      }
    }
    paras_sample.loglik(iter) = paras.loglik;
  };

  void run_gibbs(){
    initialize_gibbs_sample();
    std::cout << "Scalar on image: running gibbs " <<std::endl;
    arma::vec Geweke_SSE = arma::zeros(sgd_control.Geweke_interval);
    // Progress prog(gibbs_control.total_iter, display_progress);
    for(iter=0; iter<gibbs_control.total_iter; iter++){
      // prog.increment();
      update_beta();
      update_mu();
      update_gamma();
      update_SSE();
      update_inv_sigma_sq();
      save_gibbs_sample();
      if(gibbs_control.verbose > 0){
        if(iter%gibbs_control.verbose==0){
          std::cout << "Scalar on image: iter: " << iter <<  " sigma_sq: "<< 1.0/paras.inv_sigma_sq << std::endl;
          std::cout <<" mu_quad = "<<CAR.E_mu_quad<< std::endl;
        }
      }

      // if(iter > sgd_control.Geweke_interval){
      //   Geweke_SSE = arma::shift(Geweke_SSE,-1);
      //   Geweke_SSE(sgd_control.Geweke_interval-1) = paras.SSE;
      //   bool Geweke_fail = geweke_test(Geweke_SSE, sgd_control.Geweke_pvalue);
      //   if(Geweke_fail && iter > gibbs_control.burnin){
      //     break;
      //   }
      // }else{
      //   Geweke_SSE(iter) = paras.SSE;
      // }
    }
  };
  
  List get_gibbs_post_mean(){
    arma::vec beta = mean(paras_sample.beta,1);
    arma::vec gamma = mean(paras_sample.gamma,1);
    arma::vec delta_rho = mean(paras_sample.delta_rho,1);
    arma::vec f_mu = mean(paras_sample.f_mu,1);
    arma::vec IP = mean(paras_sample.IP,1);
    double sigma_sq = mean(paras_sample.sigma_sq);
    double sigma_sq_beta = mean(paras_sample.sigma_sq_beta);
    double sigma_sq_gamma = mean(paras_sample.sigma_sq_gamma);
    double tau_sq_mu = mean(paras_sample.tau_sq_mu);
    return List::create(Named("beta") = beta,
                        Named("IP") = IP,
                        Named("f_mu") = f_mu,
                        Named("delta_rho") = delta_rho,
                        Named("gamma") = gamma,
                        Named("sigma_sq") = sigma_sq,
                        Named("sigma_beta_sq") = sigma_sq_beta,
                        Named("sigma_gamma_sq") = sigma_sq_gamma,
                        Named("tau_mu_sq") = tau_sq_mu);
  
  };
  List get_gibbs_trace(){
    return List::create(Named("loglik") = paras_sample.loglik,
                        Named("beta") = paras_sample.beta,
                        Named("gamma") = paras_sample.gamma,
                        Named("delta_rho") = paras_sample.delta_rho,
                        Named("f_mu") = paras_sample.f_mu,
                        Named("IP") = paras_sample.IP,
                        Named("sigma_sq") = paras_sample.sigma_sq,
                        Named("sigma_beta_sq") = paras_sample.sigma_sq_beta,
                        Named("sigma_gamma_sq") = paras_sample.sigma_sq_gamma,
                        Named("tau_mu_sq") = paras_sample.tau_sq_mu);   
  
  };
  List get_gibbs_control(){
    return List::create(Named("mcmc_sample")= gibbs_control.mcmc_sample,
                        Named("burnin") = gibbs_control.burnin,
                        Named("thinning") = gibbs_control.thinning,
                        Named("total_iter") = gibbs_control.total_iter,
                        Named("verbose") = gibbs_control.verbose,
                        Named("save_profile") = gibbs_control.save_profile,
                        Named("total_profile") = gibbs_control.total_profile);
  
  };
};


//' @title Scalar on Image regression standalone
//' @description
//' Scalar on Image regression using the sparse-mean prior
//' @name Scalar_on_Image
//' @param y outcome
//' @param X vector covariate
//' @param M Matrix of functional images
//' @param lambda thresholding parameter
//' @param rho scaling parameter in CAR model
//' @param B matrix of covariance neighborhood
//' @param in_Sigma_inv
//' @param in_D
//' @param init_paras
//' @param method
//' @import Rcpp
//' @useDynLib STCAR, .registration=TRUE
//' @export
//' [[Rcpp::export(rng = false)]]
List SonI_CAVI_rho(arma::vec& y, arma::mat& X, 
                    arma::mat& M, 
                    double lambda,
                    double rho, const arma::sp_mat& B,
                    const arma::sp_mat& W,
                    const arma::sp_mat& in_Sigma_inv,
                    const arma::vec& in_D,
                    CharacterVector method,
                    List& init_paras,
                    List& SGD_controls,
                    List& Geweke_controls,
                    List& sigmasq_step_controls,
                    const arma::vec& in_delta_rho,
                    double initial_sigma_sq = 1,
                    double initial_sigma_beta_sq = 1,
                    double initial_sigma_gamma_sq = 1,
                    double initial_tau_mu_sq = 1,
                    int mcmc_sample = 1000, 
                    int burnin = 1000, 
                    int thinning = 1,
                    int max_iter = 1000,
                    int begin_f_beta = 0,
                    int f_beta_interval = 1,
                    double paras_diff_tol = 1e-6,
                    double SGD_step = 1e-2,
                    int ELBO_stop = 1,
                    double ELBO_diff_tol = 1e-6,
                    int verbose = 5000,
                    int save_profile = 1,
                    bool trace_all_ELBO = false,
                    bool include_Confounder = true,
                    bool update_beta = true,
                    bool update_gamma = true,
                    bool update_delta_rho = true,                      
                    bool display_progress = false){
 
 arma::wall_clock timer, timer_preprocess;
 
 timer.tic();
 timer_preprocess.tic();
 SonI_standalone model;
 
 Rcout<<"Scalar_on_Image...1"<<std::endl;
 model.include_Confounder = include_Confounder;
 model.load_data(y,X,M,lambda);
 model.set_method(method);
 Rcout<<"Scalar_on_Image...2"<<std::endl;
 model.load_CAR(rho, B, W, in_Sigma_inv, in_D);
 Rcout<<"Scalar_on_Image...3"<<std::endl;
 model.display_progress = display_progress;
 
 if(model.method == 0){
  model.set_gibbs_control(mcmc_sample, burnin, thinning, verbose, save_profile, begin_f_beta);
 }else{
  model.set_vb_control(max_iter,
                      paras_diff_tol,
                      ELBO_stop,
                      ELBO_diff_tol,
                      trace_all_ELBO,
                      verbose,
                      save_profile,
                      begin_f_beta,
                      update_beta,
                      update_gamma,
                      update_delta_rho);
 
 }
  model.set_sigmasq_beta_annealing_control(sigmasq_step_controls);
  // model.set_SGD_controls(SGD_controls, Geweke_controls);
  if(model.method == 2){
    model.set_SGD_controls(SGD_controls, Geweke_controls);
  }
  
 std::cout << "set control done" << std::endl;
 
 model.set_paras_initial_values(init_paras,
                                in_delta_rho,
                                initial_sigma_sq, 
                                initial_sigma_beta_sq,
                                initial_sigma_gamma_sq,
                                initial_tau_mu_sq);
 Rcout<<"Scalar_on_Image...6"<<std::endl;
 
 if(model.method == 1){
   model.run_CAVI(f_beta_interval); 
   Rcout<<"Scalar_on_Image...7-2"<<std::endl;
 } else if(model.method == 2){
   model.run_SGD(f_beta_interval);
   Rcout<<"Scalar_on_Image...7-3"<<std::endl;
 } else if(model.method == 0){
   model.run_gibbs();
   Rcout<<"Scalar_on_Image...7-1"<<std::endl;
 }
 
 
 double elapsed = timer.toc();
 
 
 List output;
 if(model.method >0){
    output = List::create(Named("post_mean") = model.get_vb_post_mean(),
                       Named("iter") = model.get_iter(),
                       Named("trace") = model.get_vb_trace(),
                       Named("vb_control") = model.get_vb_control(),
                       Named("test_output") = model.get_test_output(),
                       Named("elapsed") = elapsed);
 
 }else{
    output = List::create(Named("post_mean") = model.get_gibbs_post_mean(),
                       Named("iter") = model.get_iter(),
                       Named("trace") = model.get_gibbs_trace(),
                       Named("gibbs_control") = model.get_gibbs_control(),
                       Named("test_output") = model.get_test_output(),
                       Named("elapsed") = elapsed);
 }
 
 return output;
 
}
