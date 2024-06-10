#include "STCAR_help.h"
// [[Rcpp::depends(RcppArmadillo)]]


// [[Rcpp::depends(BH)]]
#include <RcppArmadillo.h>


using namespace Rcpp;
// using namespace arma;

// #include <progress.hpp>
// #include <progress_bar.hpp>
// // [[Rcpp::depends(RcppProgress)]]


class IonS_standalone{
private:
    struct LinearRegData{
        int n;
        int p;
        int m;
        arma::mat M; // p by n
        arma::mat M_star; // L by n
        arma::vec X; // n by 1
        arma::mat C; // m by n
        double lambda;
        double X_l2;
        arma::vec C_l2;
    }dat, sub_dat;

    struct GPbasis{
        int L;
        int num_region;
        Rcpp::List Qlist;
        Rcpp::List region_idx;
        Rcpp::List L_idx;
        arma::vec D_vec;
    }basis;

    struct VBLinearRegParas{
        arma::vec E_alpha;
        arma::vec E_theta_alpha;
        arma::vec E_IP;
        arma::vec grad_E_alpha;
        double Var_alpha; // marginal
        

        arma::vec E_neighbor_mean;
        arma::vec E_mu_neighbor_2moment;
        arma::vec E_mu;
        arma::vec E_f_mu;
        arma::vec E_mu2;
        arma::vec E_f_mu2;
        arma::vec Var_f_mu;
        arma::mat mix_prob; // L by 3
        arma::vec E_delta_rho;
        arma::vec E_delta_rho_prior;
        

        arma::mat E_theta_zeta; // L by m
        arma::mat E_theta_eta; // L by n
        arma::mat theta_eta_mean;
        arma::mat Var_theta_eta; // L by n
        arma::mat Var_theta_zeta; // L by m
        
        double E_inv_sigma_sq;
        double E_inv_sigma_sq_alpha;
        double E_inv_sigma_sq_zeta;
        double E_inv_sigma_sq_eta;
        double E_inv_tau_sq_mu;
        double E_inv_a;
        double E_inv_a_alpha;
        double E_inv_a_zeta;
        double E_inv_a_eta;
        double E_SSE;
    } vb_paras;

    struct GibbsParams{
      arma::vec alpha;
      arma::vec theta_alpha;
      arma::vec f_mu;
      arma::mat theta_zeta;
      arma::mat theta_eta;
      arma::mat theta_eta_mean;
      arma::vec IP;
      
      arma::vec neighbor_mean;
      arma::vec delta_rho;// 0 or 1
      arma::vec delta_rho_prior;

      double inv_sigma_sq;
      double inv_sigma_sq_alpha;
      double inv_sigma_sq_zeta;
      double inv_sigma_sq_eta;
      double inv_tau_sq_mu;
      double inv_a;
      double inv_a_alpha;
      double inv_a_zeta;
      double inv_a_eta;
      double loglik;
      double SSE;


    }paras;

    struct MCMCsample{
      arma::mat alpha;
      arma::mat f_mu;
      arma::mat IP;
      arma::mat delta_rho;
      arma::cube theta_zeta;
      arma::vec sigma_sq;
      arma::vec sigma_sq_alpha;
      arma::vec sigma_sq_zeta;
      arma::vec sigma_sq_eta;
      arma::vec tau_sq_mu;
      arma::vec loglik;
    }paras_sample;

    struct GibbsSamplerControl{
      int total_iter;
      int burnin;
      int mcmc_sample;
      int thinning;
      int verbose;
      int save_profile;
      int total_profile;
      int begin_f_alpha;
      int eta_freq;
    } gibbs_control;


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

    struct VBProfile{
        arma::vec E_inv_sigma_sq;
        arma::mat E_alpha;
        arma::mat E_delta_rho;
        arma::mat E_IP;
        arma::mat E_f_mu;
        arma::vec E_inv_sigma_sq_alpha;
        arma::vec E_inv_sigma_sq_eta;
        arma::vec E_inv_sigma_sq_zeta;
        arma::vec E_inv_tau_sq_mu;
    } vb_profile;
    
    struct VBControl{
      int max_iter;
      double para_diff_tol;
      int verbose;
      int save_profile;
      int total_profile;
      int begin_f_mu;
      int eta_freq;
    } vb_control;

    struct AnnealingControl{
        double a;
        double b;
        double gamma;
    }sigmasq_alpha_control;

    struct SGDControl{
        double step;
        int subsample_size;
        arma::uvec subsample_idx;
    }sgd_control;

    int iter;

public:
    List test_output;
    double elapsed_sgd = 0;
    bool display_progress = true;
    int method;

    void set_method(CharacterVector in_method){
        if(in_method(0)=="CAVI"){
        std::cout << "Image on scalar: Coordinate Ascent Mean Field Variational Bayes" << std::endl;
        method = 1;
        } else if(in_method(0)=="SGD"){
        std::cout << "Image on scalar: Stochastic gradient descent on global variables VI" << std::endl;
        method = 2;
        } else if(in_method(0)=="Gibbs"){
          std::cout << "Image on scalar: Gibbs sampler" << std::endl;
          method = 0;
        }
    };

    void load_data(const arma::mat& in_X, const  arma::mat& in_M, 
                 const  arma::mat& in_C, 
                 const double& in_lambda){
        Rcout<<"Image on scalar: Loading data...."<<std::endl;
        Rcout<<"Dimensions: M (p by n), C(m by n)"<<std::endl;
        dat.X = in_X;
        dat.M = in_M;
        dat.C = in_C;
        dat.lambda = in_lambda;
        dat.X_l2 = arma::accu(dat.X%dat.X);
        dat.C_l2 = sum(dat.C%dat.C,1);
        
        
        dat.n = dat.X.n_elem;
        dat.p = dat.M.n_rows;
        
        dat.m = dat.C.n_rows;
        
        Rcout<<"Scalar on image: Load data successful"<<std::endl;
    };

    void load_basis(const Rcpp::List& in_basis){
    
        Rcpp::List Qlist = in_basis["Phi_Q"];
        Rcpp::List region_idx = in_basis["region_idx_cpp"];
        Rcpp::List L_idx = in_basis["L_idx_cpp"];
        arma::vec D_vec = in_basis["D_vec"];
        Rcout<<"load_basis...2"<<std::endl;
        basis.Qlist = Qlist;
        basis.region_idx = region_idx;
        basis.D_vec = D_vec;
        basis.L = D_vec.n_elem;
        basis.num_region = basis.Qlist.size();
        basis.L_idx = L_idx;
        dat.M_star = High_to_low(dat.M, basis.L, basis.Qlist,
                                basis.region_idx, basis.L_idx);
    }

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
                                const double& in_sigma_sq_alpha,
                                const double& in_sigma_sq_zeta,
                                const double& in_sigma_sq_eta,
                                const double& in_tau_sq_mu){
      arma::vec alpha = init["alpha"];
      arma::vec f_alpha = init["f_alpha"];
      arma::mat zeta = init["zeta_m"];
      arma::mat theta_eta = init["theta_eta"];
      if(arma::max(arma::abs(alpha - f_alpha)) < 1e-3){
        Rcout<<"Warning: initial value for beta and f_beta cannot be too close."<<std::endl;
        Rcout<<" Automatically adding error N(0,1) to the initial value of beta."<<std::endl;
        alpha = alpha + arma::randn(dat.p,1);
      }
      arma::mat theta_zeta = High_to_low(zeta, basis.L, basis.Qlist,
                                        basis.region_idx, basis.L_idx);
      arma::mat eta = Low_to_high(theta_eta, dat.p, basis.Qlist,
                                  basis.region_idx, basis.L_idx);
      arma::vec theta_alpha = High_to_low_vec(alpha, basis.L, basis.Qlist,
                                          basis.region_idx, basis.L_idx);
      
      if(method != 0){
        vb_paras.E_delta_rho = in_delta_rho;
        vb_paras.E_delta_rho_prior = arma::ones(dat.p,1)*0.5;
        vb_paras.E_alpha = alpha;
        vb_paras.E_theta_alpha = theta_alpha;
        vb_paras.E_mu = f_alpha + arma::randn( dat.p, arma::distr_param(0.0, 0.01) );
        vb_paras.E_theta_eta = theta_eta;
        vb_paras.grad_E_alpha = arma::zeros(dat.p,1);
        vb_paras.E_f_mu = f_alpha;
        vb_paras.Var_alpha = 1; // marginal variance
        vb_paras.E_theta_zeta =  theta_zeta;
        vb_paras.theta_eta_mean = arma::zeros(basis.L,dat.n);
        vb_paras.Var_theta_zeta = in_sigma_sq_zeta*arma::ones(basis.L, dat.m);
        vb_paras.Var_theta_eta = in_sigma_sq_eta* arma::ones(basis.L, 1);
        vb_paras.E_inv_sigma_sq = 1.0/in_sigma_sq;
        vb_paras.E_inv_sigma_sq_alpha = 1.0/in_sigma_sq_alpha;
        vb_paras.E_inv_sigma_sq_eta = 1.0/in_sigma_sq_eta;
        vb_paras.E_inv_sigma_sq_zeta = 1.0/in_sigma_sq_zeta;
        vb_paras.E_inv_tau_sq_mu = 1.0/in_tau_sq_mu;
        vb_paras.E_inv_a_alpha = 1.0;
        vb_paras.E_inv_a = 1.0;
        vb_paras.E_inv_a_zeta = vb_paras.E_inv_a_eta = 1.0;
        vb_paras.E_SSE = 1;
        vb_paras.E_IP = arma::zeros(dat.p,1);

        // initialize sparse mean params
        vb_paras.E_mu_neighbor_2moment = arma::ones(dat.p,1);
        vb_paras.E_mu2 = arma::ones(dat.p,1);
        vb_paras.Var_f_mu =  arma::ones(dat.p,1);
        vb_paras.E_f_mu2 = arma::ones(dat.p,1);
        vb_paras.E_neighbor_mean = CAR.rho * CAR.B*vb_paras.E_mu;
        vb_paras.E_neighbor_mean %= vb_paras.E_delta_rho;
      
        vb_paras.mix_prob.zeros(dat.p,3);
        vb_paras.mix_prob.col(0).ones();  
      }else{
        paras.alpha = alpha;
        paras.theta_alpha = theta_alpha;
        paras.f_mu = f_alpha;
        paras.theta_zeta = theta_zeta;
        paras.theta_eta = theta_eta;
        paras.theta_eta_mean = arma::zeros(basis.L,dat.n);
        paras.IP = arma::zeros(dat.p,1);

        paras.neighbor_mean = CAR.rho * CAR.B*paras.f_mu;
        paras.neighbor_mean %= in_delta_rho;
        
        paras.delta_rho = in_delta_rho;
        paras.delta_rho_prior = arma::ones(dat.p,1)*0.5;

        paras.inv_sigma_sq = 1.0/in_sigma_sq;
        paras.inv_sigma_sq_alpha = 1.0/in_sigma_sq_alpha;
        paras.inv_sigma_sq_eta = 1.0/in_sigma_sq_eta;
        paras.inv_sigma_sq_zeta = 1.0/in_sigma_sq_zeta;
        paras.inv_tau_sq_mu = 1.0/in_tau_sq_mu;
        paras.inv_a_alpha = 1.0;
        paras.inv_a = 1.0;
        paras.inv_a_zeta = paras.inv_a_eta = 1.0;
        paras.SSE = 1;
      }
      
      
      Rcout<<"Image on scalar: set initial values successful"<<std::endl;
    };

    void initialize_paras_sample(){
      paras_sample.alpha = arma::zeros(dat.p, gibbs_control.mcmc_sample);
      paras_sample.f_mu = arma::zeros(dat.p, gibbs_control.mcmc_sample);
      paras_sample.IP = arma::zeros(dat.p, gibbs_control.mcmc_sample);
      paras_sample.delta_rho = arma::zeros(dat.p, gibbs_control.mcmc_sample);
      paras_sample.theta_zeta = arma::zeros(basis.L, dat.m, gibbs_control.mcmc_sample);
      paras_sample.sigma_sq = arma::zeros(gibbs_control.mcmc_sample);
      paras_sample.sigma_sq_alpha = arma::zeros(gibbs_control.mcmc_sample);
      paras_sample.sigma_sq_zeta = arma::zeros(gibbs_control.mcmc_sample);
      paras_sample.sigma_sq_eta = arma::zeros(gibbs_control.mcmc_sample);
      paras_sample.tau_sq_mu = arma::zeros(gibbs_control.mcmc_sample);
      paras_sample.loglik = arma::zeros(gibbs_control.total_iter);

    };

    

    void set_vb_control(int in_max_iter, 
                      double in_para_diff_tol, 
                      int in_verbose,
                      int in_save_profile,
                      int begin_f_mu){
      vb_control.max_iter = in_max_iter;
      vb_control.para_diff_tol = in_para_diff_tol;
      vb_control.verbose = in_verbose;
      vb_control.save_profile = in_save_profile;
      vb_control.begin_f_mu = begin_f_mu;

      if(vb_control.save_profile > 0){
        vb_control.total_profile = vb_control.max_iter/vb_control.save_profile;
      } else{
        vb_control.total_profile = 0;
      }
      
    };

    void set_sigmasq_alpha_annealing_control(List& in_sigmasq_alpha_control){
      double a = in_sigmasq_alpha_control["a"];
      double b = in_sigmasq_alpha_control["b"];
      double gamma = in_sigmasq_alpha_control["gamma"];
      sigmasq_alpha_control.a = a;
      sigmasq_alpha_control.b = b;
      sigmasq_alpha_control.gamma = gamma;
    };

    void update_E_SSE(){
      arma::mat res_star = dat.M_star - vb_paras.E_theta_alpha*dat.X.t() - 
        vb_paras.E_theta_zeta*dat.C -
        vb_paras.E_theta_eta;
      arma::mat var_term = vb_paras.Var_alpha*arma::ones(basis.L,1) * arma::trans(dat.X%dat.X);
      var_term += vb_paras.Var_theta_zeta * (dat.C% dat.C);
      var_term.each_col() += vb_paras.Var_theta_eta;
      
      vb_paras.E_SSE = arma::accu(res_star%res_star) + arma::accu(var_term);
      

    };

    void update_E_alpha(){
      arma::mat other_star =  vb_paras.E_theta_zeta*dat.C + vb_paras.E_theta_eta;
      arma::mat other_highd = Low_to_high(other_star, dat.p, basis.Qlist,
                                          basis.region_idx, basis.L_idx);
      arma::mat E_res = dat.M - other_highd;
      double var = 1.0/(vb_paras.E_inv_sigma_sq*dat.X_l2 + vb_paras.E_inv_sigma_sq_alpha);
      
      arma::vec mu = vb_paras.E_inv_sigma_sq*(E_res*dat.X);
      mu += vb_paras.E_inv_sigma_sq_alpha * vb_paras.E_f_mu;
      mu *= var;
      vb_paras.E_alpha = mu;
      vb_paras.E_theta_alpha = High_to_low_vec(vb_paras.E_alpha, basis.L, basis.Qlist,
                                        basis.region_idx, basis.L_idx);
      vb_paras.Var_alpha = var;
      

      // annealing on sigmasq_alpha
      vb_paras.E_inv_sigma_sq_alpha = 1.0/(sigmasq_alpha_control.a*pow(sigmasq_alpha_control.b+iter,
                                          sigmasq_alpha_control.gamma));
    };

    void update_E_mu(){
    
      double E_inv_sigma_sq_mu = vb_paras.E_inv_tau_sq_mu;
      
      
      arma::vec Vi = 1./(vb_paras.E_inv_sigma_sq_alpha + E_inv_sigma_sq_mu*CAR.D);
      
      arma::vec Vi_sqrt = sqrt(Vi);
      arma::vec V0 = 1./(E_inv_sigma_sq_mu*CAR.D);
      arma::vec V0_inv = 1/V0;
      arma::vec V0_sqrt = sqrt(V0);
      
      arma::vec y_plus_lambda = vb_paras.E_alpha + dat.lambda;
      arma::vec y_plus_lambda2 = y_plus_lambda%y_plus_lambda;
      arma::vec y_minus_lambda = vb_paras.E_alpha - dat.lambda;
      arma::vec y_minus_lambda2 = y_minus_lambda%y_minus_lambda;
      
      arma::vec y2 = vb_paras.E_alpha%vb_paras.E_alpha;
      
      arma::vec mu_pos = Vi % (vb_paras.E_inv_sigma_sq_alpha*(y_plus_lambda) + 
        V0_inv%vb_paras.E_neighbor_mean);
      arma::vec mu_neg = Vi % (vb_paras.E_inv_sigma_sq_alpha*(y_minus_lambda) + 
        V0_inv%vb_paras.E_neighbor_mean);
      arma::vec mu_zero = vb_paras.E_neighbor_mean;
      
      // tn_mean, tn_var, tn_2moment, Z
      arma::mat pos_stats = truncated_normal_stats_vec(mu_pos, Vi_sqrt, dat.lambda, 1, 4);
      arma::mat neg_stats = truncated_normal_stats_vec(mu_neg,  Vi_sqrt, dat.lambda, -1, 4);
      arma::mat zero_stats = truncated_normal_stats_vec(mu_zero, V0_sqrt, dat.lambda,0,4);
      
      // get mixing prob
      arma::vec C_pos = V0_inv%vb_paras.E_mu_neighbor_2moment;
      arma::vec C_neg = C_pos; arma::vec C_zero = C_pos;
      
      C_pos += vb_paras.E_inv_sigma_sq_alpha*y_plus_lambda2;
      C_neg += vb_paras.E_inv_sigma_sq_alpha*y_minus_lambda2;
      C_zero += vb_paras.E_inv_sigma_sq_alpha*y2;
      
      arma::vec log_Z_pos = -0.5*C_pos + 0.5*mu_pos%mu_pos/Vi + 0.5*log(Vi) + log(pos_stats.col(3));
      arma::vec log_Z_neg = -0.5*C_neg + 0.5*mu_neg%mu_neg/Vi + 0.5*log(Vi) + log(neg_stats.col(3));
      arma::vec log_Z_zero = -0.5*C_zero + 0.5*mu_zero%mu_zero/V0 + 0.5*log(V0) + log(zero_stats.col(3));
      
      
      
      vb_paras.mix_prob = get_mix_prob(log_Z_neg, log_Z_zero, log_Z_pos);
      
      
      arma::mat E_alpha_mat = vb_paras.mix_prob % 
        join_horiz(neg_stats.col(0),zero_stats.col(0),pos_stats.col(0));
      E_alpha_mat.elem(find(vb_paras.mix_prob==0)).zeros(); 
      vb_paras.E_mu = sum(E_alpha_mat,1);
      
      E_alpha_mat = vb_paras.mix_prob % 
        join_horiz(neg_stats.col(2),zero_stats.col(2),pos_stats.col(2));
      E_alpha_mat.elem(find(vb_paras.mix_prob==0)).zeros();
      vb_paras.E_mu2 = sum(E_alpha_mat,1);  
      
      E_alpha_mat = vb_paras.mix_prob % 
        join_horiz(neg_stats.col(0)+dat.lambda,
                  arma::zeros(dat.p,1),
                  pos_stats.col(0)-dat.lambda);
      E_alpha_mat.elem(find(vb_paras.mix_prob==0)).zeros();
      vb_paras.E_f_mu = sum(E_alpha_mat,1);  
      
      
      if(vb_paras.E_alpha.has_nan()){
        Rcout<<"Error!!! E_alpha has nan"<<std::endl;
        test_output = List::create(Named("zero_stats")=zero_stats,
                                  Named("pos_stats")=pos_stats,
                                  Named("neg_stats")=neg_stats,
                                  Named("mix_prob")=vb_paras.mix_prob,
                                  Named("mu_pos")= mu_pos,
                                  Named("Vi_sqrt")= Vi_sqrt);
      }
      if(vb_paras.E_mu2.has_nan()){
        Rcout<<"Error!!! E_alpha2 has nan"<<std::endl;
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
      
      arma::mat Cov_q_full = vb_paras.E_alpha * vb_paras.E_alpha.t();
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
      
      
      vb_paras.E_inv_tau_sq_mu = 0.5*(dat.p+1)/(0.5*b_mid + 1/2); // need to change sigma_beta accordingly
      
      // update IP
      vb_paras.E_IP = 1 - vb_paras.mix_prob.col(1);
      
      // update delta_rho
      
      E_inv_sigma_sq_mu = vb_paras.E_inv_tau_sq_mu;
      arma::vec Var_mu = vb_paras.E_mu2 - vb_paras.E_mu % vb_paras.E_mu;
      arma::vec bias = vb_paras.E_mu - vb_paras.E_neighbor_mean;
      arma::vec mse = bias%bias + Var_mu;
      arma::vec log_L1 = -0.5*(E_inv_sigma_sq_mu*CAR.D) % mse + log(vb_paras.E_delta_rho_prior);
      arma::vec log_L0 = -0.5*(E_inv_sigma_sq_mu*CAR.D) % vb_paras.E_mu2 + log(1-vb_paras.E_delta_rho_prior);
      arma::vec log_Z = log_L1 - log_L0;
      vb_paras.E_delta_rho = 1/(1+exp(-log_Z));
      
    };

    void update_E_theta_zeta(){
    
      arma::mat E_res_star = dat.M_star - vb_paras.E_theta_alpha*dat.X.t() - vb_paras.E_theta_eta;
      
      for(int k=0; k<dat.m; k++){
        arma::uvec k_c = complement(k,k,dat.m);
        arma::vec var_k = 1/(vb_paras.E_inv_sigma_sq*dat.C_l2(k) + 
          vb_paras.E_inv_sigma_sq_zeta/basis.D_vec);
        vb_paras.Var_theta_zeta.col(k) = var_k;
        
        arma::vec mu_k = vb_paras.E_inv_sigma_sq*(E_res_star - vb_paras.E_theta_zeta.cols(k_c) *
          dat.C.rows(k_c)) *
          arma::trans(dat.C.row(k)); // L by 1
        mu_k %= var_k;
        vb_paras.E_theta_zeta.col(k) = arma::randn(basis.L,1)%sqrt(var_k) + mu_k;
        
      }
      // update sigma_zeta
      double a = 0.5*(basis.L*dat.m + 1);
      arma::mat temp = vb_paras.E_theta_zeta.each_col()%(1.0/basis.D_vec);
      double E_quad =  arma::accu( temp % vb_paras.E_theta_zeta );
      double b_sigma_zeta = 0.5* E_quad+ vb_paras.E_inv_a_zeta;
      vb_paras.E_inv_sigma_sq_zeta = a/b_sigma_zeta;
      vb_paras.E_inv_a_zeta = 1.0/(1+vb_paras.E_inv_sigma_sq_zeta);
      
    }

    void update_E_theta_eta(){
    
      arma::mat E_res_star = dat.M_star - vb_paras.E_theta_alpha*dat.X.t() - vb_paras.E_theta_zeta*dat.C;
      vb_paras.Var_theta_eta = 1/(vb_paras.E_inv_sigma_sq + vb_paras.E_inv_sigma_sq_eta/basis.D_vec);
      E_res_star.each_col() %= vb_paras.Var_theta_eta*vb_paras.E_inv_sigma_sq;
      vb_paras.E_theta_eta = E_res_star;
      
      
      // update sigma_eta
      double a = 0.5*(basis.L*dat.n + 1);
      arma::mat temp = vb_paras.E_theta_eta.each_col()%(1.0/basis.D_vec);
      double E_quad = arma::accu( temp % vb_paras.E_theta_eta ) ;
      double b_sigma_eta = 0.5*E_quad+ vb_paras.E_inv_a_eta;
      vb_paras.E_inv_sigma_sq_eta = a/b_sigma_eta; // uncomment this
      vb_paras.E_inv_a_eta = 1.0/(1+vb_paras.E_inv_sigma_sq_eta);

    };

    void update_E_inv_sigma_sq(){
      double b_sigma = 0.5*vb_paras.E_SSE + vb_paras.E_inv_a;
      vb_paras.E_inv_sigma_sq = (dat.n*basis.L+1)/2/b_sigma;
      vb_paras.E_inv_a = 1/(1+vb_paras.E_inv_sigma_sq);
    
    };

    double compute_paras_diff(arma::vec& beta, arma::vec& beta_prev){
      arma::vec temp = beta - beta_prev;
      return accu(temp%temp)/beta.n_elem;
    };

    void initialize_vb_profile(){
      if(vb_control.save_profile>0){
        vb_profile.E_inv_sigma_sq.zeros(vb_control.total_profile);
        vb_profile.E_inv_sigma_sq_alpha.zeros(vb_control.total_profile);
        vb_profile.E_inv_sigma_sq_eta.zeros(vb_control.total_profile);
        vb_profile.E_inv_sigma_sq_zeta.zeros(vb_control.total_profile);
        vb_profile.E_inv_tau_sq_mu.zeros(vb_control.total_profile);
        vb_profile.E_alpha.zeros(dat.p, vb_control.total_profile);
        vb_profile.E_delta_rho.zeros(dat.p, vb_control.total_profile);
        vb_profile.E_f_mu.zeros(dat.p, vb_control.total_profile);
        vb_profile.E_IP.zeros(dat.p, vb_control.total_profile);
        
      }
      
    };

    void save_vb_profile(){
      if(vb_control.save_profile > 0){
        if(iter%vb_control.save_profile==0){
          int profile_iter = iter/vb_control.save_profile;
          vb_profile.E_inv_sigma_sq(profile_iter) = vb_paras.E_inv_sigma_sq;
          vb_profile.E_inv_sigma_sq_eta(profile_iter) = vb_paras.E_inv_sigma_sq_eta;
          vb_profile.E_inv_sigma_sq_zeta(profile_iter) = vb_paras.E_inv_sigma_sq_zeta;
          vb_profile.E_inv_tau_sq_mu(profile_iter) = vb_paras.E_inv_tau_sq_mu;
          vb_profile.E_inv_sigma_sq_alpha(profile_iter) = vb_paras.E_inv_sigma_sq_alpha;
          vb_profile.E_f_mu.col(profile_iter) = vb_paras.E_f_mu;
          vb_profile.E_alpha.col(profile_iter) = vb_paras.E_alpha;
          vb_profile.E_delta_rho.col(profile_iter) = vb_paras.E_delta_rho;
          vb_profile.E_IP.col(profile_iter) = vb_paras.E_IP;
        }
      }
    };

    void monitor_vb(){
      if(vb_control.verbose > 0){
        if(iter%vb_control.verbose==0){
          std::cout << "Image on scalar: iter: " << iter <<  " sigma_sq: "<<  1.0/vb_paras.E_inv_sigma_sq;
          std::cout <<" mu_quad = "<<CAR.E_mu_quad<< std::endl;
          if(method == 2){
            std::cout<<"min(grad) = "<<min(vb_paras.grad_E_alpha)<<"; max(grad)="<<max(vb_paras.grad_E_alpha)<<std::endl;
          }
        }
      }
    }

  void run_CAVI(int f_alpha_interval){
    initialize_vb_profile();
    std::cout << "Image on scalar: running CAVI " <<std::endl;

    for(iter=0; iter<vb_control.max_iter; iter++){

      arma::vec E_alpha_prev = vb_paras.E_alpha;
      if((iter % f_alpha_interval == 0) & (iter>vb_control.begin_f_mu)){
        update_E_mu();
      }
      update_E_alpha();

      update_E_theta_zeta();

      if( (iter%vb_control.eta_freq) == 0){
        update_E_theta_eta();
      }
      update_E_SSE();
      
      update_E_inv_sigma_sq();

      
      save_vb_profile();
      monitor_vb();

      if(compute_paras_diff(vb_paras.E_alpha,E_alpha_prev) < vb_control.para_diff_tol){
        save_vb_profile();
        monitor_vb();
        break;
      }


    }

  };

  void set_SGD_controls(List SGD_controls){
    double step = SGD_controls["step"];
    sgd_control.step = step;
    int subsample_size = SGD_controls["subsample_size"];
    sgd_control.subsample_size = subsample_size;
  };
  
  void SGD_initialize_subsample(const arma::uvec& sub_sample){
    sub_dat.n = sub_sample.n_elem;
    sub_dat.M = dat.M.cols(sub_sample);
    sub_dat.X = dat.X.rows(sub_sample);
    sub_dat.M_star = dat.M_star.cols(sub_sample);
    sub_dat.X = dat.X.rows(sub_sample);
    sub_dat.C = dat.C.cols(sub_sample);
    sgd_control.subsample_idx = sub_sample;
  }

  void SGD_update_E_alpha(){
    
    arma::mat other =  vb_paras.E_theta_zeta*sub_dat.C +
      vb_paras.E_theta_eta.cols(sgd_control.subsample_idx);
    arma::mat other_highd = Low_to_high(other, dat.p, basis.Qlist,
                                  basis.region_idx, basis.L_idx); // p by n
    arma::mat E_res_sub = sub_dat.M - other_highd;
    
    arma::vec temp = E_res_sub*(-sub_dat.X); // p by 1

    vb_paras.grad_E_alpha = - vb_paras.E_inv_sigma_sq*temp -  
      vb_paras.E_inv_sigma_sq * vb_paras.E_alpha*arma::dot(sub_dat.X,sub_dat.X);
    arma::vec post_grad = vb_paras.grad_E_alpha;
    arma::vec prior_grad = -vb_paras.E_inv_sigma_sq_alpha*(vb_paras.E_alpha -
      vb_paras.E_f_mu);
    vb_paras.grad_E_alpha += prior_grad;
    vb_paras.E_alpha += sgd_control.step * (dat.n/sub_dat.n*post_grad + prior_grad);
    
    vb_paras.E_theta_alpha = High_to_low_vec(vb_paras.E_alpha, basis.L, basis.Qlist,
                                             basis.region_idx, basis.L_idx);

    
    
    // annealing on sigmasq_alpha
    vb_paras.E_inv_sigma_sq_alpha = 1.0/(sigmasq_alpha_control.a*pow(sigmasq_alpha_control.b+iter,
                                        sigmasq_alpha_control.gamma));
  }

  void run_SGD(int f_alpha_interval){
    arma::wall_clock timer;
    
    initialize_vb_profile();
    std::cout << "Image on scalar: running SGD " <<std::endl;
    
    // Progress prog(vb_control.max_iter, display_progress);
    for(iter=0; iter<vb_control.max_iter; iter++){
      // prog.increment();

      arma::vec E_alpha_prev = vb_paras.E_alpha;
      if((iter % f_alpha_interval == 0) & (iter>vb_control.begin_f_mu)){
        update_E_mu();
      }
      sgd_control.subsample_idx = arma::randperm(dat.n, sgd_control.subsample_size);
      SGD_initialize_subsample(sgd_control.subsample_idx);
      SGD_update_E_alpha();// Use SGD to update all locations on a subsample
      
      update_E_theta_zeta();

      if( (iter%vb_control.eta_freq) == 0){
        update_E_theta_eta();
      }
      update_E_SSE();
      
      update_E_inv_sigma_sq();

      
      save_vb_profile();
      monitor_vb();

      if(compute_paras_diff(vb_paras.E_alpha,E_alpha_prev) < vb_control.para_diff_tol){
        save_vb_profile();
        monitor_vb();
        break;
      }

      
    }// end of for-loop
  };
    
    // gibbs related functions
    void set_gibbs_control(int in_mcmc_sample, int in_burnin, int in_thinning, 
                         int in_verbose, int in_save_profile, int in_begin_f_alpha, 
                         int eta_freq){
      gibbs_control.mcmc_sample = in_mcmc_sample;
      gibbs_control.burnin = in_burnin;
      gibbs_control.thinning = in_thinning;
      gibbs_control.total_iter = gibbs_control.burnin;
      gibbs_control.total_iter += gibbs_control.mcmc_sample*gibbs_control.thinning; 
      gibbs_control.verbose = in_verbose;
      gibbs_control.save_profile = in_save_profile;
      gibbs_control.begin_f_alpha = in_begin_f_alpha;
      gibbs_control.eta_freq = eta_freq;
      if(gibbs_control.save_profile > 0){
        gibbs_control.total_profile = gibbs_control.total_iter/gibbs_control.save_profile;
      } else{
        gibbs_control.total_profile = 0;
      }
      Rcout<<"Image on scalar: set gibbs control successful"<<std::endl;
    };
  

    void update_alpha(){
      arma::mat other = paras.theta_zeta * dat.C + paras.theta_eta; 
      arma::mat other_highd = Low_to_high(other, dat.p, basis.Qlist,
                                          basis.region_idx, basis.L_idx); // p by n
      arma::mat res = dat.M - other_highd;

      double var = 1/(paras.inv_sigma_sq*dat.X_l2 + paras.inv_sigma_sq_alpha);
      
      arma::vec mu = paras.inv_sigma_sq*res*dat.X;
      mu += paras.inv_sigma_sq_alpha * paras.f_mu;
      mu *= var;
      paras.alpha = arma::randn(mu.n_elem)*sqrt(var) + mu;
      
      paras.theta_alpha = High_to_low_vec(paras.alpha, basis.L, basis.Qlist,
                                          basis.region_idx, basis.L_idx); 

      // annealing on sigmasq_alpha
      paras.inv_sigma_sq_alpha = 1.0/(sigmasq_alpha_control.a*pow(sigmasq_alpha_control.b+iter,
                                          sigmasq_alpha_control.gamma));
      
    }

    void update_mu(){
      
        double inv_sigma_sq_mu = paras.inv_tau_sq_mu;
        
        arma::vec Vi = 1./(paras.inv_sigma_sq_alpha + inv_sigma_sq_mu*CAR.D);
        
        arma::vec Vi_sqrt = sqrt(Vi);
        arma::vec V0 = 1./(inv_sigma_sq_mu*CAR.D);
        arma::vec V0_inv = 1/V0;
        arma::vec V0_sqrt = sqrt(V0);
        
        arma::vec y_plus_lambda = paras.alpha + dat.lambda;
        arma::vec y_plus_lambda2 = y_plus_lambda%y_plus_lambda;
        arma::vec y_minus_lambda = paras.alpha - dat.lambda;
        arma::vec y_minus_lambda2 = y_minus_lambda%y_minus_lambda;
        
        arma::vec y2 = paras.alpha%paras.alpha;
        
        arma::vec mu_pos = Vi % (paras.inv_sigma_sq_alpha*(y_plus_lambda) + 
          V0_inv%paras.neighbor_mean);
        arma::vec mu_neg = Vi % (paras.inv_sigma_sq_alpha*(y_minus_lambda) + 
          V0_inv%paras.neighbor_mean);
        arma::vec mu_zero = paras.neighbor_mean;

        
        // tn_mean, tn_var, tn_2moment, Z
        arma::mat pos_stats = truncated_normal_stats_vec(mu_pos, Vi_sqrt, dat.lambda, 1, 4);
        arma::mat neg_stats = truncated_normal_stats_vec(mu_neg,  Vi_sqrt, dat.lambda, -1, 4);
        arma::mat zero_stats = truncated_normal_stats_vec(mu_zero, V0_sqrt, dat.lambda,0,4);
        
        // get mixing prob
        arma::vec C_pos = V0_inv%paras.neighbor_mean%paras.neighbor_mean;
        arma::vec C_neg = C_pos; arma::vec C_zero = C_pos;
        
        C_pos += paras.inv_sigma_sq_alpha*y_plus_lambda2;
        C_neg += paras.inv_sigma_sq_alpha*y_minus_lambda2;
        C_zero += paras.inv_sigma_sq_alpha*y2;

        
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
          



    }

  void update_theta_zeta(){
    arma::mat res_star = dat.M_star - paras.theta_alpha*dat.X.t() - paras.theta_eta;
    
    for(int k=0; k<dat.m; k++){
      arma::uvec k_c = complement(k,k,dat.m);
      arma::vec var_k = 1/(paras.inv_sigma_sq*dat.C_l2(k) + 
        paras.inv_sigma_sq_zeta/basis.D_vec);
      
      arma::vec mu_k = paras.inv_sigma_sq*(res_star - 
                        paras.theta_zeta.cols(k_c)*dat.C.rows(k_c))*
                        arma::trans(dat.C.row(k)); // L by 1
      mu_k %= var_k;
      paras.theta_zeta.col(k) = arma::randn(basis.L,1)%sqrt(var_k) + mu_k;
      
    }

    // update sigma_zeta
    double a = 0.5*(basis.L*dat.m + 1);
    arma::mat temp = paras.theta_zeta.each_col()%(1.0/basis.D_vec);
    double E_quad =  arma::accu( temp % paras.theta_zeta );
    double b_sigma_zeta = 0.5* E_quad+ paras.inv_a_zeta;
    paras.inv_sigma_sq_zeta = arma::randg(arma::distr_param(a,1.0/b_sigma_zeta));
  };

  void update_theta_eta(){
    arma::mat res_star = dat.M_star - paras.theta_alpha*dat.X.t() - paras.theta_zeta*dat.C;
    arma::vec Var_theta_eta = 1/(paras.inv_sigma_sq + paras.inv_sigma_sq_eta/basis.D_vec);
    res_star.each_col() %= Var_theta_eta*paras.inv_sigma_sq;
    paras.theta_eta = arma::randn(Var_theta_eta.n_elem, dat.n);
    paras.theta_eta.each_col() %= sqrt(Var_theta_eta);
    paras.theta_eta += res_star;

    // update sigma_eta
    double a = 0.5*(basis.L*dat.n + 1);
    arma::mat temp = paras.theta_eta.each_col()%(1.0/basis.D_vec);
    double E_quad = arma::accu( temp % paras.theta_eta ) ;
    double b_sigma_eta = 0.5*E_quad+ paras.inv_a_eta;
    paras.inv_sigma_sq_eta = arma::randg(arma::distr_param(a,1.0/b_sigma_eta));
  };

  void update_SSE(){
    arma::mat res_star = dat.M_star - paras.theta_alpha*dat.X.t() - 
      paras.theta_zeta*dat.C -
      paras.theta_eta;
    paras.SSE = arma::accu(res_star%res_star);
  }

  void update_inv_sigma_sq(){
    double b_sigma = 0.5*paras.SSE + paras.inv_a;
    paras.inv_sigma_sq = arma::randg(arma::distr_param(0.5*(dat.n*basis.L+1),1.0/b_sigma));
    paras.inv_a = arma::randg(arma::distr_param(1.0,1.0/(1+paras.inv_sigma_sq)));

    paras.loglik = -0.5*dat.n*basis.L*log(2*arma::datum::pi) + 0.5*dat.n*basis.L*log(paras.inv_sigma_sq) - 
      0.5*paras.SSE*paras.inv_sigma_sq - 0.5*dat.n*basis.L;
  };

  void save_paras_sample(){
    if(iter > gibbs_control.burnin){
      if ((iter - gibbs_control.burnin)%gibbs_control.thinning==0){
        int mcmc_iter = (iter - gibbs_control.burnin)/gibbs_control.thinning;
        paras_sample.alpha.col(mcmc_iter) = paras.alpha;  
        paras_sample.f_mu.col(mcmc_iter) = paras.f_mu;
        paras_sample.IP.col(mcmc_iter) = paras.IP;
        paras_sample.theta_zeta.slice(mcmc_iter) = paras.theta_zeta;
        paras_sample.delta_rho.col(mcmc_iter) = paras.delta_rho;

        paras_sample.sigma_sq(mcmc_iter) = 1.0/paras.inv_sigma_sq;
        paras_sample.sigma_sq_eta(mcmc_iter) = 1.0/paras.inv_sigma_sq_eta;
        paras_sample.sigma_sq_zeta(mcmc_iter) = 1.0/paras.inv_sigma_sq_zeta;
        paras_sample.tau_sq_mu(mcmc_iter) = 1.0/paras.inv_tau_sq_mu;
        paras_sample.sigma_sq_alpha(mcmc_iter) = 1.0/paras.inv_sigma_sq_alpha;

        
      }
      
    }
    paras_sample.loglik(iter) = paras.loglik;
  }
  
  void run_Gibbs(){
    std::cout << "Image on scalar: running Gibbs " <<std::endl;
    initialize_paras_sample();
    int theta_eta_counter = 0;
    for(iter=0; iter<gibbs_control.total_iter; iter++){
      if(iter%gibbs_control.verbose==0){
        std::cout << "Image on scalar: iter: " << iter <<  " sigma_sq: "<<  1.0/paras.inv_sigma_sq;
        std::cout <<" mu_quad = "<<CAR.E_mu_quad<< std::endl;
      }
      update_alpha();
      update_mu();
      update_theta_zeta();
      if( (iter%gibbs_control.eta_freq) == 0){
        update_theta_eta();
        theta_eta_counter++;
        paras.theta_eta_mean += paras.theta_eta;
      }
      update_SSE();
      update_inv_sigma_sq();
      save_paras_sample();
    }
    paras.theta_eta_mean /= theta_eta_counter;
  }

  List get_gibbs_post_mean(){
    arma::vec alpha = arma::mean(paras_sample.alpha,1);
    arma::vec f_mu = arma::mean(paras_sample.f_mu,1);
    arma::vec IP = arma::mean(paras_sample.IP,1);
    arma::mat theta_zeta = arma::mean(paras_sample.theta_zeta,2);
    double sigma_sq = arma::mean(paras_sample.sigma_sq);
    double sigma_sq_eta = arma::mean(paras_sample.sigma_sq_eta );
    double sigma_sq_zeta = arma::mean(paras_sample.sigma_sq_zeta );
    double tau_sq_mu = arma::mean(paras_sample.tau_sq_mu );
    double sigma_sq_alpha = arma::mean(paras_sample.sigma_sq_alpha );
    arma::mat zeta = Low_to_high(theta_zeta, dat.p, basis.Qlist,
                                     basis.region_idx, basis.L_idx);
    arma::vec delta_rho = arma::mean(paras_sample.delta_rho,1);

    return Rcpp::List::create(Rcpp::Named("alpha") = alpha,
                            Rcpp::Named("f_mu") = f_mu,
                            Rcpp::Named("IP") = IP,
                            Rcpp::Named("delta_rho") = delta_rho,
                            Rcpp::Named("zeta") = zeta,
                            Rcpp::Named("sigma_sq") = sigma_sq,
                            Rcpp::Named("sigma_sq_eta") = sigma_sq_eta,
                            Rcpp::Named("sigma_sq_zeta") = sigma_sq_zeta,
                            Rcpp::Named("tau_sq_mu") = tau_sq_mu,
                            Rcpp::Named("sigma_sq_alpha") = sigma_sq_alpha);

  }

  List get_gibbs_sample(){
    return Rcpp::List::create(Rcpp::Named("alpha") = paras_sample.alpha,
                            Rcpp::Named("f_mu") = paras_sample.f_mu,
                            Rcpp::Named("IP") = paras_sample.IP,
                            Rcpp::Named("delta_rho") = paras_sample.delta_rho,
                            Rcpp::Named("theta_zeta") = paras_sample.theta_zeta,
                            Rcpp::Named("sigma_sq") = paras_sample.sigma_sq,
                            Rcpp::Named("sigma_sq_eta") = paras_sample.sigma_sq_eta,
                            Rcpp::Named("sigma_sq_zeta") = paras_sample.sigma_sq_zeta,
                            Rcpp::Named("tau_sq_mu") = paras_sample.tau_sq_mu,
                            Rcpp::Named("sigma_sq_alpha") = paras_sample.sigma_sq_alpha,
                            Rcpp::Named("loglik") = paras_sample.loglik);
  }

  List get_gibbs_control(){
    return List::create(Named("total_iter") = gibbs_control.total_iter,
                        Named("burnin") = gibbs_control.burnin,
                        Named("mcmc_sample") = gibbs_control.mcmc_sample,
                        Named("thinning") = gibbs_control.thinning,
                        Named("verbose") = gibbs_control.verbose,
                        Named("save_profile") = gibbs_control.save_profile,
                        Named("total_profile") = gibbs_control.total_profile);
  }
  

  List get_vb_post_mean(){
    return List::create(Named("alpha") = vb_paras.E_alpha,
                        Named("IP") = vb_paras.E_IP,
                        Named("f_mu") = vb_paras.E_f_mu,
                        Named("Var_alpha") = vb_paras.Var_alpha,
                        Named("theta_eta") = vb_paras.E_theta_eta,
                        Named("theta_eta_mean") = vb_paras.theta_eta_mean,
                        Named("theta_zeta") = vb_paras.E_theta_zeta,
                        Named("sigma_sq") = 1/vb_paras.E_inv_sigma_sq,
                        Named("sigma_alpha_sq") = 1/vb_paras.E_inv_sigma_sq_alpha,
                        Named("sigma_zeta_sq") = 1/vb_paras.E_inv_sigma_sq_zeta,
                        Named("sigma_eta_sq") = 1/vb_paras.E_inv_sigma_sq_eta);
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
    List output_list;
    output_list = List::create(Named("iters") = iters,
                              Named("E_sigma_sq") = 1/vb_profile.E_inv_sigma_sq.rows(0,actual_profile_iter-1),
                              Named("E_sigma_sq_eta") = 1/vb_profile.E_inv_sigma_sq_eta.rows(0,actual_profile_iter-1),
                              Named("E_sigma_sq_zeta") = 1/vb_profile.E_inv_sigma_sq_zeta.rows(0,actual_profile_iter-1),
                              Named("E_tau_sq_mu") = 1/vb_profile.E_inv_tau_sq_mu.rows(0,actual_profile_iter-1),
                              Named("E_sigma_sq_alpha") = 1/vb_profile.E_inv_sigma_sq_alpha.rows(0,actual_profile_iter-1),
                              Named("E_alpha") = vb_profile.E_alpha.cols(0,actual_profile_iter-1),
                              Named("E_delta_rho") = vb_profile.E_delta_rho.cols(0,actual_profile_iter-1),
                              Named("E_f_mu") = vb_profile.E_f_mu.cols(0,actual_profile_iter-1),
                              Named("E_IP") = vb_profile.E_IP.cols(0,actual_profile_iter-1));
    return output_list;
  }
  
  List get_vb_control(){
    return List::create(Named("max_iter")= vb_control.max_iter,
                        Named("para_diff_tol") = vb_control.para_diff_tol,
                        Named("verbose") = vb_control.verbose,
                        Named("save_profile") = vb_control.save_profile,
                        Named("total_profile") = vb_control.total_profile);
  };

  int get_iter(){
    return iter;
  };
  
  List get_test_output(){
    return List::create(Named("Image_on_scalar") = test_output);
  };
  
  
  
};

//' @title Image on scalar regression
//' @description
//' Scalar on Image regression using the sparse-mean prior
//' @name Image_on_scalar
//' @param y outcome
//' @param X vector covariate
//' @param M Matrix of functional images
//' @param lambda thresholding parameter
//' @param rho scaling parameter in CAR model
//' @param B matrix of covariance neighborhood
//' @param in_Sigma_inv
//' @param in_D
//' @param init_paras
//' @param batch_control
//' @param method
//' @useDynLib STCAR, .registration=TRUE
//' @export
// [[Rcpp::export]]
 List IonS_CAVI_rho(arma::mat& M, arma::vec& X, arma::mat& C, 
                      double lambda, List& basis,
                      double rho, const arma::sp_mat& B,
                      const arma::sp_mat& W,
                      const arma::sp_mat& in_Sigma_inv,
                      const arma::vec& in_D,
                      CharacterVector method,
                      List& init_paras,
                      List& sigmasq_step_controls,
                      const arma::vec& in_delta_rho,
                      List& SGD_controls,
                      double initial_sigma_sq = 1,
                      double initial_tau_mu_sq = 0.01,
                      double initial_sigma_alpha_sq = 1,
                      double initial_sigma_zeta_sq = 1,
                      double initial_sigma_eta_sq = 1,
                      int eta_freq = 1,
                      int max_iter = 1000,
                      int mcmc_sample = 1000,
                      int burnin = 1000,
                      int thinning = 1,
                      int begin_f_alpha = 0,
                      int f_alpha_interval = 1,
                      double paras_diff_tol = 1e-6,
                      int verbose = 5000,
                      int save_profile = 1,
                      bool display_progress = false){
   
   arma::wall_clock timer;

   timer.tic();
   IonS_standalone model;
   
   
   Rcout<<"Image on scalar...1"<<std::endl;
   model.load_data(X,M,C,lambda);
   model.set_method(method);
   Rcout<<"Image on scalar...2"<<std::endl;
   model.load_basis(basis);
   Rcout<<"Image on scalar...3"<<std::endl;
   model.load_CAR(rho, B, W, in_Sigma_inv, in_D);
   Rcout<<"Image on scalar...4"<<std::endl;
   model.display_progress = display_progress;

   if(model.method==0){
    model.set_gibbs_control(mcmc_sample,
                            burnin,
                            thinning,
                            verbose,
                            save_profile,
                            begin_f_alpha,
                            eta_freq);
   }else{
    model.set_vb_control(max_iter,
                        paras_diff_tol,
                        verbose,
                        save_profile,
                        begin_f_alpha);
   }
   
   model.set_sigmasq_alpha_annealing_control(sigmasq_step_controls);
  if(model.method == 2){
    model.set_SGD_controls(SGD_controls);
  }
  std::cout << "set control done" << std::endl;
   
   model.set_paras_initial_values(init_paras,
                                  in_delta_rho,
                                  initial_sigma_sq, 
                                  initial_sigma_alpha_sq,
                                  initial_sigma_zeta_sq,
                                  initial_sigma_eta_sq,
                                  initial_tau_mu_sq);
   Rcout<<"Image on scalar...6"<<std::endl;
   
   Rcout<<"model.method = "<<model.method<<std::endl;
   if(model.method==1){
     model.run_CAVI(f_alpha_interval); 
     Rcout<<"Image on scalar...cavi, done."<<std::endl;
   }else if(model.method ==2){
    // run sgd
    model.run_SGD(f_alpha_interval);
    Rcout<<"Image on scalar...sgd, done."<<std::endl;
   }else if(model.method ==0){
      model.run_Gibbs();
      Rcout<<"Image on scalar...gibbs, done."<<std::endl;
   }
   
   double elapsed = timer.toc();
   
   
   List output;
   if(model.method == 0){
    output = List::create(Named("post_mean") = model.get_gibbs_post_mean(),
                          Named("iter") = model.get_iter(),
                          Named("trace") = model.get_gibbs_sample(),
                          Named("gibbs_control") = model.get_gibbs_control(),
                          Named("test_output") = model.get_test_output(),
                          Named("method") = method,
                          Named("elapsed") = elapsed);
   }else{
    output = List::create(Named("post_mean") = model.get_vb_post_mean(),
                          Named("iter") = model.get_iter(),
                          Named("trace") = model.get_vb_trace(),
                          Named("vb_control") = model.get_vb_control(),
                          Named("test_output") = model.get_test_output(),
                          Named("method") = method,
                          Named("elapsed") = elapsed);
   
   }
   
   
   return output;
   
 }