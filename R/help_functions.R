library(ggplot2)
library(viridis)
library(BayesGPfit)
library(RSpectra)
library(ggpubr)

#' Generate grids for 4 regions
#'@param side side length of the square grid
#'@param d dimension
#'@export
gen_grids_df_4region = function(side,d = 2){
  num_region = 4
  region_idx = vector("list",num_region)
  grids = GP.generate.grids(d=d,num_grids=side)
  grids_df = as.data.frame(grids)
  
  idx_mat = matrix(1:(side*side),ncol = num_region)
  idx_matr = matrix(1:(side*side),ncol = side)
  p_length = NULL
  seq_idx = cbind(1:(side/2), (1+side/2):side)
  region_idx[[1]] = intersect(idx_matr[seq_idx[,1],],idx_mat[,1:2])
  region_idx[[2]] = intersect(idx_matr[seq_idx[,2],],idx_mat[,1:2])
  region_idx[[3]] = intersect(idx_matr[seq_idx[,1],],idx_mat[,3:4])
  region_idx[[4]] = intersect(idx_matr[seq_idx[,2],],idx_mat[,3:4])
  
  return(region_idx)
}


plot_ROC = function(ip_list, beta, len=20){
  thresh_grid = seq(0,1,length.out = len)
  FPR_all = NULL; TPR_all = NULL
  
  for(i in 1:length(ip_list)){
    ip = ip_list[[i]]
    TPR = rep(NA,len)
    FPR = rep(NA,len)
    
    if(names(ip_list)[i] == "MUA"){
      p_adj = p.adjust(ip,"BH")
      p_adj_thresh = quantile(p_adj,probs = thresh_grid)
      
      for(l in 1:len){
        res = as.matrix(table(p_adj <= p_adj_thresh[l], beta!=0))
        TPR[l] = res["TRUE","TRUE"]/sum(res[,"TRUE"]) # TP/P
        FPR[l] = res["TRUE","FALSE"]/sum(res[,"FALSE"]) # FP/N
      }
      TPR = c(TPR,1)
      FPR = c(FPR,1)
    }else{
      # ip_thresh_grid = quantile(ip,probs = thresh_grid)
      ip_thresh_grid = thresh_grid
      for(l in 1:len){
        thresh = ip_thresh_grid[l]
        res = as.matrix(table(ip>=thresh, beta!=0))
        if(! "TRUE" %in% rownames(res)){
          res = as.matrix(rbind(res,c(0,0)))
          rownames(res) = c("FALSE","TRUE")
        }else if(! "FALSE" %in% rownames(res)){
          res = as.matrix(rbind(c(0,0),res))
          rownames(res) = c("FALSE","TRUE")
        }
        TPR[l] = res["TRUE","TRUE"]/sum(res[,"TRUE"]) # TP/P
        FPR[l] = res["TRUE","FALSE"]/sum(res[,"FALSE"]) # FP/N
        
      }
      TPR = c(1,TPR)
      FPR = c(1,FPR)
    }
    
    TPR_all = cbind(TPR_all, TPR)
    FPR_all = cbind(FPR_all, FPR)
    
  }
  
  return(list(FPR = FPR_all,TPR= TPR_all))
  
}


get_size = function(x){
  return(format(object.size(x), units = "auto"))
}

#' plot_img
#' 
#' @param img A vector of input image
#' @param grids_df A data frame to indicate the position of pixels
#' @import ggplot2
#' @import viridis
plot_img = function(img, grids_df,title="img",col_bar = NULL, legend = T){
  if(length(img)!=dim(grids_df)[1]){
    print("dimension of img and grids do not match!")
  }else{
    g = ggplot(grids_df, aes(x=x1,y=x2)) +
      geom_tile(aes(fill = img)) +
      scale_fill_viridis_c(limits = col_bar, oob = scales::squish)+
      ggtitle(title)+
      theme(plot.title = element_text(size=20),legend.text=element_text(size=10))
    if(!legend){
      g = g + theme(legend.position="none")
    }
  }
  
  g
  
}


plot_multi_img = function(list_of_image, grids_df, n_img_per_row = 3,
                          col_bar = NULL, layout_matrix=NULL, font_size=20,
                          legend_position = "right"){
  n_img = length(list_of_image)
  if(is.null(col_bar)){
    col_bar = range(unlist(list_of_image))
  }
  n_row_img = ceiling(n_img/n_img_per_row)
  all_image_p = vector("list", n_img)
  for(i in 1:n_img){
    all_image_p[[i]] <-  local({
      i <- i
      ggplot(grids_df, aes(x=x1,y=x2)) +
        geom_tile(aes(fill = list_of_image[[i]] )) +
        ggtitle(names(list_of_image)[i])+
        theme(plot.title = element_text(size=font_size),
              legend.text = element_text(size=font_size*0.7))+
        ggplot2::scale_fill_gradient2(low = "blue",
                                      mid="white",
                                      high="red",
                                      midpoint = 0, limits = col_bar,
                                      name = "Value")
    })
    
  }
  
  ggpubr::ggarrange(plotlist = all_image_p, 
                    ncol=n_img_per_row, 
                    nrow=n_row_img, 
                    common.legend = TRUE, legend=legend_position)
  
  
  
}

plot_multi_img_new = function(list_of_image, grids_df, n_img_per_row = 3,col_bar = c(0,1)){
  n_img = length(list_of_image)
  n_row_img = ceiling(n_img/n_img_per_row)
  all_image = vector("list", n_img)
  names(all_image) =  names(list_of_image)
  for(i in 1:n_img){
    img = list_of_image[[i]]
    title = names(list_of_image)[i]
    g = ggplot(grids_df, aes(x=x1,y=x2)) +
      geom_tile(aes(fill = img)) +
      scale_fill_viridis_c(limits = col_bar, oob = scales::squish)+
      ggtitle(title)+
      theme(plot.title = element_text(size=20),legend.text=element_text(size=10))+
      theme(legend.position = "none")
    all_image[[i]] = g
    # all_image[[i]] = ggplotGrob(g)
  }
  
  layout_matrix = matrix(NA, nrow = n_row_img, ncol = n_img_per_row)

  layout_matrix[1:n_img] = 1:n_img
  
  ggarrange(
    all_image,
    nrow = ceiling(n_img/n_img_per_row),
    ncol = n_img_per_row, byrow = TRUE,
    common.legend = TRUE
    # layout_matrix = layout_matrix
  )
  
  # Extract the legend
  legend <- get_legend(all_image[[1]] + theme(legend.position = "bottom"))
  
  n_row = ceiling(n_img / n_img_per_row)
  # Arrange plots in a grid with a row-first layout
  combined_plot <- ggarrange(plotlist = all_image, nrow = n_row, ncol = n_img_per_row, byrow = TRUE)
  # grob <- ggplotGrob(combined_plot)
  
  # Add the overall legend
  combined_plot <- ggarrange(combined_plot, legend, nrow = n_row, heights = c(10, 1))
  
  combined_plot
  
  # gridExtra::grid.arrange(
  #   grobs = all_image,
  #   nrow = ceiling(n_img/n_img_per_row)
  #   # layout_matrix = layout_matrix
  # )
  
}


Soft_threshold = function(x,lambda){
  return( (x-sign(x)*lambda)*(abs(x)>lambda))
}


neighbor_sparse_mat = function( grids, scale = 2, rho = 0.3,
                                bandwidth = NULL,if_ind = F,
                                band_dist = NULL,if_beep=F){
  n = dim(grids)[1]
  m = Matrix::Matrix(0, n, n, sparse = TRUE); 
  if(is.null(bandwidth)){
    bandwidth = n/2
  }
  if(is.null(band_dist)){
    band_dist = 1000
  }
  pb = txtProgressBar(min = 1, max = dim(grids)[1]-1, initial = 1) 
  B = Matrix::Matrix(0, n, n, sparse = TRUE); 
  m_sum = rep(0,n)
  Sigma_inv = Matrix::Matrix(0, n, n, sparse = TRUE); 
  for(i in 1:(dim(grids)[1]-1)){
    setTxtProgressBar(pb,i)
    rg = min(n, i+bandwidth)
    if(rg > i+1){
      dist_i = apply(grids[(i+1):rg,], 1, function(x,x_loc){sqrt(sum((x-x_loc)^2))},
                     grids[i,])
    }else{
      dist_i = sqrt(sum((grids[(i+1):rg,] - grids[i,])^2))
    }
    
    neighbor_idx = ((i+1):rg)[dist_i<band_dist]
    m[i,neighbor_idx] = exp(-dist_i[dist_i<band_dist]/scale)
    m[neighbor_idx,i] = m[i,neighbor_idx]
    m_sum[i] = sum(m[i,])
    B[i,] = m[i,]/m_sum[i]
    Sigma_inv[i,i] = m_sum[i]*(1-rho*B[i,i] )
    Sigma_inv[i,-i] = -m_sum[i]*(rho*B[i,-i] )
  }
  m_sum[n] = sum(m[n,])
  B[n,] = m[n,]/m_sum[n]
  Sigma_inv[n,n] = m_sum[n]*(1-rho*B[n,n] )
  Sigma_inv[n,-n] = -m_sum[n]*(rho*B[n,-n] )
  
  close(pb)
  if(if_beep){
    beepr::beep()
  }
  D_vec = 1/m_sum
  return(list(B = B, D = D_vec, rho = rho, inv_Sigma = Sigma_inv))
}


neighbor_sparse_mat_RANN = function( grids, scale = 2, rho = 0.3,
                                      bandwidth = NULL,if_beep=F){
  n = dim(grids)[1]
  m = Matrix::Matrix(0, n, n, sparse = TRUE); 
  if(is.null(bandwidth)){
    bandwidth = n/2
  }
  
  
  m_sum = rep(0,n)
  Sigma_inv = Matrix::Matrix(0, n, n, sparse = TRUE); 
  
  kdtree <- RANN::nn2(grids, k = bandwidth+1) # l2 dist
  # Create a sparse matrix
  W <- Matrix::Matrix(0, nrow = n, ncol = n, sparse = TRUE)
  B = Matrix::Matrix(0, n, n, sparse = TRUE); 
  
  
  pb <- txtProgressBar(min = 1, max = n, initial = 1)
  W_sum = rep(0,n)
  print("begin W creation ...")
  for(i in 1:n){
    setTxtProgressBar(pb, i)
    neighbors <- kdtree$nn.idx[i, ]
    distances <- kdtree$nn.dists[i, ]
    valid_indices <- distances > 0
    valid_neighbors <- neighbors[valid_indices]
    valid_distances <- distances[valid_indices]
    W[i, valid_neighbors] <- exp(-valid_distances/scale)
    W_sum[i] = sum(W[i,])
    row_indices = which(W[i,]!=0)
    B[i, row_indices] = W[i, row_indices]/W_sum[i]
    Sigma_inv[i, row_indices] = W_sum[i] - rho*W[i, row_indices]
  }
  
  
  # })
  close(pb)
  if(if_beep){
    beepr::beep()
  }
  
  return(list(B = B, D = W_sum, rho = rho, inv_Sigma = Sigma_inv, W=W))
}

neighbor_mat = function( grids, method = "l2",scale = 2, rho = 0.3,
                         bandwidth = NULL,
                         band_dist = NULL,if_beep=F){
  # m = diag(1, n)
  # if(type == "banded"){
  #   for(r in 1:rg){
  #     m[abs(row(m) - col(m)) == r] = 1/(r+1)
  #   }
  # }
  # if(type == "AR1"){
  #   exponent = abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
  #                     (1:n - 1))
  #   m = rho^exponent
  # }
  # m = sweep(m,1,apply(m,1,sum),"/")
 
  
  n = dim(grids)[1]
  m = diag(0,n); m_scaled = m
  if(is.null(bandwidth)){
    bandwidth = n/2
  }
  if(is.null(band_dist)){
    band_dist = 1000
  }
  pb = txtProgressBar(min = 1, max = dim(grids)[1]-1, initial = 1) 
  for(i in 1:(dim(grids)[1]-1)){
    setTxtProgressBar(pb,i)
    rg = min(n, i+bandwidth)
    if(rg > i+1){
      dist_i = apply(grids[(i+1):rg,], 1, function(x,x_loc){sqrt(sum((x-x_loc)^2))},
        grids[i,])
    }else{
      dist_i = sqrt(sum((grids[(i+1):rg,] - grids[i,])^2))
    }
    
    neighbor_idx = ((i+1):rg)[dist_i<band_dist]
    # m[i,neighbor_idx] = 1/(dist_i[dist_i<band_dist]/scale)
    m[i,neighbor_idx] = exp(-dist_i[dist_i<band_dist]/scale)
    # m[i,neighbor_idx] = exp(-dist_i[dist_i<band_dist]/scale)
    # m_scaled[i,neighbor_idx] = m[i,neighbor_idx]/sum(m[i,neighbor_idx])
  }
  close(pb)
  if(if_beep){
    beepr::beep()
  }
  
  B = m + t(m) 
  W = sweep(B,1,apply(B,1,sum),"/")
  D_vec = 1/apply(B,1,sum)
  D = diag(D_vec)
  isSymmetric(W)
  Sigma_inv = solve(D,diag(rep(1,n))-rho*W)

  return(list(B = W, D = D_vec, rho = rho, inv_Sigma = Sigma_inv))
}

generate_SonI_data = function(beta_img, n,q=2,sigma_M = 0.1,
                              sigma_C = 1,include_Confounder = T,
                                   sigma_Y = 0.1, SNR = NULL){
  
  p = length(beta_img)
  X = rnorm(n)
  C = matrix(rnorm(n*q, sd = sigma_C),q,n) # q by n
  # zeta_y = sample(-10:10,q)
  zeta_y = runif(q)
  zeta_m = matrix(rnorm(p*q),p,q)
  eta = matrix(rnorm(p*n),p,n)
  # gamma = sample(-10:10,1)
  gamma = runif(1)
  
  
  # generate image
  # M = alpha_img %*% t(X) + zeta_m %*% C + eta + matrix(rnorm(p*n, sd = sigma_M),p,n)
  M = eta + matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  if(!is.null(SNR)){
    sigma_Y_sq  = var(c(t(beta_img) %*% M))/SNR
    sigma_Y = sqrt(sigma_Y_sq)
    print(paste("Using SNR = ",SNR,", sigma_Y=",sigma_Y))
  }
  if(!include_Confounder){
    gamma = 0
    zeta_y = rep(0,q)
  }
  Y = t(beta_img) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + 
    rnorm(n, sd = sigma_Y)
  
  
  # true_params
  true_params = list(beta = beta_img,  zeta_y = zeta_y,
                     eta = eta, gamma = gamma, sigma_M = sigma_M, 
                     sigma_Y = sigma_Y)
  
  return(list(Y = Y, M = t(M), X = X, C = C, true_params = true_params))
}



generate_IonS_data = function(alpha_img, basis_sq,
                              n,
                              q=2,sigma_M = 0.1){
  
  p = length(alpha_img)
  X = rnorm(n)
  C = matrix(rnorm(n*q),q,n) # q by n
  
  L = length(unlist(basis_sq$Phi_D))
  theta_zeta_m = matrix(runif(L*q),L,q)*0.5
  
  
  zeta_m = Low_to_high(theta_zeta_m, p, basis_sq$Phi_Q,
                       basis_sq$region_idx_cpp, basis_sq$L_idx_cpp)
  theta_eta = matrix(rnorm(L*n),L,n)
  eta = Low_to_high(theta_eta, p, basis_sq$Phi_Q,
                    basis_sq$region_idx_cpp, basis_sq$L_idx_cpp)
  
  
  # generate image
  M = alpha_img %*% t(X) + zeta_m %*% C + eta + matrix(rnorm(p*n, sd = sigma_M),p,n) # p by n
  
  
  # true_params
  true_params = list(alpha = alpha_img, zeta_m = zeta_m,
                     theta_eta = theta_eta, eta = eta,
                     sigma_M = sigma_M)
  
  return(list(M = M, X = X, C = C, true_params = true_params))
  
}

generate_mediation_data = function(beta_img, alpha_img, basis_sq,n,q=2,sigma_M = 0.1,
                                   sigma_Y = 0.1){
  
  p = length(beta_img)
  X = rnorm(n)
  C = matrix(rnorm(n*q),q,n) # q by n
  
  L = length(unlist(basis_sq$Phi_D))
  zeta_y = runif(q)
  theta_zeta_m = matrix(runif(L*q),L,q)*0.5
  
  
  zeta_m = Low_to_high(theta_zeta_m, p, basis_sq$Phi_Q,
                       basis_sq$region_idx_cpp, basis_sq$L_idx_cpp)
  theta_eta = matrix(rnorm(L*n),L,n)
  eta = Low_to_high(theta_eta, p, basis_sq$Phi_Q,
                    basis_sq$region_idx_cpp, basis_sq$L_idx_cpp)
  # gamma = sample(-10:10,1)
  gamma = runif(1)
  
  
  # generate image
  M = alpha_img %*% t(X) + zeta_m %*% C + eta + matrix(rnorm(p*n, sd = sigma_M),p,n)
  
  # generate outcome
  Y = t(beta_img) %*% M + t(c(gamma, zeta_y)) %*% rbind(t(X),C) + rnorm(n, sd = sigma_Y)
  
  # true_params
  true_params = list(beta = beta_img, alpha = alpha_img, zeta_y = zeta_y, zeta_m = zeta_m,
                     theta_eta = theta_eta, gamma = gamma, sigma_M = sigma_M, sigma_Y = sigma_Y)
  
  return(list(Y = Y, M = t(M), X = X, C = C, true_params = true_params))
}


#' simulate_round_image
#' 
#' @import BayesGPfit
simulate_round_image = function(center_shift = c(0,0),lambda = 0.1,side = 30, 
                                range = c(0,1)){
  n_sqrt = side
  n = n_sqrt*n_sqrt
  grids = GP.generate.grids(d=2L,num_grids=n_sqrt)
  center = apply(grids,2,mean) + center_shift
  rad = apply(grids,1,function(x){sum((x-center)^2)})
  inv_rad = 2-rad
  inv_rad_ST = Soft_threshold(inv_rad,1.2)
  f_mu = Soft_threshold(log(inv_rad_ST^2+1),lambda)
  
  y = f_mu
  nonzero = y[abs(y)>0]
  a = range[1]; b = range[2]
  nonzero_mapped = (nonzero-min(nonzero))/(max(nonzero)-min(nonzero))*(b-a) + a
  y[abs(y)>0] = nonzero_mapped
  
  grids_df = as.data.frame(grids)
  
  
  return(list(img = y, grids_df = grids_df))
}

#' simulate_triang_image
#' 
#' @import BayesGPfit
simulate_triang_image = function(center_shift = c(0,0),side = 30, 
                                 rad = 0.5, radiance_bool = T,
                                range = c(0,1)){
  n_sqrt = side
  n = n_sqrt*n_sqrt
  grids = GP.generate.grids(d=2L,num_grids=n_sqrt)
  center = apply(grids,2,mean) + center_shift
  
  
  bool1 = 1*(grids[,2] > -rad/2)
  bool2 = 1*(grids[,2] < -sqrt(3)*grids[,1]+rad)
  bool3 = 1*(grids[,2] < sqrt(3)*grids[,1]+rad)
  
  beta = bool1*bool2*bool3

  if(radiance_bool){
    radiance = apply(grids,1,function(x){sum((x-center)^2)})
    inv_rad = 2-radiance
    inv_rad_ST = Soft_threshold(inv_rad,1.2)
    beta = beta*inv_rad_ST
  }
  
  
  grids_df = as.data.frame(grids)
  
  
  return(list(img = beta, grids_df = grids_df))
}

reverse_ST = function(x,lambda){
  y = x
  y[x>0] = x[x>0]+lambda
  y[x<0] = x[x<0]-lambda
  y
}

plot_vb = function(vb){
  model_name = deparse(substitute(vb))
  par(mfrow=c(2,3))
  plot(vb$trace$ELBO,main = paste(model_name, ":ELBO") )
  plot(vb$trace$E_beta[1,], main = paste(model_name,":E_beta[1]") )
  plot(vb$trace$E_gamma[1,], main = paste(model_name,":E_gamma[1]") )
  plot(vb$trace$sparse_mean$E_f_beta[1,], main = paste(model_name,":E_f_beta[1]") )
  plot(vb$post_mean$beta,datsim$true_params$beta, main=paste(model_name,":beta vs truth") ); abline(0,1,col="red")
  plot(vb$post_mean$gamma,init_true$gamma, main=paste(model_name,":gamma vs truth") ); abline(0,1,col="red")
  par(mfrow=c(1,1))
  
}

plot_vb.IonS = function(vb){
  model_name = deparse(substitute(vb))
  par(mfrow=c(2,3))
  plot(vb$trace$ELBO,main = paste(model_name, ":ELBO") )
  plot(vb$trace$E_alpha[1,], main = paste(model_name,":E_alpha[1]") )
  plot(vb$trace$E_sigma_sq_alpha, main = paste(model_name,":E_sigma_sq_alpha"))
  plot(vb$trace$E_sigma_sq_eta, main = paste(model_name,":E_sigma_sq_eta"))
  plot(vb$post_mean$alpha, datsim$true_params$alpha,
       main = paste(model_name,":alpha vs truth") );abline(0,1,col="red")
  plot(vb$post_mean$f_alpha, Soft_threshold(datsim$true_params$alpha, in_lambda),
       main = paste(model_name,":f_alpha vs truth") );abline(0,1,col="red")
  par(mfrow=c(1,1))
  
}

plot_gs = function(gibbs){
  par(mfrow = c(2,3))
  plot(gibbs$trace$loglik, main = "gibbs:loglik")
  plot(gibbs$mcmc$beta[1,], main = "gibbs:beta[1]")
  plot(gibbs$mcmc$gamma[1,], main = "gibbs:gamma[1]")
  plot(gibbs$mcmc$f_beta[1,], main = "gibbs:f_beta[1]")
  plot(gibbs$post_mean$beta, datsim$true_params$beta,
       main="gibbs:beta vs truth");abline(0,1,col="red")
  plot(gibbs$post_mean$gamma,init_true$gamma, main = "gibbs:gamma vs truth");abline(0,1,col="red")
  par(mfrow = c(1,1))
}

plot_gs.IonS = function(gibbs){
  par(mfrow = c(2,3))
  plot(gibbs$trace$loglik, main = "gibbs:loglik")
  plot(gibbs$mcmc$alpha[1,], main = "gibbs:alpha[1]")
  plot(gibbs$mcmc$f_alpha[1,], main = "gibbs:f_alpha[1]")
  plot(gibbs$mcmc$sigma_sq, main = "gibbs: sigma_M")
  plot(gibbs$mcmc$sigma_eta_sq, main = "gibbs: sigma_eta")
  # plot(gibbs$mcmc$gamma[1,], main = "gibbs:gamma[1]")
  # plot(gibbs$mcmc$f_beta[1,], main = "gibbs:f_beta[1]")
  plot(gibbs$post_mean$alpha, datsim$true_params$alpha,
       main="gibbs:alpha vs truth");abline(0,1,col="red")
  # plot(gibbs$post_mean$gamma,init_true$gamma, main = "gibbs:gamma vs truth");abline(0,1,col="red")
  par(mfrow = c(1,1))
}

FDR = function(active_region, true_region){
  sum(active_region!=0 & true_region==0)/sum(active_region!=0)
}
Precision = function(active_region, true_region){
  mean(I(active_region!=0) == I(true_region!=0))
}
Power = function(active_region, true_region){
  sum(active_region !=0 & true_region!=0)/sum(true_region!=0)
}

adapt_selection = function(IP, truth, method="autoFDR", tune_step = 0.05,
                           max_iter = 100, thresh_begin = 0.5){
  total = length(IP)
  selection = rep(0,total)
  
  if(method == "proportion"){
    total = length(truth)
    truth_bin = 1*(abs(truth)>0)
    total_pos = sum(truth_bin)
    PIP_order = order(IP)[total-(1:total_pos)]
    selection[PIP_order] = 1
  }
  
  if(method == "autoFDR"){
    thresh = thresh_begin
    fdr_target = 0.1
    for(i in 1:max_iter){
      mapping = 1*(IP>thresh)
      fdr = FDR(mapping, truth)
      print(paste("fdr=",fdr,"thresh=",thresh))
      if(is.na(fdr)){
        print("fdr=NA, target FDR is too small")
        thresh = thresh*(1-tune_step)
        mapping = 1*(IP>thresh)
        fdr = FDR(mapping, truth)
        print(paste("Use current fdr=",fdr,"thresh=",thresh))
        break
      }
      if(fdr<=fdr_target){
        break
      }
      thresh = thresh*(1+tune_step)
      if(thresh>1){
        print("New thresh>1, keep thresh at the current value and return result.")
        break
      }
    }# end for
    selection = mapping
  }
  
  selection
}

GP.simulate.curve.fast.new = function(x,poly_degree,a,b,
                                      center=NULL,scale=NULL,max_range=6){
  
  x = cbind(x)
  d = ncol(x)
  
  if(is.null(center)){
    center = apply(x,2,mean)
  }
  c_grids = t(x) - center
  if(is.null(scale)){
    max_grids =pmax(apply(c_grids,1,max),-apply(c_grids,1,min))
    scale=as.numeric(max_grids/max_range)
  }
  
  work_x = GP.std.grids(x,center=center,scale=scale,max_range=max_range)
  Xmat = GP.eigen.funcs.fast(grids=work_x,
                             poly_degree =poly_degree,
                             a =a ,b=b)
  lambda = GP.eigen.value(poly_degree=poly_degree,a=a,b=b,d=d)
  betacoef = rnorm(ncol(Xmat),mean=0,sd=sqrt(lambda))
  f = Xmat%*%betacoef
  return(list(f=f,x=x,work_x=work_x, eigen.func = Xmat, eigen.value = lambda))
}

generate_sq_basis = function(grids, region_idx_list,poly_degree_vec,a = 0.01, b=10, poly_degree=20,
                             show_progress=FALSE){
  num_block = length(region_idx_list)
  Phi_D = vector("list",num_block)
  Phi_Q = vector("list",num_block)
  Lt = NULL; pt = NULL
  for(i in 1:num_block){
    if(show_progress){
      print(paste("Computing basis for block ",i))
    }
    GP = GP.simulate.curve.fast.new(x=grids[region_idx_list[[i]],], a=a ,b=b,poly_degree=poly_degree) # try to tune b, increase for better FDR
    K_esq = GP$eigen.func
    K_QR = qr(K_esq)
    Phi_Q[[i]] = qr.Q(K_QR)
    Phi_D[[i]] = GP$eigen.value
    Lt = c(Lt, length(Phi_D[[i]]))
    pt = c(pt, dim(Phi_Q[[i]])[1])
  }
  return(list(Phi_D = Phi_D,
              region_idx_block = region_idx_list,
              Phi_Q = Phi_Q,L_all = Lt,p_length=pt))
}

matern_kernel = function(x,y,nu,l=1){
  d = sqrt(sum((x-y)^2))/l
  y = 2^(1-nu)/gamma(nu)*(sqrt(2*nu)*d)^nu*besselK(sqrt(2*nu)*d,nu)
  return(y)
}
#‘ Generate a matern basis
#' @importFrom RSpectra eigs_sym
generate_matern_basis2 = function(grids, region_idx_list, L_vec,scale = 2,nu = 1/5,
                                  show_progress = FALSE){
  if(nu=="vec"){
    nu_vec = region_idx_list["nu_vec"]
  }
  num_block = length(region_idx_list)
  Phi_D = vector("list",num_block)
  Phi_Q = vector("list",num_block)
  Lt = NULL; pt = NULL
  for(i in 1:num_block){
    if(show_progress){
      print(paste("Computing basis for block ",i))
    }
    p_i = length(region_idx_list[[i]])
    kernel_mat = matrix(NA,nrow = p_i, ncol=p_i)
    for(l in 1:p_i){
      if(nu=="vec"){
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu_vec[i],l=scale)
      }else{
        kernel_mat[l,] = apply(grids[region_idx_list[[i]],],1,matern_kernel,y=grids[region_idx_list[[i]],][l,],nu = nu,l=scale)
      }
    }
    diag(kernel_mat) = 1
    K = eigs_sym(kernel_mat,L_vec[i])
    K_QR = qr(K$vectors)
    Phi_Q[[i]] = qr.Q(K_QR )
    Phi_D[[i]] = K$values
    Lt = c(Lt, length(Phi_D[[i]]))
    pt = c(pt, dim(Phi_Q[[i]])[1])
  }
  return(list(Phi_D = Phi_D,
              region_idx_block = region_idx_list,
              Phi_Q = Phi_Q,L_all = Lt,p_length=pt))
}

plot_cavi = function(cavi_rho, init_true){
  par(mfrow = c(2,3))
  plot(cavi_rho$trace$E_sigma_sq,main="E_sigma_sq")
  plot(cavi_rho$trace$E_sigma_sq_beta,main="E_sigma_sq_beta")
  plot(cavi_rho$trace$E_sigma_sq_gamma,main="E_sigma_sq_gamma")
  plot(cavi_rho$trace$E_tau_sq_mu,main="E_tau_sq_mu")
  plot(cavi_rho$post_mean$gamma, init_true$gamma,asp=1,main="gamma vs truth");abline(0,1)
  plot(cavi_rho$post_mean$beta, init_true$beta,asp=1,main="beta vs truth");abline(0,1)
  par(mfrow = c(1,1))
}