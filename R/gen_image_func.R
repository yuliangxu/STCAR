library(imager)


map_to_range = function(x,range=c(0,1)){
  (x-min(x))/(max(x)-min(x))*(range[2]-range[1])+range[1]
}

process_png = function(im, side, center_rad = NULL, if_noncenter = F,
                       value_range = c(0.2,1)){
  im_array = as.array(im[,,1,1])
  chosen = round(seq(1,dim(im_array)[1],length.out = side))
  im_array_lowdim = im_array[chosen,chosen]
  
  
  grids = create_img_coords(imgdim = c(side, side))
  center = apply(grids,2,mean)
  
  # add radiations
  # im_array_lowdim = map_to_range(im_array_lowdim,c(0,1))
  im_array_lowdim = map_to_range(im_array_lowdim,c(0,1))
  background_val = im_array_lowdim[1]
  cutoff = im_array_lowdim[side/2,side/2]
  if(if_noncenter){
    cutoff = quantile(c(im_array_lowdim),probs = 0.1)
  }
  if(background_val > cutoff){
    im_array_lowdim = max(im_array_lowdim)-im_array_lowdim
  }
  
  background_val = im_array_lowdim[1]
  center_value = im_array_lowdim[side/2,side/2]
  rad_support = which(im_array_lowdim > background_val)
  
  if(is.null(center_rad)){
    center_rad = center
  }
  im_array_lowdim_rad = im_array_lowdim + 
    2*exp(-2*sqrt((grids[,1]-center_rad[1])^2 + (grids_df_loc[,2]-center_rad[2])^2))
  im_array_lowdim[rad_support] = im_array_lowdim_rad[rad_support]
  nonzero_support = which(abs(im_array_lowdim) >0)
  im_array_lowdim[nonzero_support] = map_to_range(im_array_lowdim[nonzero_support],
                                                  value_range)
  
  return(list(img = c(im_array_lowdim), grids = grids))
  
}


#'@import ggplot2
#'@import reshape2
#'@title Draw two-dimensional functions
#'@param f numeric matrix with each column representing the function values evaluated at grids
#'@param grids numeric matrix with number grid points 
#'@param names character vector where the length equals to the number of columns of f
#'@param nrow number of rows in the layout for more than one functions
#'@param ncol number of columns in the layout for more than one functions
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'n_sqrt <- 50
#'x <- as.matrix(expand.grid(seq(-1,1,length=n_sqrt),seq(-1,1,length=n_sqrt)))
#'bases <- create_random_ReLU_bases(x,10)
#'colnames(bases$f) <- paste("basis",1:ncol(bases$f))
#'L <- ncol(bases$f)
#'m <- 5
#'theta <- matrix(rnorm(L*m),nrow=L,ncol=m)
#'y <- bases$f%*%theta
#'g <- plot_2D_funcs(y,bases$x)
#'plot(g)
#'@export
plot_2D_funcs = function(f,grids,names=NULL,
                         nrow=NULL,
                         ncol=NULL){
  fdat = data.frame(f,coord_x=grids[,1],coord_y=grids[,2])
  if(!is.null(names)){
    names(fdat) <- c(names,"coord_x","coord_y")
  }
  fdat1 = reshape2::melt(fdat,id.vars=c("coord_x","coord_y"))
  return(ggplot2::ggplot(fdat1,ggplot2::aes(coord_x,coord_y,fill=value))+
           ggplot2::facet_wrap(~variable,nrow=nrow,ncol=ncol)+
           ggplot2::coord_equal()+
           ggplot2::geom_tile()+
           #ggplot2::scale_fill_gradientn(colors=create_BYR_colors()))
           ggplot2::scale_fill_gradient2(low = "blue",
                                         mid="white",
                                         high="red"))
}

#' @title Create 256 colors gradually transitioning from
#' Blue to Yellow to Red.
#' @param num  A integer number to specify the number of colors to generate. The default value is 256.
#' @return A vector of RGB colors
#' @author Jian Kang <jiankang@umich.edu>
#' @examples
#' colors = create_BYR_colors(101L)
#' require(graphics)
#' filled.contour(volcano,col=colors,nlevels=length(colors)-1,asp=1)
#' filled.contour(volcano,color.palette = create_BYR_colors, nlevels=256, asp = 1)
#' @export
create_BYR_colors = function(num=256L){
  num = as.integer(num)
  if(num<1L || num>256L){
    stop("The number of colors has to be between 1 and 256!")
  }
  R = c(rep(0,length=96),
        seq(0.015625,0.984375,length=63),
        rep(1,length=65),
        seq(0.984375,0.5,length=32))
  G = c(rep(0,length=32),
        seq(0.015625,0.984375,length=63),
        rep(1,length=65),
        seq(0.984375,0.015625,length=63),
        rep(0,length=33))
  B = c(seq(0.5,0.984375,length=32),
        rep(1,length=64),
        seq(0.984375,0.015625,length=63),
        rep(0,length=97))
  return(rgb(cbind(R[1:num],G[1:num],B[1:num])))
}

#'@title Simulate multiple images
#'@param num_imgs positive integer for number of images
#'@param coords numeric matrix
#'@param bases numeric matrix
#'@param imgdim integer vector
#'@param coord_range numeric vector
#'@param rescale logical scalar (TRUE/FALSE) indicating whether or not rescale the 
#'@return list object with three elements 
#'\describe{
#'\item{img}{numeric matrix of multiple images}
#'\item{coords}{numeric matrix for coordinates}
#'\item{bases}{list object with basis functions (f), coords (x) and other information}
#'\item{theta}{numeric matrix of coefficients to generate images}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'imgdat <- simul_images(1000)
#'plot_2D_funcs(imgdat$img[,1:10],dat$bases$x)
#'@export
simul_images <- function(num_imgs=9,coords=NULL,
                         bases=NULL,
                         imgdim=c(50,50),
                         coord_range = c(-1,1),
                         max_num_bases=500,
                         rescale=TRUE){
  if(is.null(coords)){
    coords <- create_img_coords(imgdim,coord_range) 
  }
  if(is.null(bases)){
    bases <- create_random_ReLU_bases(coords,max_num_bases)
  } 
  theta <- matrix(rnorm(num_imgs*ncol(bases$f)),nrow=ncol(bases$f),
                  ncol=num_imgs)
  f <- bases$f%*%theta
  if(rescale){
    f <- t(scale(t(f)))
  }
  imgs <- list(img=f,coords=coords,bases=bases,theta=theta)
  return(imgs)
}

#'@title Simulate response variables from linear regression 
#'@param X design matrix
#'@param betacoef numeric vector for regression coefficients
#'@param R_sq R-squared indicating the signal-to-noise ratio
#'@param sigma_sq noise variance
#'@param intercept numeric scalar to specify the intercept. The default value is 0.0.
#'@return List object with multiple elements
#'\describe{
#'\item{y}{vector of the response variable}
#'\item{X}{covariate matrix}
#'\item{betacoef}{vector of regression coefficients}
#'\item{R_sq}{R squared}
#'\item{sigma_sq}{Random noise variance}
#'\item{intercept}{scalar intercept}
#'}
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'n <- 100
#'imgs <- simul_images(n,imgdim=c(50,50))
#'coords <- create_img_coords(imgdim=c(50,50))
#'signals <- spatial_sparse_signals(coords,p=1L,shape="UMlogo")
#'regdat <- simul_linear_reg(X=t(imgs$img),betacoef=signals$img/sqrt(nrow(coords)))
#'fig1 <- ggplot2::qplot(regdat$y,bins=round(sqrt(length(regdat$y))))
#'fig2 <- with(regdat,ggplot2::qplot(X%*%betacoef,y))
#'fig3 <- plot_2D_funcs(f=regdat$betacoef,grids=coords)
#'fig4 <- plot_2D_funcs(f=imgs$img[,1],grids=coords,names="X1")
#'gridExtra::grid.arrange(fig1,fig2,fig3,fig4,nrow=2,ncol=2)
#'@export
simul_linear_reg <- function(X, betacoef=NULL,
                             R_sq = 0.9,
                             sigma_sq = NULL,
                             intercept = 0.0
){
  if(is.null(betacoef)){
    betacoef <- rep(0,length=ncol(X))
    betacoef[1] <- 1
    betacoef[2] <- 2
  }
  mu <- X%*%betacoef+intercept
  if(is.null(sigma_sq)){
    sigma_sq = var(mu)/R_sq*(1 - R_sq)
  } else{
    R_sq <- var(mu)
    R_sq <- R_sq/(sigma_sq+R_sq)
  }
  y = mu + rnorm(length(mu),sd=sqrt(sigma_sq))
  dat <- list(y=y, 
              X=X, 
              betacoef = betacoef,
              R_sq = R_sq, 
              sigma_sq = sigma_sq,
              intercept = intercept)
  return(dat)
}

#'@title Create coordinates for images
#'@param imgdim integer vector for creating image dimensions
#'@param coord_range numeric vector for the range of coordinates for each dimension
#'@return matrix of coordinates
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#' coords_1 <- create_img_coords()
#' coords_2 <- create_img_coords(imgdim = c(10,20,30), coord_range=c(-1,1,-2,2,-3,3))
#'@export
create_img_coords <- function(imgdim = c(50,50), coord_range=c(-1,1)){
  coord_range <- rep(coord_range,length=length(imgdim)*2)
  grids <- lapply(1:length(imgdim),function(i) seq(coord_range[1+(i-1)*2],coord_range[2+(i-1)*2],length=imgdim[i]))
  coords <- as.matrix(expand.grid(grids))
  return(coords)
}

#'@title Generate one or multiple 2D images with sparse signals with different shapes
#'@param coords n by 2 matrix for (x y) coordinates
#'@param p positive integer indicating the number of images
#'@param shape character scalar or vector including one or more of the following shapes "circle","triangle","square","Tshape" and "UMlogo"
#'@param effect_size positive numeric scalar indicating the magnitude of nonzero signals
#'@param smooth positive numeric scalar indicating the smoothness of the nonzero signals
#'@param radius positive numeric scalar indicating the size of activation shape in the image
#'@param offset minimum signal value in range(0,1)
#'@author Jian Kang <jiankang@umich.edu>
#'@examples
#'coords <- create_img_coords(imgdim = c(32,32)) 
#'signals <- spatial_sparse_signals(coords,p=5L,shape=c("circle","square","triangle","Tshape","UMlogo"),radius=0.5,
#' center = c(0.0,0.0), offset=0.1, effect_size=2.0,smooth=1)
#'plot_2D_funcs(signals$img,signals$coords)
#' write.csv(signals$coords, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/coords_32x32_eff2_offset01.csv",row.names = F)
#' write.csv(signals$img, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/image_32x32_eff2_offset01.csv",row.names = F)
#' 
#' coords <- create_img_coords(imgdim = c(64,64)) 
#'signals <- spatial_sparse_signals(coords,p=5L,shape=c("circle","square","triangle","Tshape","UMlogo"),radius=0.5,
#' center = c(0.0,0.0), offset=0.1, effect_size=2.0,smooth=1)
#'plot_2D_funcs(signals$img,signals$coords)
#' write.csv(signals$coords, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/coords_64x64_eff2_offset01.csv",row.names = F)
#' write.csv(signals$img, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/image_64x64_eff2_offset01.csv",row.names = F)
#' 
#' coords <- create_img_coords(imgdim = c(128,128)) 
#'signals <- spatial_sparse_signals(coords,p=5L,shape=c("circle","square","triangle","Tshape","UMlogo"),radius=0.5,
#' center = c(0.0,0.0), offset=0.1, effect_size=2.0,smooth=1)
#'plot_2D_funcs(signals$img,signals$coords)
#' write.csv(signals$coords, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/coords_128x128_eff2_offset01.csv",row.names = F)
#' write.csv(signals$img, "~/Dropbox (University of Michigan)/DeepLearningGPKernel/Code/DKL/I_S/sim/data/image_128x128_eff2_offset01.csv",row.names = F)
#'@export
spatial_sparse_signals = function(coords,p=1L,shape = c("circle","triangle",
                                                        "square","Tshape","UMlogo"),
                                  offset=0.2,center = c(0.0,0.0),
                                  effect_size=1.0,smooth=0.1,radius = 0.5){
  create.shape.in.2D.image = function(voxels,shape="circle",center = c(0.0,0.0),radius = 0.25){
    if(shape=="circle"){
      return((voxels[,1]-center[1])^2+(voxels[,2]-center[2])^2<radius^2)
    }
    if(shape=="triangle"){
      return(points.in.triangle(voxels[,1]-center[1],voxels[,2]-center[2],-radius*1.5,0.0,radius*1.5,radius*1.5,radius*1.5,-radius*1.5))
    }
    if(shape=="square"){
      square = points.in.rectangle(voxels[,1]-center[1],voxels[,2]-center[2],-radius,-radius,radius,radius)
      return(square)
    }
    if(shape=="Tshape"){
      square1 = points.in.rectangle(voxels[,1]-center[1],voxels[,2]-center[2],-radius,0.75*radius,radius,1.5*radius)
      square2 = points.in.rectangle(voxels[,1]-center[1],voxels[,2]-center[2],-0.25*radius,-1.25*radius,0.25*radius,0.75*radius)
      return(square1 | square2)
    }
    if(shape=="UMlogo"){
      rect1 <- rbind( c(25,44),
                      c(82,75))
      rect2 <- rbind(c(36,75),
                     c(71,122))
      rect3 <- rbind(c(25,122),
                     c(71,153))
      triang1 <- rbind(c(71,153),
                       c(100,114),
                       c(100,64))
      triang2 <- rbind(c(71,101),
                       c(71,153),
                       c(100,64))
      rect1 <- (rect1-100)/100
      rect2 <- (rect2-100)/100
      rect3 <- (rect3-100)/100
      triang1 <- (triang1 - 100)/100
      triang2 <- (triang2 - 100)/100
      rect1p <- rect1
      rect2p <- rect2
      rect3p <- rect3
      triang1p <- triang1
      triang2p <- triang2
      triang1p[,1] = -triang1p[,1]
      triang2p[,1] = -triang2p[,1]
      rect1p[,1] = -rect1p[2:1,1]
      rect2p[,1] = -rect2p[2:1,1]
      rect3p[,1] = -rect3p[2:1,1]
      res <- points.in.rectangle(voxels[,1],voxels[,2],rect1[1,1],rect1[1,2],rect1[2,1],rect1[2,2])
      res <- res | points.in.rectangle(voxels[,1],voxels[,2],rect2[1,1],rect2[1,2],rect2[2,1],rect2[2,2])
      res <- res | points.in.rectangle(voxels[,1],voxels[,2],rect3[1,1],rect3[1,2],rect3[2,1],rect3[2,2])
      res <- res | points.in.rectangle(voxels[,1],voxels[,2],rect1p[1,1],rect1p[1,2],rect1p[2,1],rect1p[2,2])
      res <- res | points.in.rectangle(voxels[,1],voxels[,2],rect2p[1,1],rect2p[1,2],rect2p[2,1],rect2p[2,2])
      res <- res | points.in.rectangle(voxels[,1],voxels[,2],rect3p[1,1],rect3p[1,2],rect3p[2,1],rect3p[2,2])
      res <- res | points.in.triangle(voxels[,1],voxels[,2],
                                      triang1[1,1],triang1[1,2],triang1[2,1],triang1[2,2],triang1[3,1],triang1[3,2])
      res <- res | points.in.triangle(voxels[,1],voxels[,2],
                                      triang2[1,1],triang2[1,2],triang2[2,1],triang2[2,2],triang2[3,1],triang2[3,2])
      
      res <- res | points.in.triangle(voxels[,1],voxels[,2],
                                      triang1p[1,1],triang1p[1,2],triang1p[2,1],triang1p[2,2],triang1p[3,1],triang1p[3,2])
      res <- res | points.in.triangle(voxels[,1],voxels[,2],
                                      triang2p[1,1],triang2p[1,2],triang2p[2,1],triang2p[2,2],triang2p[3,1],triang2p[3,2])
      return(res)
    }
  }
  
  
  points.in.rectangle = function(x,y,x0,y0,x1,y1){
    return((x>x0 & x<x1) & (y>y0 & y<y1))
  }
  
  points.in.triangle = function(x,y, x0,y0, x1,y1, x2,y2) {
    s = y0 * x2 - x0 * y2 + (y2 - y0) * x + (x0 - x2) * y;
    t = x0 * y1 - y0 * x1 + (y0 - y1) * x + (x1 - x0) * y;
    
    res = rep(FALSE,length=length(x))
    
    false_idx = which((s < 0) != (t < 0))
    if(length(false_idx)<length(x)){
      A = -y1 * x2 + y0 * (x2 - x1) + x0 * (y1 - y2) + x1 * y2;
      if (A < 0.0) {
        s = -s;
        t = -t;
        A = -A;
      }
      res = (s > 0 & t > 0 & (s + t) <= A)
    }
    res[false_idx] = FALSE
    return(res)
  }
  
  shape_list = rep(shape,length=p)
  sign_list = rep(c(1,-1),length=p)
  imgs = array(0, dim=c(nrow(coords),p))
  for(j in 1:p){
    i = create.shape.in.2D.image(coords,shape=shape_list[j],radius=radius)
    if(shape_list[j]=="circle"){
      dist = apply((coords[i,]-center)^2,1,sum)
      dist = exp(-smooth * dist)
      dist = (dist - min(dist)) / (max(dist) - min(dist)) * (1 - offset) + offset
      
    } else if (shape_list[j]=="square"){
      dist = apply((coords[i,]-center)^2,1,sum)
      dist = exp(-smooth * dist)
      dist = (dist - min(dist)) / (max(dist) - min(dist)) * (1 - offset) + offset
      dist = 1-dist+offset
    } else if (shape_list[j]=="triangle"){
      dist = exp(-smooth*apply((coords[i,]-0.5)^2,1,sum))
      dist = (dist - min(dist)) / (max(dist) - min(dist)) * (1 - offset) + offset
      dist = 1-dist+offset
    } else if (shape_list[j]=="Tshape"){
      dist = exp(-smooth*apply((coords[i,]-0.5)^2,1,sum))
      dist = (dist - min(dist)) / (max(dist) - min(dist)) * (1 - offset) + offset
    } else if (shape_list[j]=="UMlogo"){
      dist = 0.8
    } 
    
    #imgs[i,j] = sign_list[j]*effect_size*exp(-smooth*apply((coords[i,]-0.5)^2,1,sum))
    imgs[i,j] = sign_list[j]*effect_size*dist
    if(shape_list[j]=="UMlogo"){
      imgs[i,j] = imgs[i,j] + sign_list[j]*effect_size*exp(-smooth*apply((coords[i,]-0.5)^2,1,sum))
    }
  }
  return(list(img=imgs,coords=coords))
}
