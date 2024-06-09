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
