# Denoising OCT images metrics (CNR, MSR, TP, EP and ENL)

- CNR: Contrast-to-Noise Ratio

- MSR: Mean-to-Standard-deviation Ratio

- TP: Texture Preservation

- EP: Edge Preservation

- ENL: Equivalent Number of Looks

# How to use

## If you want to calculate metrics for a single image
- you can use `comp_single_main.py` with some below variables modifications :
  - `denoised` is the denoised image file
  - `noisy` is the noisy image
  - `ROIs` is the list contains ROIs with the format `(x,y,w,h)` 
  - `background_indices` is the list contains indices of background regions in ROIs

## If you want to calculate metrics for all images in a folder
- you can use `comp_multi_main.py` with some below variables modifications :
  - `denoised` is the folder contains all denoised images
  - `noisy` is the folder contains all noisy images
  - `result_path` is the file csv that you want to put the result into
  - `ROIs` is the list contains ROIs with the format `(x,y,w,h)`
  - `background_indices` is the list contains indices of background regions in ROIs



