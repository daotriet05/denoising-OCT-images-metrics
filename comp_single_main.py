import cv2
from EP import comp_EP
from TP import comp_TP
from MSR_CNR_ENL import comp_MSR_CNR_ENL

if __name__=='__main__':
    denoised = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/input/self_fusion.jpg', cv2.IMREAD_GRAYSCALE) # denoised
    denoised=cv2.resize(denoised,(300,300))
    noisy = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/original/epoch_20.jpg', cv2.IMREAD_GRAYSCALE) # noisy
    noisy=cv2.resize(noisy,(300,300))

    ROIs = [(250,0,50,50), (100, 130, 30, 30), (170, 150, 30, 30), (10, 35, 30, 30), (50,150,50,50), (200,200,50,50)] # ROIs
    background_indices = [0] # indices of background ROIs
    tp = comp_TP(denoised, noisy, ROIs, background_indices)
    ep = comp_EP(denoised, noisy, ROIs, background_indices)
    msr, cnr, enl = comp_MSR_CNR_ENL(denoised, noisy, ROIs, background_indices)

    print("Texture Preservation:", tp)
    print("Edge Preservation: ", ep)
    print("Mean-to-standard-deviation Ratio: ", msr)
    print("Contrast-to-Noise Ratio: ", cnr)
