import cv2
from EP import comp_EP
from TP import comp_TP
from MSR_CNR_ENL import comp_MSR_CNR_ENL

if __name__=='__main__':
    im_out = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/input/self_fusion.jpg', cv2.IMREAD_GRAYSCALE) # denoised
    im_out=cv2.resize(im_out,(300,300))
    ref = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/original/epoch_20.jpg', cv2.IMREAD_GRAYSCALE) # noisy
    ref=cv2.resize(ref,(300,300))
    pos = [(250,0,50,50), (100, 130, 30, 30), (170, 150, 30, 30), (10, 35, 30, 30), (50,150,50,50), (200,200,50,50)] # ROIs
    background_indices = [0] # indices of background ROIs
    tp = comp_TP(im_out, ref, pos, background_indices)
    ep = comp_EP(im_out, ref, pos, background_indices)
    msr, cnr, enl = comp_MSR_CNR_ENL(im_out, ref, pos, background_indices)
    print("Texture Preservation:", tp)
    print("Edge Preservation: ", ep)
    print("Mean-to-standard-deviation Ratio: ", msr)
    print("Contrast-to-Noise Ratio: ", cnr)
