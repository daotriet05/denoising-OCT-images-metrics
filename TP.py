import numpy as np
import cv2
from get_roi_pos import *


def comp_var_ratio(rois_out, rois_ref):
    """Compute the variance ratio for given sets of ROIs."""
    var_ratio = 0
    n = len(rois_out)
    for i in range(n):
        if np.var(rois_ref[i]) > 0:  # Avoid division by zero
            var_ratio += np.var(rois_out[i]) / np.var(rois_ref[i])
    return var_ratio

def comp_TP(im_out, ref, pos, background_indices):
    """Compute texture preservation (TP) quality metric."""
    fore_pos=[roi for i,roi in enumerate(pos) if i not in background_indices]
    #fore_pos=pos
    rois_out,_ = get_roi_pos(im_out, fore_pos)
    rois_ref,_ = get_roi_pos(ref, fore_pos)

    var_ratio = comp_var_ratio(rois_out, rois_ref)  # Compute variance ratio

    N = len(fore_pos)  # Total number of ROIs
    mean_ratio = np.sqrt(np.mean(im_out) / np.mean(ref))  # Global mean ratio
    TP = (1 / N) * mean_ratio * var_ratio  # Compute texture preservation
    return TP

# Example usage:
if __name__=='__main__':
    im_out = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/input/self_fusion.jpg', cv2.IMREAD_GRAYSCALE)
    im_out=cv2.resize(im_out,(256,256))
    ref = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/original/epoch_20.jpg', cv2.IMREAD_GRAYSCALE) 
    ref=cv2.resize(ref,(256,256))
    pos = [(200,0,50,50), (100, 110, 30, 30), (170, 130, 30, 30), (10, 35, 30, 30), (60,150,50,50), (170,180,50,50)]
    background_indices = [0]
    tp_value = comp_TP(im_out, ref, pos, background_indices)
    print("Texture Preservation:", tp_value)
