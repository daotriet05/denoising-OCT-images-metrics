import numpy as np
from get_roi_pos import *

def comp_MSR_CNR_ENL(im_out,ref, pos, background_indices):
    """
    Computes three non-referenced quality assessment metrics (MSR, CNR, and ENL).

    Args:
        roi (list of numpy arrays): List of ROI vectors.
        background_indices (numpy array): Vector indicating which ROIs are background ROI.
                                          e.g., background_indices = np.array([1]) means that
                                          the first region is the only background region.

    Returns:
        MSR (float): Mean-to-standard-deviation ratio.
        CNR (float): Contrast to Noise Ratio.
        ENL (float): Equivalent number of looks.
    """
    roi,_=get_roi_pos(im_out,pos)
    N = len(roi)  # number of ROI vectors

    foreground_indices = np.setdiff1d(np.arange(N), background_indices)

    # ===========
    # Compute MSR
    # MSR = average of mean to standard deviation ratios on foreground regions

    f_mean = 0  # mean of intensity values in the foreground ROIs
    f_std = 0   # std of intensity values in the foreground ROIs
    MSR = 0

    for j in foreground_indices:
        m = np.mean(roi[j])
        s = np.std(roi[j])
        MSR += m / s
        f_mean += m
        f_std += s

    Nf = len(foreground_indices)  # number of foreground regions

    MSR /= Nf
    f_mean /= Nf
    f_std /= Nf

    # ===========
    # Compute CNR
    # CNR = measures the contrast between foreground regions and background regions.

    # background ROI vectors
    broi = np.array([roi[idx] for idx in background_indices])

    CNR = abs(f_mean - np.mean(broi)) / np.sqrt(0.5 * (f_std**2 + np.std(broi)**2))

    # ===========
    # Compute ENL
    ENL = np.mean(broi)**2 / np.std(broi)**2

    return MSR, CNR, ENL
