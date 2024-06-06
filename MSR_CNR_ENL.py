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

    f_mean = 0  # mean of intensity values in the foreground ROIs
    f_std = 0   # std of intensity values in the foreground ROIs
    MSR = 0
    CNR = 0

    # background ROI vectors
    broi = np.array([roi[idx] for idx in background_indices])
    m_broi = np.mean(broi)
    std_broi = np.std(broi)

    # mean and std foreground ROI
    for j in foreground_indices:
        m = np.mean(roi[j])
        s = np.std(roi[j])
        MSR += m / s 
        CNR+=10*np.log(abs(m-m_broi) / np.sqrt(0.5 * (s**2+std_broi**2) ))

    Nf = len(foreground_indices)  # number of foreground regions

    MSR /= Nf
    CNR /= Nf
    ENL = m_broi**2 / std_broi**2

    return MSR, CNR, ENL
