import numpy as np
import cv2

from get_roi_pos import*

def comp_EP(im_out, ref, pos, background_indices):
    """
    Compute Edge Preservation (EP) quality metric.
    
    Parameters:
        im_out (numpy.array): Denoised image.
        ref (numpy.array): Reference image (can be noisy or noise-free).
        pos (list of tuples): List of ROI positions in (x, y, w, h) format.
        background_indices (list): Indices of background ROIs.
    
    Returns:
        float: Edge Preservation metric.
    """
    # Create Laplacian filter
    h = cv2.Laplacian(np.float32(ref), cv2.CV_32F)
    
    # Get ROIs and masks
    _, masks = get_roi_pos(ref, pos)
    
    foreground_indices = [i for i in range(len(pos)) if i not in background_indices]
    
    # Calculate EP for each foreground ROI
    EPm = []
    for i in foreground_indices:
        ROI_N = ref * masks[i]
        ROI_N_Lap = cv2.filter2D(ROI_N, -1, h)

        ROI = im_out * masks[i]
        ROI_Lap = cv2.filter2D(ROI, -1, h)

        num = np.corrcoef(ROI_N_Lap.flatten(), ROI_Lap.flatten())[0, 1]
        denom = np.sqrt(np.corrcoef(ROI_N_Lap.flatten(), ROI_N_Lap.flatten())[0, 1] *
                        np.corrcoef(ROI_Lap.flatten(), ROI_Lap.flatten())[0, 1])
        EPm.append(num / denom)

    # Average EP across all foreground ROIs
    EP = np.mean(EPm)
    return EP

# Example usage:
if __name__=='__main__':
    im_out = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/input/epoch_10.jpg', cv2.IMREAD_GRAYSCALE)
    im_out=cv2.resize(im_out,(256,256))
    ref = cv2.imread('D:/projects/IEEE SPS 2024/code/metrics/data_2/original/epoch_20.jpg', cv2.IMREAD_GRAYSCALE) 
    ref=cv2.resize(ref,(256,256))
    pos = [(200,0,50,50), (100, 110, 30, 30), (170, 130, 30, 30), (10, 35, 30, 30), (60,150,50,50), (170,180,50,50)]
    background_indices = [0]
    ep_value = comp_EP(im_out, ref, pos, background_indices)
    print("Edge Preservation:", ep_value)
