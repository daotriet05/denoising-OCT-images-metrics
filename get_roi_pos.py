import numpy as np

def get_roi_pos(img, pos):
    """
    Returns ROI vectors from their positions and corresponding masks.
    
    Parameters:
        img (numpy.array): Image from which ROIs are extracted.
        pos (list of tuples): List of ROI positions in (x, y, w, h) format.
    
    Returns:
        list: List of ROI arrays.
        list: List of ROI masks as binary arrays.
    """
    R, C = img.shape
    Nroi = len(pos)
    rois = []
    masks = []
    
    for i in range(Nroi):
        x, y, w, h = pos[i]
        rows = np.clip(np.arange(y, y + h), 0, R)
        cols = np.clip(np.arange(x, x + w), 0, C)
        mask = np.zeros((R, C), dtype=bool)
        mask[np.ix_(rows, cols)] = True
        roi = img * mask
        rois.append(roi[mask])
        masks.append(mask)
    
    return rois, masks