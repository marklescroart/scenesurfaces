# File input / output
import numpy as np
import cv2

def load_exr_normals(fname, xflip=True, yflip=True, zflip=True, clip=True):
    """Load an exr (floating point) image to surface normal array

    """
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    imc = img-1
    y, z, x = imc.T
    rev_x = xflip
    rev_y = yflip
    rev_z = zflip
    if rev_x: 
        x = -x
    if rev_y:
        y = -y
    if rev_z:
        z = -z
    imc = np.dstack([x.T,y.T,z.T])
    if clip:
        imc = np.clip(imc, -1, 1)
    return imc


def load_exr_zdepth(fname, thresh=1000):
    """Load an exr (floating point) image to absolute distance array"""
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    z = img[..., 0]
    z[z > thresh] = np.nan
    return z