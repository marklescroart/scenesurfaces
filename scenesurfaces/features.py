# Compute 3D scene structure features as in Lescroart & Gallant 2017
import numpy as np
import cv2

# Default parameter values

# NOTE: It is a somewhat difficult problem to place equi-distant points
# around a sphere or half-sphere. See:
# http://www.math.niu.edu/~rusin/known-math/95/sphere.faq
# ...for potential improvements in selecting normal bin centers
# Here, all vectors are x, y, z vectors; +Y is up, +Z is toward viewer
NORM_BIN_CENTERS = np.array([[-1, 0, 0],  # Cardinal directions
                             [0, 1, 0],
                             [1, 0, 0],
                             [0, -1, 0],
                             [-1, -1, 1],  # Oblique directions
                             [-1, 1, 1],
                             [1, 1, 1],
                             [1, -1, 1],
                             [0, 0, 1]])  # Straight ahead
# NOTE: Reasonable depth divisions will depend on the depth input. If absolute
# depth at approximately human scales is used, this scaling makes sense. If
# some measure of relative depth in the scene is desired, this makes less
# sense (unless that relative depth is scaled 0-100 or some such)
N_BINS_DIST = 10
MAX_DIST = 100
DIST_BIN_EDGES = np.logspace(np.log10(1), np.log10(MAX_DIST), N_BINS_DIST)
DIST_BIN_EDGES = np.hstack([0, DIST_BIN_EDGES[:-1], 999])

# Code


def compute_distance_orientation_bins(normals,
                                      distance,
                                      camera_vector=None,
                                      norm_bin_centers=NORM_BIN_CENTERS,
                                      dist_bin_edges=DIST_BIN_EDGES,
                                      sky_channel=True,
                                      remove_camera_rotation=False,
                                      assure_normals_equal_1=True,
                                      pixel_norm=False,
                                      dist_normalize=False,
                                      n_bins_x=1,
                                      n_bins_y=1,
                                      ori_norm=2,
                                      ):
    """Compute % of pixels in specified distance & orientation bins

    Preprocessing for normal & depth map images to compute scene features
    for Lescroart & Gallant, 2018

    Parameters
    ----------
    normals: array
        4D array of surface normal images (x, y, xyz_normal, frames)
    distance: array
        3D array of distance images (x, y, frames)
    norm_bin_centers: array
        array of basis vectors that specify the CENTERS (not edges) of bins for
        surface orientation
    dist_bin_edges: array
        array of EDGES (not centers) of distance bins
    sky_channel: bool
      If true, include a separate channel for sky (all depth values above max
      value in dist_bin_edges)
    dist_normalize
    remove_camera_rotation: bool or array of bools
        whether to remove X, Y, or Z rotation of camera. Defaults to False
        (do nothing to any rotation)

    ori_norm: scalar
        How to normalize norms (??): 1 = L1 (max), 2 = L2 (Euclidean)
    """
    bins_x = np.linspace(0, 1, n_bins_x+1)
    bins_x[-1] = np.inf
    bins_y = np.linspace(0, 1, n_bins_y+1)
    bins_y[-1] = np.inf

    # Computed parameters
    n_norm_bins = norm_bin_centers.shape[0]
    n_dist_bins = dist_bin_edges.shape[0] - 1
    # Normalize bin vectors
    L2norm = np.linalg.norm(norm_bin_centers, axis=1, ord=2)
    norm_bin_centers = norm_bin_centers / L2norm[:, np.newaxis]

    # Note, that the bin width for these bins will not be well-defined (or,
    # will not be uniform). For now, take the average min angle between bins
    if n_norm_bins == 1:
        # Normals can't deviate by more than 90 deg (unless they're un-
        # rotated) BUT: We don't actually want to soft bin if there is only one
        # normal, we want to assign ALL pixels EQUALLY to the ONE BIN.
        # Thus d = np.inf
        d = np.inf
    else:
        # Soft-bin normals
        d = np.arccos(norm_bin_centers.dot(norm_bin_centers.T))
        d[np.abs(d) < 0.00001] = np.nan
    norm_bin_width = np.mean(np.nanmin(d))
    # Add an extra buffer to this? We don't want "stray" pixels with normals
    # that don't fall into any bin (but we also don't want to double-count
    # pixels)

    print('Done with stim file set-up: check!')
    # Optionaly remove any camera rotations
    if remove_camera_rotation is False:
        remove_camera_rotation = np.array([False, False, False])
    if np.any(remove_camera_rotation):
        normals = remove_rotation(normals,
                                  -camera_angles,
                                  is_normalize_normals=is_normalize_normals,
                                  angle_to_remove=remove_camera_rotation)
    # Get number of images
    x, y, n_ims = distance.shape
    n_tiles = n_bins_y * n_bins_x
    n_dims = n_tiles * n_dist_bins * n_norm_bins
    if sky_channel:
        n_dims = n_dims + n_tiles
    else:
        n_dims = n_tiles * n_dist_bins * n_norm_bins

    output = np.zeros((n_ims, n_dims)) * np.nan
    for iS in range(n_ims):
        if n_ims>200:
            if iS % 200 == 0:
                print("Done to image %d / %d"%(iS, n_ims)) #progressdot(iS,200,2000,n_ims)
        elif (n_ims < 200) and (n_ims > 1):
            disp('computing Scene Depth Normals...')
        # Pull single image for preprocessing
        z = distance[..., iS]  #S.(zVar)(:,:,iS)
        n = normals[..., iS]  # S.Normals(:,:,:,iS)
        height, width, nd = n.shape
        xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        idx = np.arange(n_norm_bins)
        for d_st, d_fin in zip(dist_bin_edges[:-1], dist_bin_edges[1:]):
            dIdx = (z >= d_st) & (z < d_fin)
            for ix in range(n_bins_x):
                hIdx = (xx >= bins_x[ix]) & (xx < bins_x[ix+1])
                for iy in range(n_bins_y):
                    vIdx = (yy >= bins_y[iy]) & (yy < bins_y[iy + 1])
                    this_section = (dIdx & hIdx) & vIdx
                    if this_section.sum()==0:
                        output[iS, idx] = 0
                        idx += n_norm_bins
                        continue
                    if n_norm_bins > 1:
                        nn = n[this_section, :]
                        # Compute orientation of pixelwise surface normals relative
                        # to all normal bins
                        o = nn.dot(norm_bin_centers.T)
                        #print(o.shape)
                        #L2nn = np.linalg.norm(nn, axis=1, ord=2)
                        #o = bsxfun(@rdivide,o,Lb) # Norm of norm_bin_centers should be 1
                        o /= np.linalg.norm(nn, axis=1, ord=2)[:, np.newaxis]
                        if np.max(o-1) > 0.0001:
                            raise Exception('The magnitude of one of your normal bin vectors crossed with a stimulus normal is > 1 - Check on your stimulus / normal vectors!')
                        # Get rid of values barely > 1 to prevent imaginary output
                        o = np.minimum(o, 1)  
                        angles = np.arccos(o)
                        # The following is a "soft" histogramming of normals.
                        # i.e., if a given normal falls partway between two
                        # normal bins, it is partially assigned to each of the
                        # nearest bins (not exclusively to one).
                        tmp_out = np.maximum(0, norm_bin_width - angles) / norm_bin_width
                        # Sum over all pixels w/ depth in this range
                        tmp_out = np.sum(tmp_out, axis=0)
                        # Normalize across different normal orientation bins
                        tmp_out = tmp_out / np.linalg.norm(tmp_out, ord=ori_norm)
                        if pixel_norm:
                            tmp_out = tmp_out * len(nn)/len(dIdx.flatten())
                    else:
                        # Special case: one single normal bin
                        # compute the fraction of screen pixels in this screen
                        # tile at this depth
                        tmp_out = np.mean(this_section)
                    # Illegal for more than two bins of normals within the same
                    # depth / horiz/vert tile to be == 1
                    if sum(tmp_out == 1) > 1:
                        error('Found two separate normal bins equal to 1 - that should be impossible!')
                    if dist_normalize and not (n_norm_bins == 1):
                        # normalize normals by n pixels at this depth/screen tile
                        tmp_out = tmp_out * pct_pix_this_depth
                    output[iS, idx] = tmp_out
                    idx += n_norm_bins
        # Do sky channel(s) after last depth channel, add (n tiles) sky channels
        if sky_channel and (np.max(dist_bin_edges) < np.inf):
            dSky = z >= dist_bin_edges(-1)
            skyidx = np.arange((n_dims - n_tiles + 1), n_dims)
            tmp = np.zeros(n_bins_y, n_bins_x)
            for x_st, x_fin in zip(bins_x[:-1], bins_x[1:]):
                hIdx = (xx >= x_st) & (xx < x_fin)
                for y_st, y_fin in zip(bins_y[:-1], bins_y[1:]):
                    vIdx = (yy >= y_st) & (yy < y_fin)
                    tmp[iy, ix] = np.mean(dSky & hIdx & vIdx)
            output[iS, skyidx] = tmp.flatten()

    # Cleanup
    output[np.isnan(output)] = 0
    params = dict()  # Fill me
    return output, params


def remove_rotation(angle_to_remove=(True, False, False), do_normalize=True):
    """Remove rotation about one or more axes from normals

    Parameters
    ----------
    angle_to_remove: tuple or list
        list of boolean values indicating whether to remove [x, y, z] rotations
    do_normalize: bool
        whether to re-normalize angles after rotation is removed.
    """
    # Load normals for test scene, w/ camera moving around square block:
    SzY, SzX, Ch, nFr = N.shape
    NN = np.zeros_like(N)

    if do_normalize:
        # old stinky matlab: bsxfun(@rdivide,V,sum(V.^2,2).^.5)
        V /= np.linalg.norm(V, axis=1)

    for iFr in range(nFr):
        if iFr % 100 == 0:
            print("Done to frame %d/%d"%(iFr, nFr))
        c_vec = V[iFr]
        camera_matrix = vector_to_camera_matrix(c_vec, ~angle_to_remove)
        n = N[..., iFr].reshape(-1, 3)  # reshape(N(:,:,:,iFr),[],3);
        nT = camera_matrix.dot(n.T)
        nn = nT.T.reshape((SzY, SzX, Ch))
        NN[:, :, :, iFr] = nn
    return NN


def vector_to_camera_matrix(c_vec, ignore_rot_xyz):
    """Gets the camera (perspective) transformation matrix, given a vector

    Vector should be from camera->fixation.  Optionally, sets one (or more)
    axes of rotation for the camera to zero (this provides a matrix that
    "un-rotates" the camera perspective, but leaves whatever axes are set to
    "true" untouched (i.e., in their original image space).

    Deals with ONE VECTOR AT A TIME

    IgnoreRot = [false,true,false] by default (there should be no y rotation
      [roll] of cameras anyway!)

    """
    xr, yr, zr = (~np.array(ignore_rot_xyz)).astype(np.bool)
    # Vector to Euler angles:
    if xr:
        xr = np.arctan2(c_vec[3], (c_vec[1]**2 + c_vec[2]**2)**0.5)
    if yr:
        raise Exception("SORRY I don''t compute y rotations! Pls consult wikipedia!")
    else:
        yr = 0
    if zr:
        zr = -np.arctan2(c_vec[0], c_vec[1])
    # Rotation matrices, given Euler angles:
    # X rotation
    xRot = np.array([[1., 0., 0.],
                     [0., np.cos(xr), np.sin(xr)],
                     [0., -np.sin(xr), np.cos(xr)]])
    # Y rotation
    yRot = np.array([[np.cos(yr), 0., -np.sin(yr)],
                     [0., 1., 0.],
                     [np.sin(yr), 0., np.cos(yr)]])
    # Z rotation
    zRot = np.array([[np.cos(zr), np.sin(zr), 0.],
                     [-np.sin(zr), np.cos(zr), 0.],
                     [0., 0., 1.]])
    # Multiply rotations to get final matrix
    camera_matrix = xRot.dot(yRot).dot(zRot)
    return camera_matrix
