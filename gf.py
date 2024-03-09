import numpy as np
import scipy as sp
import scipy.ndimage
import imageio

def box_filter(img, r):
    """Apply a box filter to the image.

    Args:
        img (ndarray): Input image.
        r (int): Radius of the box filter.

    Returns:
        ndarray: Filtered image.
    """
    (rows, cols) = img.shape[:2]
    im_dst = np.zeros_like(img)

    tile = [1] * img.ndim
    tile[0] = r
    im_cum = np.cumsum(img, 0)
    im_dst[0:r + 1, :, ...] = im_cum[r:2 * r + 1, :, ...]
    im_dst[r + 1:rows - r, :, ...] = im_cum[2 * r + 1:rows, :, ...] - im_cum[0:rows - 2 * r - 1, :, ...]
    im_dst[rows - r:rows, :, ...] = np.tile(im_cum[rows - 1:rows, :, ...], tile) - im_cum[ rows - 2 * r - 1:rows - r - 1, :, ...]

    tile = [1] * img.ndim
    tile[1] = r
    im_cum = np.cumsum(im_dst, 1)
    im_dst[:, 0:r + 1, ...] = im_cum[:, r:2 * r + 1, ...]
    im_dst[:, r + 1:cols - r, ...] = im_cum[:, 2 * r + 1:cols, ...] - im_cum[:, 0:cols - 2 * r - 1, ...]
    im_dst[:, cols - r: cols, ...] = np.tile(im_cum[:, cols - 1:cols, ...], tile) - im_cum[ :, cols - 2 * r - 1:cols - r - 1, ...]

    return im_dst


def guided_filter(I, p, r, eps, s=None):
    """Guided filter for image dehazing.

    Args:
        I (ndarray): Guide image.
        p (ndarray): Filtering input.
        r (int): Window radius.
        eps (float): Regularization parameter.
        s (int): Subsampling factor for fast guided filter.

    Returns:
        ndarray: Filtered output.
    """
    if p.ndim == 2:
        p3 = p[:, :, np.newaxis]
    else:
        p3 = p

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:, :, ch] = _gf_colorgray(I, p3[:, :, ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out


def _gf_colorgray(I, p, r, eps, s=None):
    """Automatically choose color or gray guided filter based on image dimensions."""
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        raise ValueError("Invalid guide dimensions:", I.shape)


def _gf_gray(I, p, r, eps, s=None):
    """Grayscale guided filter."""
    if s is not None:
        Isub = sp.ndimage.zoom(I, 1 / s, order=1)
        Psub = sp.ndimage.zoom(p, 1 / s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p

    (rows, cols) = Isub.shape
    N = box_filter(np.ones([rows, cols]), r)
    meanI = box_filter(Isub, r) / N
    meanP = box_filter(Psub, r) / N
    corrI = box_filter(Isub * Isub, r) / N
    corrIp = box_filter(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = box_filter(a, r) / N
    meanB = box_filter(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    q = meanA * I + meanB
    return q


def _gf_color(I, p, r, eps, s=None):
    """Color guided filter."""
    fullI = I
    fullP = p
    if s is not None:
        I = sp.ndimage.zoom(fullI, [1 / s, 1 / s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1 / s, 1 / s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box_filter(np.ones((h, w)), r)

    mI_r = box_filter(I[:, :, 0], r) / N
    mI_g = box_filter(I[:, :, 1], r) / N
    mI_b = box_filter(I[:, :, 2], r) / N

    mP = box_filter(p, r) / N

    mIp_r = box_filter(I[:, :, 0] * p, r) / N
    mIp_g = box_filter(I[:, :, 1] * p, r) / N
    mIp_b = box_filter(I[:, :, 2] * p, r) / N

    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    var_I_rr = box_filter(I[:, :, 0] * I[:, :, 0], r) / N - mI_r * mI_r
    var_I_rg = box_filter(I[:, :, 0] * I[:, :, 1], r) / N - mI_r * mI_g
    var_I_rb = box_filter(I[:, :, 0] * I[:, :, 2], r) / N - mI_r * mI_b

    var_I_gg = box_filter(I[:, :, 1] * I[:, :, 1], r) / N - mI_g * mI_g
    var_I_gb = box_filter(I[:, :, 1] * I[:, :, 2], r) / N - mI_g * mI_b

    var_I_bb = box_filter(I[:, :, 2] * I[:, :, 2], r) / N - mI_b * mI_b

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            sig = np.array([
                [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]]
            ])
            covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
            a[i, j, :] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:, :, 0] * mI_r - a[:, :, 1] * mI_g - a[:, :, 2] * mI_b

    meanA = box_filter(a, r) / N[..., np.newaxis]
    meanB = box_filter(b, r) / N

    if s is not None:
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    q = np.sum(meanA * fullI, axis=2) + meanB
    return q


def test_gf():
    """Test the guided filter."""
    cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    tulips = imageio.imread('tulips.bmp').astype(np.float32) / 255

    r = 8
    eps = 0.05

    cat_smoothed = guided_filter(cat, cat, r, eps)
    cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)

    imageio.imwrite('cat_smoothed.png', cat_smoothed)
    imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

    tulips_smoothed4s = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed4s[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps, s=4)
    imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

    tulips_smoothed = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps)
    imageio.imwrite('tulips_smoothed.png', tulips_smoothed)