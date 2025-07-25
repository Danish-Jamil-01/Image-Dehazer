import numpy as np
import cv2

def _box_filter(img, r):
    ksize = (2 * r + 1, 2 * r + 1)
    return cv2.boxFilter(img, -1, ksize, normalize=True) * (ksize[0] * ksize[1])

def _gf_gray(I, p, r, eps, s=None):
    """Grayscale guided filter."""
    if s is not None:
        I_sub = cv2.resize(I, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
        p_sub = cv2.resize(p, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
        r = round(r / s)
    else:
        I_sub = I
        p_sub = p

    (rows, cols) = I_sub.shape
    N = _box_filter(np.ones([rows, cols], dtype=np.float32), r)

    mean_I = _box_filter(I_sub, r) / N
    mean_p = _box_filter(p_sub, r) / N
    corr_I = _box_filter(I_sub * I_sub, r) / N
    corr_Ip = _box_filter(I_sub * p_sub, r) / N

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r) / N
    mean_b = _box_filter(b, r) / N

    if s is not None:
        mean_a = cv2.resize(mean_a, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b
    return q

def _gf_color(I, p, r, eps, s=None):
    """Color guided filter."""
    full_I = I.astype(np.float32)
    full_p = p.astype(np.float32)

    if s is not None:
        I = cv2.resize(full_I, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
        p = cv2.resize(full_p, None, fx=1/s, fy=1/s, interpolation=cv2.INTER_NEAREST)
        r = round(r / s)

    h, w, _ = I.shape
    N = _box_filter(np.ones((h, w), dtype=np.float32), r)

    mI_r, mI_g, mI_b = [(_box_filter(I[:,:,i], r) / N) for i in range(3)]
    mP = _box_filter(p, r) / N
    mIp_r, mIp_g, mIp_b = [(_box_filter(I[:,:,i] * p, r) / N) for i in range(3)]
    
    covIp_r, covIp_g, covIp_b = mIp_r - mI_r*mP, mIp_g - mI_g*mP, mIp_b - mI_b*mP

    var_I_rr = _box_filter(I[:,:,0]**2, r)/N - mI_r**2
    var_I_rg = _box_filter(I[:,:,0]*I[:,:,1], r)/N - mI_r*mI_g
    var_I_rb = _box_filter(I[:,:,0]*I[:,:,2], r)/N - mI_r*mI_b
    var_I_gg = _box_filter(I[:,:,1]**2, r)/N - mI_g**2
    var_I_gb = _box_filter(I[:,:,1]*I[:,:,2], r)/N - mI_g*mI_b
    var_I_bb = _box_filter(I[:,:,2]**2, r)/N - mI_b**2
    
    inv_Sigma = np.empty((h, w, 3, 3), dtype=np.float32)
    inv_Sigma[:,:,0,0] = var_I_gg*var_I_bb - var_I_gb**2
    inv_Sigma[:,:,0,1] = var_I_rb*var_I_gb - var_I_rg*var_I_bb
    inv_Sigma[:,:,0,2] = var_I_rg*var_I_gb - var_I_rb*var_I_gg
    inv_Sigma[:,:,1,0] = inv_Sigma[:,:,0,1]
    inv_Sigma[:,:,1,1] = var_I_rr*var_I_bb - var_I_rb**2
    inv_Sigma[:,:,1,2] = var_I_rb*var_I_rg - var_I_rr*var_I_gb
    inv_Sigma[:,:,2,0] = inv_Sigma[:,:,0,2]
    inv_Sigma[:,:,2,1] = inv_Sigma[:,:,1,2]
    inv_Sigma[:,:,2,2] = var_I_rr*var_I_gg - var_I_rg**2

    det_Sigma = (inv_Sigma[:,:,0,0]*(var_I_rr) + 
                 inv_Sigma[:,:,0,1]*(var_I_rg) + 
                 inv_Sigma[:,:,0,2]*(var_I_rb))

    det_Sigma += eps
    inv_det_Sigma = 1.0 / det_Sigma

    a = np.empty((h, w, 3), dtype=np.float32)
    a[:,:,0] = (inv_Sigma[:,:,0,0]*covIp_r + inv_Sigma[:,:,0,1]*covIp_g + inv_Sigma[:,:,0,2]*covIp_b) * inv_det_Sigma
    a[:,:,1] = (inv_Sigma[:,:,1,0]*covIp_r + inv_Sigma[:,:,1,1]*covIp_g + inv_Sigma[:,:,1,2]*covIp_b) * inv_det_Sigma
    a[:,:,2] = (inv_Sigma[:,:,2,0]*covIp_r + inv_Sigma[:,:,2,1]*covIp_g + inv_Sigma[:,:,2,2]*covIp_b) * inv_det_Sigma

    b = mP - a[:,:,0]*mI_r - a[:,:,1]*mI_g - a[:,:,2]*mI_b
    
    mean_a = _box_filter(a, r) / N[..., np.newaxis]
    mean_b = _box_filter(b, r) / N
    
    if s is not None:
        mean_a = cv2.resize(mean_a, (full_I.shape[1], full_I.shape[0]), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (full_I.shape[1], full_I.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return np.sum(mean_a * full_I, axis=2) + mean_b

def guided_filter(I, p, r, eps, s=None):
    """
    Main guided filter function. Dispatches to color or grayscale version.
    """
    if p.ndim == 2:
        p_3d = p[..., np.newaxis]
    else:
        p_3d = p

    if I.ndim == 2:
        I_3d = I[..., np.newaxis]
    else:
        I_3d = I
    
    if I_3d.shape[2] == 1:
        # Grayscale guide
        return np.squeeze(_gf_gray(I_3d[...,0], p, r, eps, s))
    elif I_3d.shape[2] == 3:
        # Color guide
        if p_3d.shape[2] == 1:
            return np.squeeze(_gf_color(I_3d, p_3d[...,0], r, eps, s))
        else: # p has 3 channels
            out = np.empty_like(p_3d)
            for ch in range(p_3d.shape[2]):
                out[...,ch] = _gf_color(I_3d, p_3d[...,ch], r, eps, s)
            return out
    else:
        raise ValueError("Guide image must have 1 or 3 channels")