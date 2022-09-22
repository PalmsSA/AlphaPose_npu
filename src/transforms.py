import numpy as np
import cv2


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0]
    width = heatmaps.shape[2]
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 1)
    maxvals = np.max(heatmaps_reshaped, 1)

    maxvals = maxvals.reshape((num_joints, 1))
    idx = idx.reshape((num_joints, 1))

    preds = np.tile(idx, (1, 2)).astype(np.float32)

    preds[:, 0] = (preds[:, 0]) % width
    preds[:, 1] = np.floor((preds[:, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_pred_batch(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def heatmap_to_coord_simple(hms, bbox, hms_flip=None, **kwargs):
    if hms_flip is not None:
        hms = (hms + hms_flip) / 2
    if not isinstance(hms, np.ndarray):
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms)

    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing
    for p in range(coords.shape[0]):
        hm = hms[p]
        px = int(round(float(coords[p][0])))
        py = int(round(float(coords[p][1])))
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
            diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                             hm[py + 1][px] - hm[py - 1][px]))
            coords[p] += np.sign(diff) * .25

    preds = np.zeros_like(coords)

    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center, scale,
                                   [hm_w, hm_h])

    return preds, maxvals
