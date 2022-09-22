import numpy as np
import cv2

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
alpha = 0.1
vis_thr = 0.2
oks_thr = 0.9


def pose_nms(bboxes, bbox_scores, bbox_ids, pose_preds, pose_scores, areaThres=0):
    '''
    Parametric Pose NMS algorithm
    bboxes:         bbox locations list (n, 4)
    bbox_scores:    bbox scores list (n, 1)
    bbox_ids:       bbox tracking ids list (n, 1)
    pose_preds:     pose locations list (n, kp_num, 2)
    pose_scores:    pose scores list    (n, kp_num, 1)
    '''
    # global ori_pose_preds, ori_pose_scores, ref_dists

    pose_scores[pose_scores == 0] = 1e-5
    kp_nums = pose_preds.shape[1]
    res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids = [], [], [], [], [], []

    ori_bboxes = bboxes.copy()
    ori_bbox_scores = bbox_scores.copy()
    ori_bbox_ids = bbox_ids.copy()
    ori_pose_preds = pose_preds.copy()
    ori_pose_scores = pose_scores.copy()

    xmax = bboxes[:, 2]
    xmin = bboxes[:, 0]
    ymax = bboxes[:, 3]
    ymin = bboxes[:, 1]

    widths = xmax - xmin
    heights = ymax - ymin
    ref_dists = alpha * np.maximum(widths, heights)

    nsamples = bboxes.shape[0]
    human_scores = pose_scores.mean(axis=1)

    human_ids = np.arange(nsamples)
    mask = np.ones(len(human_ids)).astype(bool)

    # Do pPose-NMS
    pick = []
    merge_ids = []
    while (mask.any()):
        numpy_mask = np.array(mask) == True
        # Pick the one with highest score
        pick_id = np.argmax(human_scores[numpy_mask])
        pick.append(human_ids[mask][pick_id])

        # Get numbers of match keypoints by calling PCK_match
        ref_dist = ref_dists[human_ids[mask][pick_id]]
        simi = get_parametric_distance(pick_id, pose_preds[numpy_mask], pose_scores[numpy_mask], ref_dist)
        num_match_keypoints = PCK_match(pose_preds[numpy_mask][pick_id], pose_preds[numpy_mask], ref_dist)

        # Delete humans who have more than matchThreds keypoints overlap and high similarity
        delete_ids = np.arange(human_scores[numpy_mask].shape[0])[
            ((simi > gamma) | (num_match_keypoints >= matchThreds))]

        if delete_ids.shape[0] == 0:
            delete_ids = pick_id

        merge_ids.append(human_ids[mask][delete_ids])
        newmask = mask[mask]
        newmask[delete_ids] = False
        mask[mask] = newmask

    assert len(merge_ids) == len(pick)
    preds_pick = ori_pose_preds[pick]
    scores_pick = ori_pose_scores[pick]
    bbox_scores_pick = ori_bbox_scores[pick]
    bboxes_pick = ori_bboxes[pick]
    bbox_ids_pick = ori_bbox_ids[pick]
    # final_result = pool.map(filter_result, zip(scores_pick, merge_ids, preds_pick, pick, bbox_scores_pick))
    # final_result = [item for item in final_result if item is not None]

    for j in range(len(pick)):
        ids = np.arange(kp_nums)
        max_score = np.max(scores_pick[j, ids, 0])

        if max_score < scoreThreds:
            continue

        # Merge poses
        merge_id = merge_ids[j]
        merge_pose, merge_score = p_merge_fast(
            preds_pick[j], ori_pose_preds[merge_id], ori_pose_scores[merge_id], ref_dists[pick[j]])

        max_score = np.max(merge_score[ids])
        if max_score < scoreThreds:
            continue

        xmax = max(merge_pose[:, 0])
        xmin = min(merge_pose[:, 0])
        ymax = max(merge_pose[:, 1])
        ymin = min(merge_pose[:, 1])
        bbox = bboxes_pick[j].tolist()
        bbox_score = bbox_scores_pick[j]

        if (1.5 ** 2 * (xmax - xmin) * (ymax - ymin) < areaThres):
            continue

        res_bboxes.append(bbox)
        res_bbox_scores.append(bbox_score)
        res_bbox_ids.append(ori_bbox_ids[merge_id].tolist())
        res_pose_preds.append(merge_pose)
        res_pose_scores.append(merge_score)
        res_pick_ids.append(pick[j])

    return res_bboxes, res_bbox_scores, res_bbox_ids, res_pose_preds, res_pose_scores, res_pick_ids


def PCK_match(pick_pred, all_preds, ref_dist):
    dist = np.sqrt(np.sum(
        np.power(pick_pred[np.newaxis, :] - all_preds, 2),
        axis=2
    ))
    ref_dist = min(ref_dist, 7)
    num_match_keypoints = np.sum(
        dist / ref_dist <= 1,
        axis=1
    )

    return num_match_keypoints


def get_parametric_distance(i, all_preds, keypoint_scores, ref_dist):
    pick_preds = all_preds[i]
    pred_scores = keypoint_scores[:, :, i]
    dist = np.sqrt(np.sum(
        np.power(pick_preds[np.newaxis, :] - all_preds, 2),
        axis=2
    ))
    mask = (dist <= 1)

    kp_nums = all_preds.shape[1]
    # Define a keypoints distance
    score_dists = np.zeros([all_preds.shape[0], kp_nums])
    keypoint_scores = keypoint_scores.squeeze()
    if keypoint_scores.ndim == 1:
        keypoint_scores = np.expand_dims(keypoint_scores, axis=0)
    if pred_scores.ndim == 1:
        pred_scores = np.expand_dims(pred_scores, axis=1)
    # The predicted scores are repeated up to do broadcast
    pred_scores = np.tile(pred_scores, [1, all_preds.shape[0]]).transpose([0, 1])
    score_dists[mask] = np.tanh(pred_scores[mask] / delta1) * np.tanh(keypoint_scores[mask] / delta1)

    point_dist = np.exp((-1) * dist / delta2)
    final_dist = np.sum(score_dists, axis=1) + mu * np.sum(point_dist, axis=1)

    return final_dist


def p_merge_fast(ref_pose, cluster_preds, cluster_scores, ref_dist):
    '''
    Score-weighted pose merging
    INPUT:
        ref_pose:       reference pose          -- [kp_num, 2]
        cluster_preds:  redundant poses         -- [n, kp_num, 2]
        cluster_scores: redundant poses score   -- [n, kp_num, 1]
        ref_dist:       reference scale         -- Constant
    OUTPUT:
        final_pose:     merged pose             -- [kp_num, 2]
        final_score:    merged score            -- [kp_num]
    '''
    dist = np.sqrt(np.sum(
        np.power(ref_pose[np.newaxis, :] - cluster_preds, 2),
        axis=2
    ))

    kp_num = ref_pose.shape[0]
    ref_dist = min(ref_dist, 15)

    mask = (dist <= ref_dist)
    final_pose = np.zeros([kp_num, 2])
    final_score = np.zeros([kp_num])

    if cluster_preds.ndim == 2:
        cluster_preds = np.expand_dims(cluster_preds, axis=0)
        cluster_scores = np.expand_dims(cluster_scores, axis=0)
    if mask.ndim == 1:
        mask = np.expand_dims(mask, axis=0)

    # Weighted Merge
    masked_scores = cluster_scores.__mul__(np.expand_dims(mask.astype(np.float32), axis=-1))
    normed_scores = masked_scores / np.sum(masked_scores, axis=0)

    final_pose = np.multiply(cluster_preds, np.tile(normed_scores, [1, 1, 2])).sum(axis=0)
    final_score = np.multiply(masked_scores, normed_scores).sum(axis=0)
    return final_pose, final_score