import numpy as np
from transforms import heatmap_to_coord_simple
from nPose_nms import pose_nms
from const import joint_pairs, human_keypoint_labels


def get_pose_boxes(img, pose, need_keypoints='all'):
    h, w = img.shape[:2]
    labels = []
    boxes = []
    poses = {}
    if pose is not None and len(pose['result']) > 0:
        kp_num = len(pose['result'][0]['keypoints'])
        assert kp_num == 17
        for human in pose['result']:
            part_line = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            ren_src_index = human['index']
            # cur_pose = {}
            # 颈部关键点通过计算得出
            kp_preds = np.concatenate((kp_preds, np.expand_dims((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = np.concatenate((kp_scores, np.expand_dims((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
            # 关键点
            for n in range(kp_scores.shape[0]):
                if (need_keypoints != 'all' and human_keypoint_labels[n] not in need_keypoints) \
                        or kp_scores[n] <= 0.4:  # 移除不检测或置信度过低的关键点
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                # cur_pose[human_keypoint_labels[n]] = (cor_x, cor_y)
                boxes.append([max(0, cor_x - 15), max(0, cor_y - 20), min(w, cor_x + 15), min(h, cor_y + 20)])
                labels.append(human_keypoint_labels[n])
                part_line[n] = (cor_x, cor_y)  # 有效的关键点
            # 关键点之间的连线
            poses[ren_src_index] = part_line
    return labels, boxes, poses


def get_keypoints(ren_indexes, boxes, hm_data, cropped_boxes, fn=0):
    # 暂时先构造的scores和ids,作为后续pose_nms的参数
    min_box_area = 0
    scores = np.ones(len(boxes))
    ids = np.zeros(scores.shape)
    eval_joints = list(range(17))
    norm_type = None
    hm_size = [64, 48]
    if boxes is None or len(boxes) == 0:
        return None
    else:
        # location prediction (n, kp, 2) | score prediction (n, kp, 1)
        assert hm_data.ndim == 4
        if hm_data.shape[1] == 136:
            eval_joints = [*range(0, 136)]
        elif hm_data.shape[1] == 26:
            eval_joints = [*range(0, 26)]
        pose_coords = []
        pose_scores = []

        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = heatmap_to_coord_simple(hm_data[i][eval_joints], bbox, hm_shape=hm_size,
                                                             norm_type=norm_type)
            pose_coords.append(np.expand_dims(pose_coord, axis=0))
            pose_scores.append(np.expand_dims(pose_score, axis=0))
        preds_img = np.concatenate(pose_coords)
        preds_scores = np.concatenate(pose_scores)
        # print(preds_img)
        # print("------------------")
        # print(preds_scores)
        boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(boxes, scores, ids, preds_img,
                                                                         preds_scores, min_box_area)
        _result = []
        for k in range(len(scores)):
            _result.append(
                {
                    'keypoints': preds_img[k],
                    'kp_score': preds_scores[k],
                    'proposal_score': np.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx': ids[k],
                    'bbox': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]],
                    'index': ren_indexes[k]
                }
            )

        result = {
            'img': str(fn) + '.jpg',
            'result': _result
        }
    return result
