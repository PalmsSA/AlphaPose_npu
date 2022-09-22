import os
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)
import argparse
import sys

sys.path.append("../../../common/")
sys.path.append("../")

from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_model import Model
from preprocess import transform
from postprocess import get_keypoints, get_pose_boxes
from plot import plot_poses

model_path = "../model/fast_res50_256x192.om"


def main(img):
    """main"""
    # initialize acl runtime
    acl_resource = AclResource()
    acl_resource.init()
    model = Model(model_path)
    height, width = (224, 160)
    # box = np.array([[1.63000e+03, 7.35000e+02, 3.27600e+03, 5.57700e+03, 9.43208e-01, 1.00000e+00]])
    # box = [1.63000e+03, 7.35000e+02, 3.27600e+03, 5.57700e+03]
    box = [57, 64, 201, 392]
    h, w = img.shape[:2]
    inps, cropped_box = transform(img, box)
    inps = np.expand_dims(inps, axis=0)
    res = model.execute([inps, ])[0]
    keypoints = get_keypoints([0], np.array([box]), res, np.array([cropped_box]), fn=0)
    print("keypoints: ", keypoints)
    pose_labels, pose_boxes, poses = get_pose_boxes(img, keypoints)
    res = pose_labels, pose_boxes, poses

    for pose in res[2].values():
        for i in pose.items():
            print(i)
    if res is not None:
        plot_poses(img, res[2])

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 600, 800)
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite("outputs/swote.jpg", img)


if __name__ == '__main__':
    img_path = "swote.jpg"
    src: "ndarray" = cv2.imread(img_path)
    main(src)
