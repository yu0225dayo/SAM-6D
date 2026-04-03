import numpy as np
import os
import cv2

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # draw top layer
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img


def draw_axes(img, R, t, intrinsics, length=0.05):
    """物体座標系の x/y/z 軸を赤/緑/青の矢印で描画する。length は軸の長さ [m]。"""
    origin = (intrinsics @ t[:, np.newaxis]).flatten()
    origin = (origin[:2] / origin[2]).astype(np.int32)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # x, y, z
    for i, c in enumerate(colors):
        axis_end_3d = t + R[:, i] * length
        end = (intrinsics @ axis_end_3d[:, np.newaxis]).flatten()
        end = (end[:2] / end[2]).astype(np.int32)
        cv2.arrowedLine(img, tuple(origin), tuple(end), c, 2, tipLength=0.3)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(255, 0, 0), with_axes=True):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 512)
    pts_3d = model_points[choose].T

    axis_length = np.max(scale) * 0.6  # 軸の長さはbboxスケールに合わせる

    for ind in range(num_pred_instances):
        # draw 3d bounding box (X=赤, Y=緑, Z=青)
        transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)
        # draw coordinate axes (x=赤, y=緑, z=青)
        if with_axes:
            draw_image_bbox = draw_axes(draw_image_bbox, pred_rots[ind], pred_trans[ind], intrinsics[ind], length=axis_length)

    return draw_image_bbox
