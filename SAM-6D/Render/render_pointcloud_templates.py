"""
点群直接投影テンプレート生成 (Blenderproc不要)

42視点から PLY 点群を射影し rgb_N.png / mask_N.png / xyz_N.npy を生成する。
xyz は物体中心原点の mm単位座標で保存する (PEM が /1000 してm単位に変換する)。
"""

import os
import argparse
import numpy as np
import cv2
import open3d as o3d

RENDER_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_POSES_PATH = os.path.join(
    RENDER_DIR,
    "../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy",
)
IMG_H, IMG_W = 480, 640


def render_one_view(pts_mm, colors_rgb, R_wc, t_wc, K, img_h, img_w, splat_r):
    """
    1視点分を投影する。

    pts_mm   : (N,3) 中心原点・mm単位の点群 (xyz_mapにもそのまま保存)
    colors_rgb: (N,3) uint8
    R_wc     : (3,3) world→camera 回転
    t_wc     : (3,)  world→camera 平行移動
    """
    pts_cam = (R_wc @ pts_mm.T).T + t_wc  # (N,3) カメラ座標系

    # カメラ前方のみ
    front = pts_cam[:, 2] > 1.0
    pc = pts_cam[front]
    nc = pts_mm[front]   # mm単位の物体座標をそのまま保存
    cc = colors_rgb[front]

    if len(pc) == 0:
        return (
            np.zeros((img_h, img_w), np.uint8),
            np.zeros((img_h, img_w, 3), np.float16),
            np.zeros((img_h, img_w, 3), np.uint8),
        )

    Z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / Z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / Z + K[1, 2]

    ok = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u, v, Z, nc, cc = u[ok], v[ok], Z[ok], nc[ok], cc[ok]

    # 遠い点から順に描画して近い点が上書きするように
    order = np.argsort(-Z)
    ui = np.round(u[order]).astype(np.int32)
    vi = np.round(v[order]).astype(np.int32)
    nc = nc[order]
    cc = cc[order]

    mask    = np.zeros((img_h, img_w), np.uint8)
    xyz_map = np.zeros((img_h, img_w, 3), np.float32)
    rgb_img = np.zeros((img_h, img_w, 3), np.uint8)

    for dr in range(-splat_r, splat_r + 1):
        for dc in range(-splat_r, splat_r + 1):
            if dr * dr + dc * dc > splat_r * splat_r:
                continue
            vv = np.clip(vi + dr, 0, img_h - 1)
            uu = np.clip(ui + dc, 0, img_w - 1)
            mask[vv, uu]    = 255
            xyz_map[vv, uu] = nc
            rgb_img[vv, uu] = cc

    return mask, xyz_map.astype(np.float16), rgb_img


def render_pointcloud_templates(pcd_path, output_dir, num_views=42, splat_r=2):
    """
    Args:
        pcd_path  : 入力点群 PLY (mm単位、全点または密なダウンサンプル)
        output_dir: テンプレート保存先 (output_dir/templates/ に保存)
        num_views : 視点数 (最大42)
        splat_r   : スプラット半径 [px]

    Returns:
        save_dir (str): templates ディレクトリのパス
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float64)  # (N,3) mm

    colors_rgb = np.full((len(pts), 3), 255, dtype=np.uint8)

    # 中心を原点に移動し、最長辺が200mmになるようスケール変換
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center   = (bbox_min + bbox_max) / 2.0
    max_ext  = (bbox_max - bbox_min).max()
    scale_to_mm = 200.0 / max_ext if max_ext > 0 else 1.0
    pts_c    = ((pts - center) * scale_to_mm).astype(np.float32)  # mm単位・中心原点

    cam_poses = np.load(CAM_POSES_PATH)  # (42,4,4) camera-to-world, OpenCV convention

    # カメラ距離 = half_extent × 5 (物体が画像の ~30% を占める)
    half_ext  = max_ext / 2.0
    view_dist = 5.0 * half_ext

    # 内部パラメータ: 垂直FOV 60° (half=30°)
    fov_half_deg = 30.0
    fx = (IMG_H / 2.0) / np.tan(np.radians(fov_half_deg))
    K  = np.array([[fx, 0, IMG_W / 2.0],
                   [0, fx, IMG_H / 2.0],
                   [0,  0,          1.0]])

    save_dir = os.path.join(output_dir, "templates")
    os.makedirs(save_dir, exist_ok=True)

    print(f"[PCD Template] 点数={len(pts)}, max_extent={max_ext:.1f}mm, "
          f"view_dist={view_dist:.1f}mm, splat_r={splat_r}")

    for idx in range(min(num_views, len(cam_poses))):
        cp  = cam_poses[idx]          # camera-to-world (4×4)
        R_cw = cp[:3, :3]             # world 上のカメラ軸
        t_cw = cp[:3, 3]              # world 上のカメラ位置 (原点スケール)

        # カメラ方向を維持しつつ距離を view_dist に再スケール
        t_dir    = t_cw / (np.linalg.norm(t_cw) + 1e-8)
        t_scaled = t_dir * view_dist  # mm 単位の点群空間でのカメラ位置

        # world→camera 変換
        R_wc = R_cw.T
        t_wc = -R_wc @ t_scaled

        mask, xyz_map, rgb_img = render_one_view(
            pts_c, colors_rgb,
            R_wc, t_wc, K, IMG_H, IMG_W, splat_r,
        )

        cv2.imwrite(os.path.join(save_dir, f"rgb_{idx}.png"),
                    cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"mask_{idx}.png"), mask)
        np.save(os.path.join(save_dir, f"xyz_{idx}.npy"), xyz_map)

        n_px = int(mask.sum()) // 255
        print(f"  [{idx+1:2d}/{num_views}] visible pixels: {n_px}")

    print(f"[PCD Template] 完了: {save_dir}")
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点群直接投影テンプレート生成")
    parser.add_argument("--pcd_path",   required=True, help="入力点群 PLY (mm単位)")
    parser.add_argument("--output_dir", required=True, help="テンプレート保存先")
    parser.add_argument("--num_views",  type=int, default=42)
    parser.add_argument("--splat_r",    type=int, default=2, help="スプラット半径 [px]")
    args = parser.parse_args()

    render_pointcloud_templates(args.pcd_path, args.output_dir, args.num_views, args.splat_r)
