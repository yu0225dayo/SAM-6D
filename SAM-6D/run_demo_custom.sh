#!/bin/bash
# SAM-6D pose estimation (引数対応版)
#
# 使い方 (引数あり):
#   bash run_demo_custom.sh MESH_PATH OUTPUT_DIR CLICK_X CLICK_Y [RGB_PATH] [DEPTH_PATH] [CAM_PATH]
#
# 使い方 (引数なし・デバッグ用デフォルト値):
#   bash run_demo_custom.sh

set -e

PYTHON=/opt/conda/envs/sam6d/bin/python
SAM6D_DIR="$(cd "$(dirname "$0")" && pwd)"

# 引数またはデフォルト値
CAD_PATH="${1:-/workspace/tmp/server_reconstructions/object_seed42_mesh.ply}"
OUTPUT_DIR="${2:-/workspace/tmp/server_reconstructions/object_seed42_mesh_templates}"
CLICK_X="${3:-400}"
CLICK_Y="${4:-280}"
RGB_PATH="${5:-/workspace/tmp/rgb.png}"
DEPTH_PATH="${6:-/workspace/tmp/depth.png}"
CAM_PATH="${7:-$SAM6D_DIR/Data/Example/camera_custom.json}"

echo "=== 設定 ==="
echo "  CAD:        $CAD_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  CLICK:      ($CLICK_X, $CLICK_Y)"
echo "  RGB:        $RGB_PATH"
echo "  DEPTH:      $DEPTH_PATH"
echo "  CAM:        $CAM_PATH"

echo "=== Step 1: テンプレート確認 ==="
if [ ! -d "$OUTPUT_DIR/templates" ]; then
    echo "エラー: テンプレートが見つかりません: $OUTPUT_DIR/templates"
    exit 1
fi
echo "テンプレート確認済み: $OUTPUT_DIR/templates"

echo "=== Step 2: SAMクリック点マスク生成 ==="
cd "$SAM6D_DIR/Instance_Segmentation_Model"
$PYTHON create_mask_from_click.py \
    --rgb_path "$RGB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --click_x "$CLICK_X" \
    --click_y "$CLICK_Y"

echo "=== Step 3: Pose Estimation Model ==="
SEG_PATH="$OUTPUT_DIR/sam6d_results/detection_ism.json"
cd "$SAM6D_DIR/Pose_Estimation_Model"
$PYTHON run_inference_custom.py \
    --output_dir "$OUTPUT_DIR" \
    --cad_path "$CAD_PATH" \
    --rgb_path "$RGB_PATH" \
    --depth_path "$DEPTH_PATH" \
    --cam_path "$CAM_PATH" \
    --seg_path "$SEG_PATH"

echo "=== 完了 ==="
echo "結果: $OUTPUT_DIR/sam6d_results/"
