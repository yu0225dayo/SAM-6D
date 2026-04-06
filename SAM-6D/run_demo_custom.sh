#!/bin/bash
# SAM-6D 公式デモをカスタムデータで実行するスクリプト
# テンプレートはすでに生成済みのためステップ1 (blenderproc) をスキップ

set -e

PYTHON=/opt/conda/envs/sam6d/bin/python

SAM6D_DIR="$(cd "$(dirname "$0")" && pwd)"

# MESH="obj_000005" を指定するとCADモデル、"object_seed42_mesh" でSAM-3D生成メッシュ
MESH="obj_000005"  # 例: "obj_000005" または "object_seed42_mesh"
MESH="object_seed42_mesh"

if [ "$MESH" = "obj_000005" ]; then
    OUTPUT_DIR="/workspace/tmp/obj_000005_templates"
    CAD_PATH="/workspace/tmp/obj_000005.ply"
else
    OUTPUT_DIR="/workspace/tmp/server_reconstructions/${MESH}_templates"
    CAD_PATH="/workspace/tmp/server_reconstructions/${MESH}.ply"
fi
RGB_PATH="/workspace/tmp/rgb.png"
DEPTH_PATH="/workspace/tmp/depth.png"
CAM_PATH="$SAM6D_DIR/Data/Example/camera_custom.json"
TEMPLATE_DIR="$OUTPUT_DIR/templates"
SEGMENTOR_MODEL=sam
CLICK_X=400
CLICK_Y=280

echo "=== Step 1: テンプレートレンダリング (スキップ - 既存テンプレートを使用) ==="
mkdir -p "$OUTPUT_DIR"
if [ ! -e "$OUTPUT_DIR/templates" ]; then
    ln -s "$TEMPLATE_DIR" "$OUTPUT_DIR/templates"
    echo "テンプレートをリンク: $TEMPLATE_DIR -> $OUTPUT_DIR/templates"
else
    echo "テンプレート確認済み: $OUTPUT_DIR/templates"
fi

echo "=== Step 2: SAMクリック点マスク生成 (ISMスコアリングをスキップ) ==="
cd "$SAM6D_DIR/Instance_Segmentation_Model"
$PYTHON create_mask_from_click.py \
    --rgb_path $RGB_PATH \
    --output_dir $OUTPUT_DIR \
    --click_x $CLICK_X \
    --click_y $CLICK_Y

echo "=== Step 3: Pose Estimation Model ==="
SEG_PATH="$OUTPUT_DIR/sam6d_results/detection_ism.json"
cd "$SAM6D_DIR/Pose_Estimation_Model"
$PYTHON run_inference_custom.py \
    --output_dir $OUTPUT_DIR \
    --cad_path $CAD_PATH \
    --rgb_path $RGB_PATH \
    --depth_path $DEPTH_PATH \
    --cam_path $CAM_PATH \
    --seg_path $SEG_PATH

echo "=== 完了 ==="
echo "結果: $OUTPUT_DIR/sam6d_results/"
