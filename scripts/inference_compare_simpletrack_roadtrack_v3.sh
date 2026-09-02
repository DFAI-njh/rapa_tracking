#!/usr/bin/env bash
set -euo pipefail

DFAI_MMDET3D_ROOT="${DFAI_MMDET3D_ROOT:-/root/workspace/DFAI_mmdet3d}"
cd "${DFAI_MMDET3D_ROOT}"
export PYTHONPATH="${DFAI_MMDET3D_ROOT}:${PYTHONPATH:-}"
export PATH="${DFAI_MMDET3D_ROOT}/venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
DFAI_PYTHON="${DFAI_PYTHON:-${DFAI_MMDET3D_ROOT}/venv/bin/python3}"

RAPA_TRACKING_PACKAGE_DIR="$(
  "${DFAI_PYTHON}" -c 'import rapa_tracking; print(rapa_tracking.PACKAGE_ROOT)'
)"
ROADTRACK_V3_ASSOCIATION_METRIC="${ROADTRACK_V3_ASSOCIATION_METRIC:-3d_optimized}"
ROADTRACK_V3_INPUT_NMS_THRESHOLD="${ROADTRACK_V3_INPUT_NMS_THRESHOLD:-0.001}"

"${DFAI_PYTHON}" "${DFAI_MMDET3D_ROOT}/tools/rosbags/tracking_inference.py" \
  --config "${DFAI_MMDET3D_ROOT}/projects/RAPA_R/checkpoints/bosch/bosch_s4_v12_v016_noego_gauss_w01_260808/[bosch][enc_gmo_relvel_relabs_relabsxy_point_fusion][voxel016][sweep4][no_ego_motion][random_past_sweep_select4of8_apply05][rot03925][box_gaussian_loss_w01][normalized_delta_time][no_db_sampling][no_put_random_points]rapa-r_anhead_b16_260807.py" \
  --checkpoint "${DFAI_MMDET3D_ROOT}/projects/RAPA_R/checkpoints/bosch/bosch_s4_v12_v016_noego_gauss_w01_260808/best_3D mAP_bbox_3d_AP_overall_0.7_epoch_145.pth" \
  --device cuda:0 \
  --precision fp16 \
  --sweep-mode sweep \
  --sweeps-num 4 \
  --sweep-accumulation-mode concat \
  --score-thr 0.2 \
  --nms-thr 0.1 \
  --marker-lifetime 0.5 \
  --detection-clear-all \
  --tracking \
  --compare-simpletrack-roadtrack-v3 \
  --tracking-config "${DFAI_MMDET3D_ROOT}/projects/tracking/simpletrack/configs/bosch_vc_kf_giou.yaml" \
  --roadtrack-v3-config "${RAPA_TRACKING_PACKAGE_DIR}/roadtrack_v3/configs/bosch_roadtrack_v3.yaml" \
  --roadtrack-v3-association-metric "${ROADTRACK_V3_ASSOCIATION_METRIC}" \
  --roadtrack-v3-input-nms-thr "${ROADTRACK_V3_INPUT_NMS_THRESHOLD}" \
  --tracking-input-score-thr 0.2 \
  --tracking-output-mode all \
  --tracking-topic /rapar/bosch/tracks \
  --tracking-visualization \
  --tracking-visualization-config "${RAPA_TRACKING_PACKAGE_DIR}/visualization/config.yaml" \
  "$@"
