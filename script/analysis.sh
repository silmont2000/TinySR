cd /data/disk2/xby/TinySR
python train/pre_analysis.py \
  --output_dir outputs/act_smoothing_calib_debug \
  --max_images_per_set 0 \
  --max_qq_points 8000 \
  --max_ks_pairs_per_layer 12 \
  --mixed_precision fp16 \
  --device cuda