

export CUDA_VISIBLE_DEVICES=4,5,6,7
n_gpu=4

name='02_20_uda_segformer_mit-b3_4xb2-80k_oem-768x768-alld_ignore255_dacs_ce_th0.968'
bash tools/dist_test2.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/iter_80000.pth \
  ${n_gpu} \
  --out pseudo_labels/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root="/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train" \
  test_dataloader.dataset.data_prefix.img_path="rgb_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels" \
  test_dataloader.batch_size=1
