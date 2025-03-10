

export CUDA_VISIBLE_DEVICES=4,5,6,7
n_gpu=4
data_root="/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train"
name='02_26_uda_segformer_mit-b3_2xb4-80k_oem-768x768-alld_ignore255_dacsv2_ce_th0.968_downup'
ckpt='iter_80000.pth'
bash tools/dist_test2.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/${ckpt} \
  ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="rgb_images_part1" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels_part1" \
  test_dataloader.batch_size=1 \
  tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir=${data_root}/labels_pl_mid/${name}

bash tools/dist_test2.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/${ckpt} \
  ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="rgb_images_part2" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels_part2" \
  test_dataloader.batch_size=1 \
  tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir=${data_root}/labels_pl_mid/${name}

bash tools/dist_test2.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/${ckpt} \
  ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="rgb_images_part3" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels_part3" \
  test_dataloader.batch_size=1 \
  tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir=${data_root}/labels_pl_mid/${name}

bash tools/dist_test2.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/${ckpt} \
  ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="rgb_images_part4" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels_part4" \
  test_dataloader.batch_size=1 \
  tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir=${data_root}/labels_pl_mid/${name}

#python tools/convert_png2tif.py --dir-name ${name}_tta
