

# run mid ckpts
export CUDA_VISIBLE_DEVICES=4,5,6,7
n_gpu=4

name='02_09_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-dice'
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1

name='02_14_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_gce-lovasz'
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1

name='02_17_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_ce-lovasz'
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1

name='02_28_segformer_mit-b5_4xb2-160k_sarseg-896x896_stage1-28_ignore0_sce-lovasz'
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1

name='02_29_segformer_mit-b5_4xb2-160k_sarseg-896x896_interv3_ignore0_sce-lovasz'
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1
