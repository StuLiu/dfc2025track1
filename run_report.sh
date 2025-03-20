# This is an bash example for testing pipeline.

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4
data_root='./data/DFC2025Track1'


# generate placeholders of the test labels
python data/dfc2025/generate_zero_labels.py \
  --input_directory ${data_root}/test/sar_images \
  --output_directory ${data_root}/test/labels


# generate valid mask for test images. (zeros for black invalid area, while ones for valid area)
name='02_07_segformer_mit-b3_4xb2-10k_saredge-1024x1024'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py \
  work_dirs/${name}/iter_10000.pth ${n_gpu} \
  --out ${data_root}/test/edge_mask --tta \
  --show-dir work_dirs/${name}/vis \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test/labels" \
  test_dataloader.batch_size=1


# generate landcover predictions by our models.
name='02_28_segformer_mit-b5_4xb2-160k_sarseg-896x896_stage1-28_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1 \
  test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test/labels"

name='02_29_segformer_mit-b5_4xb2-160k_sarseg-896x896_interv3_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1 \
  test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test/labels"

name='03_01_upernet_convnextv2-base_4xb2-160k_sarseg-1024x1024_interv3_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1 \
  test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test/labels"

name='03_01_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_interv3_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1 \
  test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test/labels"


# ensemble predictions and filtering valid areas.
python tools/ensemble_predict.py
