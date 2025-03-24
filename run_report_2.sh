# This is an bash example for testing pipeline.

export CUDA_VISIBLE_DEVICES=0
n_gpu=1
data_root='./data/DFC2025Track1'

mkdir -p ${data_root}/test_rest/sar_images
mkdir -p ${data_root}/test_rest/labels

python copy_test_rest.py

name='03_01_upernet_convnextv2-base_4xb2-160k_sarseg-1024x1024_interv3_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  test_dataloader.batch_size=1 \
  test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="test_rest/sar_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="test_rest/labels"

# ensemble predictions and filtering valid areas.
pip install tqdm
python tools/ensemble_predict_forloop.py
