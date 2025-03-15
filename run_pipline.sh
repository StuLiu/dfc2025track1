# This is an bash example for training and testing pipeline.
# The dataset dir paths should be adjust before running in your equipment.
# higher accuracy can be gained when using ensemble strategy
# ensemble scripts at 'tools/ensemble_pseudo_labels.py' and 'tools/ensemble_predict.py'

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

# # stage-1
# train on the open-earth-map optical datasets.
name='03_02_uda_upernet_convnext-base_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup'
bash tools/dist_train.sh configs/_dfc_stage1/${name}.py ${n_gpu}
# generate pseudo labels
data_root="/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train"
ckpt='iter_80000.pth'
bash tools/dist_test.sh \
  configs/_dfc_stage1/${name}.py \
  work_dirs/${name}/${ckpt} \
  ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/${name}_tta \
  --show-dir work_dirs/${name}/vis_dir \
  --tta \
  --cfg-options test_dataloader.dataset.data_root=${data_root} \
  test_dataloader.dataset.data_prefix.img_path="rgb_images" \
  test_dataloader.dataset.data_prefix.seg_map_path="labels" \
  test_dataloader.batch_size=1 \
  tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir=${data_root}/labels_pl_mid/${name}
python tools/convert_png2tif.py --dir-name ${name}_tta
# if using ensemble strategy, you should train different models in the stage-1.
# And, running the ensemble script at 'tools/ensemble_pseudo_labels.py'

# # stage-2
# train the sar-seg task by the generate pseudo labels
name='03_07_upernet_convnextv2-base_4xb2-120k_sarseg-1024x1024_bestv2_ignore0_sce-lovasz'
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_120000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}" \
  train_dataloader.dataset.data_prefix.seg_map_path=train/labels_pl/${name}_tta \
cd submits/dfc_stage2/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
# if using ensemble strategy, you should train different models in the stage-2.
# And, running the ensemble script at 'tools/ensemble_predict.py'
