

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

name='03_07_upernet_convnextv2-base_4xb2-120k_sarseg-1024x1024_bestv2_ignore0_sce-lovasz'
#bash tools/dist_train3.sh configs/_dfc_stage2/${name}.py ${n_gpu}
bash tools/dist_test3.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_120000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}"
cd submits/dfc_stage2/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit


