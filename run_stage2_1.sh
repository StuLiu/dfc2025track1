

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

name='03_07_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_bestv2_ignore0_sce-lovasz'
#bash tools/dist_train.sh configs/_dfc_stage2/${name}.py ${n_gpu}
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/dfc_stage2/${name}_tta --tta \
  --show-dir submits/dfc_stage2/${name}_tta_vis \
  --cfg-options tta_model.type="SegTTAModelV2" \
  tta_model.save_mid_dir="submits_mid/dfc_stage2/${name}"
cd submits/dfc_stage2/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

#scp -P 24576 root@connect.westc.gpuhub.com:/root/mmseg-agri/work_dirs/03_07_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_bestv2_ignore0_sce-lovasz/iter_160000.pth /home/liuwang/liuwang_data/documents/projects/mmseg-agri/work_dirs/03_07_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_bestv2_ignore0_sce-lovasz
