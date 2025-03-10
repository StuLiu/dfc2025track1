

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

name='03_02_uda_upernet_swinv2-base_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup'
bash tools/dist_train_resume.sh configs/_dfc_stage1/${name}.py ${n_gpu}
#bash tools/dist_test.sh configs/_dfc_stage1/${name}.py work_dirs/${name}/iter_80000.pth ${n_gpu} \
#  --out submits/dfc_stage1/${name}_tta --tta
#cd submits/dfc_stage1/${name}_tta || exit
#zip -r ../mmseg_${name}_tta.zip ./*.png
#cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

bash run_generate_pseudo_labels2.sh
#python data/dfc2025/intersection_pesudo_labels.py
#bash run_stage2.sh