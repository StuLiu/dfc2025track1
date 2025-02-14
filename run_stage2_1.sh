

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

name='02_14_segformer_mit-b3_4xb2-40k_sarseg-768x768_official_lovasz'
bash tools/dist_train2.sh configs/_dfc_stage2/${name}.py ${n_gpu}
bash tools/dist_test2.sh configs/_dfc_stage2/${name}.py work_dirs/${name}/iter_40000.pth ${n_gpu} \
  --out submits/_dfc_stage2/${name}_tta --tta
cd submits/_dfc_stage2/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

#name='02_11_segformer_mit-b3_4xb2-40k_sarseg-768x768_official_proto-inter0k'
#bash tools/dist_train.sh configs/segformer/${name}.py ${n_gpu}
#bash tools/dist_test.sh configs/segformer/${name}.py work_dirs/${name}/iter_40000.pth ${n_gpu} \
#  --out submits/${name}_tta --tta
#cd submits/${name}_tta || exit
#zip -r ../mmseg_${name}_tta.zip ./*.png
#cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit