

export CUDA_VISIBLE_DEVICES=4,5,6,7
n_gpu=4

name='02_08_segformer_mit-b3_4xb2-40k_sarseg-1024x1024_official'
bash tools/dist_train.sh configs/segformer/${name}.py ${n_gpu}
bash tools/dist_test.sh configs/segformer/${name}.py work_dirs/${name}/iter_40000.pth ${n_gpu} \
  --out submits/${name}_tta --tta
cd submits/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

#name='02_08_segformer_mit-b5_4xb2-40k_sarseg-768x768_official'
#bash tools/dist_train.sh configs/segformer/${name}.py ${n_gpu}
#bash tools/dist_test.sh configs/segformer/${name}.py work_dirs/${name}/iter_40000.pth ${n_gpu} \
#  --out submits/${name}_tta --tta
#cd submits/${name}_tta || exit
#zip -r ../mmseg_${name}_tta.zip ./*.png
#cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit