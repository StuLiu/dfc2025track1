

export CUDA_VISIBLE_DEVICES=0,1
n_gpu=2

name='02_12_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_gce-dice'
#bash tools/dist_train.sh configs/segformer/${name}.py ${n_gpu}
bash tools/dist_test.sh configs/segformer/${name}.py work_dirs/${name}/iter_160000.pth ${n_gpu} \
  --out submits/${name}_tta --tta
cd submits/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

#name='02_11_segformer_mit-b3_4xb2-40k_sarseg-768x768_official_proto-inter0k'
#bash tools/dist_train.sh configs/segformer/${name}.py ${n_gpu}
#bash tools/dist_test.sh configs/segformer/${name}.py work_dirs/${name}/iter_40000.pth ${n_gpu} \
#  --out submits/${name}_tta --tta
#cd submits/${name}_tta || exit
#zip -r ../mmseg_${name}_tta.zip ./*.png
#cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit