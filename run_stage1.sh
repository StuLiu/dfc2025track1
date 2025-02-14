

export CUDA_VISIBLE_DEVICES=2,3,4,5
n_gpu=4

name='02_14_segformer_mit-b3_4xb2-80k_oem-768x768-alld_lovasz'
bash tools/dist_train2.sh configs/_dfc_stage1/${name}.py ${n_gpu}
bash tools/dist_test2.sh configs/_dfc_stage1/${name}.py work_dirs/${name}/iter_80000.pth ${n_gpu} \
  --out submits/dfc_stage1/${name}_tta --tta
cd submits/dfc_stage1/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit

name='02_14_segformer_mit-b5_4xb2-80k_oem-768x768-alld'
bash tools/dist_train2.sh configs/_dfc_stage1/${name}.py ${n_gpu}
bash tools/dist_test2.sh configs/_dfc_stage1/${name}.py work_dirs/${name}/iter_80000.pth ${n_gpu} \
  --out submits/dfc_stage1/${name}_tta --tta
cd submits/dfc_stage1/${name}_tta || exit
zip -r ../mmseg_${name}_tta.zip ./*.png
cd /home/liuwang/liuwang_data/documents/projects/mmseg-agri || exit
