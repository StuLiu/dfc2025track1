

export CUDA_VISIBLE_DEVICES=0,1,2,3
n_gpu=4

name='02_07_segformer_mit-b3_4xb2-10k_saredge-1024x1024'
bash tools/dist_train.sh configs/_dfc_stage2/${name}.py ${n_gpu}
bash tools/dist_test.sh configs/_dfc_stage2/${name}.py \
  work_dirs/${name}/iter_10000.pth ${n_gpu} \
  --out /home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/test/edge_mask --tta \
  --show-dir work_dirs/${name}/vis