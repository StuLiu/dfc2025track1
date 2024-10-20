


export CUDA_VISIBLE_DEVICES=4,5,6,7

model=segformer
name=00_0930_segformer_mit-b0_4xb4-80k_whispers-1024x1024

bash tools/dist_train.sh configs/${model}/${name}.py 4

bash tools/dist_test.sh \
  configs/${model}/${name}.py \
  work_dirs/${name}/iter_80000.pth 4 \
  --out temps/${name} # --tta

python data/whispers2024/transfer_png2tif.py --dir-in temps/${name} --dir-out submits/${name}

rm -rf temps/${name}

cd ./submits/${name}

zip -r ../${name}.zip ./*