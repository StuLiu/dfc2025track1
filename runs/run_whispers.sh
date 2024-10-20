


export CUDA_VISIBLE_DEVICES=1

model=segformer
name=00_1002_segformer_mit-b0_1xb16-80k_whispers-512x512_acw

python tools/train.py configs/${model}/${name}.py


python tools/test.py \
  configs/${model}/${name}.py \
  work_dirs/${name}/iter_80000.pth \
  --out temps/${name} # --tta

python data/whispers2024/transfer_png2tif.py --dir-in temps/${name} --dir-out submits/${name}

rm -rf temps/${name}

cd ./submits/${name}

zip -r ../${name}.zip ./*