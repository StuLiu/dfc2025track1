# MMSeg for Agriculture Vision 2024

***
## Install python env
install mmsegmentation env as introduced in README.md

## Reproduction of solution
Prepare data by running: 

`python data/2024-CVPR-Agriculture-Vision2/organize_supervised_into_mmseg_RGBN.py`, 

`cp [data_path]/img_dir/val/* [data_path]/img_dir/train`

`cp [data_path]/ann_dir/val/* [data_path]/ann_dir/train`

`python data/2024-CVPR-Agriculture-Vision2/class_balanced_mosaic_augmentation_resizecrop_rgbn.py`, 

Then, training models by running 

`bash runs/run.sh`


## Generate result with trained models

Run `python tools/test.py [config-path] [ckpt-path] --out [results-submit-path] --tta`
to get single model results

for example:
    `python tools/test.py configs/segformer/06_02_segformer_mit-b5_4xb4-80k_agriculture-vision-RGBN-512x512_ACWLoss_mosaicx4_rc0.75-1.5_NoiseBlur_ClassMix.py work_dirs/06_02_segformer_mit-b5_4xb4-80k_agriculture-vision-RGBN-512x512_ACWLoss_mosaicx4_rc0.75-1.5_NoiseBlur_ClassMix/iter_80000.pth --out ./temp_res --tta`

Note that: the models will uploaded in the next week

Finally, run `python ensemble4.py` and `python ensemble-mid4.py` 
to generate submitted results and middle results.
