#!/bin/bash

# other original KD methods 
export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
batch_size=16
max_iterations=40000
save_per_iters=1600

repeats=(1 2 3)

for repeat in "${repeats[@]}"
do
    # train_at.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-kd 1.0 \
        --lambda-at 10000. \
        --data cityscapes \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "work_dirs/exp3/deeplabv3_res18_at_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --log-dir "work_dirs/exp3/deeplabv3_res18_at_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth


    # train_dsd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --lambda-psd 1000. \
        --lambda-csd 10. \
        --data cityscapes \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "work_dirs/exp3/deeplabv3_res18_dsd_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --log-dir "work_dirs/exp3/deeplabv3_res18_dsd_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth


    # train cirkd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_cirkd.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --data cityscapes \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "work_dirs/exp3/deeplabv3_res18_cirkd_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --log-dir "work_dirs/exp3/deeplabv3_res18_cirkd_bs${batch_size}_iteration${max_iterations}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth
done