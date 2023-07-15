#!/bin/bash

# other KD methods integrate with DCS
export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
batch_size=16
max_iterations=40000
save_per_iters=1600
warmup_iteration=4000

repeats=(1 2 3)

for repeat in "${repeats[@]}"
do
    # train_at.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd_two_stage_diff.py \
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
        --warmup_iter ${warmup_iteration} \
        --save-dir "work_dirs/exp5/deeplabv3_res18_at_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --log-dir "work_dirs/exp5/deeplabv3_res18_at_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth


    # train_dsd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_other_kd_two_stage_diff.py \
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
        --warmup_iter ${warmup_iteration} \
        --save-dir "work_dirs/exp5/deeplabv3_res18_dsd_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --log-dir "work_dirs/exp5/deeplabv3_res18_dsd_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth


    # train cirkd.sh
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_cirkd_two_stage_diff.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --data cityscapes \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --warmup_iter ${warmup_iteration} \
        --save-dir "work_dirs/exp5/deeplabv3_res18_cirkd_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --log-dir "work_dirs/exp5/deeplabv3_res18_cirkd_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${repeat}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth
done