#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
batch_size=16
max_iterations=40000
save_per_iters=1600

# Repeat 3 times
# train_baseline
labels=(1 2 3)
for label in "${labels[@]}"
do
    python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} \
        train_baselinev2.py \
        --teacher-model deeplabv3 \
        --student-model deeplabv3 \
        --teacher-backbone resnet101 \
        --student-backbone resnet18 \
        --data cityscapes \
        --batch-size ${batch_size} \
        --max-iterations ${max_iterations} \
        --save-per-iters ${save_per_iters} \
        --val-per-iters ${save_per_iters} \
        --save-dir "work_dirs/exp1/deeplabv3_res18_baseline_bs${batch_size}_iteration${max_iterations}_repeat_${label}" \
        --log-dir "work_dirs/exp1/deeplabv3_res18_baseline_bs${batch_size}_iteration${max_iterations}_repeat_${label}" \
        --teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
        --student-pretrained-base pretrain/resnet18-imagenet.pth
done

