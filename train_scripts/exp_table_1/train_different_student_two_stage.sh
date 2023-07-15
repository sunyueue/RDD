#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
nproc_per_node=4
batch_size=16
max_iterations=40000
save_per_iters=1600

# DCS with various backbone and segmentation models
labels=(1 2 3)
warmup_iterations=(4000)


for label in "${labels[@]}"
do
	for warmup_iteration in "${warmup_iterations[@]}"
		do    
            # train_deeplabv3_res18
			python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501\
				train_onlykd_two_stage_diff.py \
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
				--save-dir "work_dirs/table1/deeplabv3_res18_onlykd_TwoStage_diff_bs${batch_size}_iteration_${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--log-dir "work_dirs/table1/deeplabv3_res18_onlykd_TwoStage_diff_bs${batch_size}_iteration_${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
				--student-pretrained-base pretrain/resnet18-imagenet.pth
            
            # train_deeplabv3_res18*
			python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501\
				train_onlykd_two_stage_diff.py \
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
				--save-dir "work_dirs/table1/deeplabv3_res18*_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--log-dir "work_dirs/table1/deeplabv3_res18*_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
				--student-pretrained-base None
            
            # train_deeplabv3_mbv2
			python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501\
				train_onlykd_two_stage_diff.py \
				--teacher-model deeplabv3 \
				--student-model deeplab_mobile \
				--teacher-backbone resnet101 \
				--student-backbone mobilenev2 \
				--data cityscapes \
				--batch-size ${batch_size} \
				--max-iterations ${max_iterations} \
				--save-per-iters ${save_per_iters} \
				--val-per-iters ${save_per_iters} \
				--warmup_iter ${warmup_iteration} \
				--save-dir "work_dirs/table1/deeplabv3_MBV2_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--log-dir "work_dirs/table1/deeplabv3_MBV2_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
				--student-pretrained-base pretrain/mobilenetv2-imagenet.pth
            
            # train_pspnet_res18
			python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501\
				train_onlykd_two_stage_diff.py \
				--teacher-model deeplabv3 \
				--student-model psp \
				--teacher-backbone resnet101 \
				--student-backbone resnet18 \
				--data cityscapes \
				--batch-size ${batch_size} \
				--max-iterations ${max_iterations} \
				--save-per-iters ${save_per_iters} \
				--val-per-iters ${save_per_iters} \
				--warmup_iter ${warmup_iteration} \
				--save-dir "work_dirs/table1/pspnet_res18_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--log-dir "work_dirs/table1/pspnet_res18_onlykd_TwoStage_diff_bs${batch_size}_iteration${max_iterations}_warmupiter_${warmup_iteration}_repeat_${label}" \
				--teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
				--student-pretrained-base pretrain/resnet18-imagenet.pth
		done
done
