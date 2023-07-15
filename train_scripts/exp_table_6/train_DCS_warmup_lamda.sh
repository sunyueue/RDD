# export CUDA_VISIBLE_DEVICES=0,1
# nproc_per_node=2

export CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=2
batch_size=16
max_iterations=40000
save_per_iters=1600

labels=(1)

warmup_iterations=(0 4000 8000)

EDD_methods=("XOR")



for label in "${labels[@]}"
do
	for warmup_iteration in "${warmup_iterations[@]}"
		do    
			for EDD_method in "${EDD_methods[@]}"
			do
				# train_deeplabv3_res18
				python -m torch.distributed.launch --nproc_per_node=${nproc_per_node} --master_port 29501\
					train_onlykd_two_stage_diff_warmup_lamda.py \
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
					--EDD_method ${EDD_method} \
					--save-dir "work_dirs/table7/deeplabv3_res18_onlykd_TwoStage_diff_warmup_lamda_bs${batch_size}_iteration_${max_iterations}_warmupiter_${warmup_iteration}_EDD_method_${EDD_method}_repeat_${label}" \
					--log-dir "work_dirs/table7/deeplabv3_res18_onlykd_TwoStage_diff_warmup_lamda_bs${batch_size}_iteration_${max_iterations}_warmupiter_${warmup_iteration}_EDD_method_${EDD_method}_repeat_${label}" \
					--teacher-pretrained pretrain/deeplabv3_resnet101_citys_best_model.pth \
					--student-pretrained-base pretrain/resnet18-imagenet.pth
			done
        done
done