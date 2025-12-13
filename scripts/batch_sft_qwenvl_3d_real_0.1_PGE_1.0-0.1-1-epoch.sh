# conda init
conda activate trl

cd /home/gaoj/share4/vlm-finetune/trl/src

MODEL_PATH="/home/gaoj/share4/vlm-finetune/trl/ctrate-qwenvl-32-364-rg-my-batch-0.1_PGE_1.0-0.1-3-epoch"
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,2 \
accelerate launch \
    --num_processes $nproc_per_node \
    --config_file /home/gaoj/share4/trl/trl/accelerate_configs/zero2.yaml \
    /home/gaoj/share4/vlm-finetune/trl/src/batch_sft_qwenvl_3d-1.0-0.1.py \
    --model_name_or_path /hdd/shiym/ckpts/cache_dir_hf/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset_name /home/gaoj/share4/vlm-finetune/trl/ctrate/train_rg_0.1.json \
    --output_dir $MODEL_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --gradient_checkpointing True \
    --logging_strategy steps \
    --logging_steps 1 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 
    # --alpha 1.0 \
    # --beta 0.01


CUDA_VISIBLE_DEVICES=0,2 \
python /home/gaoj/share4/vlm-finetune/trl/src/infer_multiturn.py \
    --model $MODEL_PATH \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset_name /hdd/shiym/datasets_processed/vlm-finetune/swift/ctrate/valid_rg.json \
    --result_path $MODEL_PATH/eval/output.json \
    --max_new_tokens 512 \
    --num_beams 1 \
    --temperature 0 \
    --batch_size 1
