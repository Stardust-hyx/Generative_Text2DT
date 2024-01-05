lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.2
LR=2e-4
SEED=2022
BatchSize=8
model_name_or_path="/disk3/hyx/huggingface/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="../data/Text2DT_SFT"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./outputs"  # 填入用来存储模型的路径

CUDA_VISIBLE_DEVICES=1 python main.py \
    --do_train \
    --do_predict \
    --train_file $your_data_path/train_dev_dt_aug.json \
    --validation_file $your_data_path/test_dt.json \
    --test_file $your_data_path/test_dt.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/lora_RE_TreeS-aug-$LR-$SEED \
    --overwrite_output_dir \
    --max_source_length 225 \
    --max_target_length 300 \
    --per_device_train_batch_size $BatchSize \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 2000 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --load_best_model_at_end \
    --metric_for_best_model 'triple_tree_avg' \
    --save_steps 200 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-5 \
    --seed $SEED \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --disable_tqdm True \
