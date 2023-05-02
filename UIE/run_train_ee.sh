#!/bin/bash
pip uninstall transformers -y
pip install transformers

python run_mlm.py \
    --model_name_or_path /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/model_hub/chinese-roberta-wwm-ext \
    --train_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm.txt \
    --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm_test.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --metric_for_best_model precision \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_exist_roberta \
    --line_by_line \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 100 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --mlm_probability 0 \
    --gradient_accumulation_steps 2 \
    --pad_to_max_length \
    --max_seq_length 512

pip uninstall transformers -y
pip install transformers==4.5.0

# python run_mlm.py \
#     --model_name_or_path /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_tri_roberta/best_model \
#     --train_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm_for_tri/ee_tri_for_mlm.txt \
#     --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm_for_tri/ee_tri_for_mlm_test.txt \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 16 \
#     --num_train_epochs 10 \
#     --do_eval \
#     --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_exist_roberta \
#     --line_by_line \
#     --save_total_limit 1 \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --logging_steps 100 \
#     --save_steps 100 \
#     --mlm_probability 0 \
#     --gradient_accumulation_steps 2 \
#     --pad_to_max_length \
#     --max_seq_length 512
