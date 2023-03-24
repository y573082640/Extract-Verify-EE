#!/bin/bash

python run_mlm.py \
    --model_name_or_path model_hub/chinese-bert-wwm-ext/ \
    --train_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/ee_obj_for_mlm.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm \
    --line_by_line \
    --save_total_limit 5
