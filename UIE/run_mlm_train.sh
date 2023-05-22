#!/bin/bash

pip uninstall transformers -y
pip install transformers

python run_mlm.py \
    --model_name_or_path /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/model_hub/chinese-roberta-wwm-ext \
    --train_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm_tgr.txt \
    --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm_test_exists.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_existOnly \
    --line_by_line \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --mlm_probability 0 \
    --pad_to_max_length \
    --max_seq_length 512

pip uninstall transformers -y
pip install transformers==4.5.0
