#!/bin/bash

pip uninstall transformers -y
pip install transformers==4.5.0
python ee_main.py

pip uninstall transformers -y
pip install transformers
python run_mlm.py \
    --model_name_or_path model_hub/chinese-bert-wwm-ext/ \
    --train_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm.txt \
    --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm_test.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_label \
    --line_by_line \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --gradient_accumulation_steps 16 \
    --logging_steps 100 \
    --save_steps 100 \
    --mlm_probability 0

pip uninstall transformers -y
pip install transformers==4.5.0