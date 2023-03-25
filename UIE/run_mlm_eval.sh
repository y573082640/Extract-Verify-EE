#!/bin/bash
python run_mlm.py \
    --model_name_or_path /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_label \
    --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/ee_obj_for_mlm_test_exists.txt \
    --per_device_eval_batch_size 32 \
    --do_eval \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/output/ee_obj_for_mlm_test_exists \
    --line_by_line \
    --evaluation_strategy steps \
    --mlm_probability 0
