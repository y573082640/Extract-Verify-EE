#!/bin/bash

pip uninstall transformers -y
pip install transformers

python run_mlm.py \
    --model_name_or_path /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_exist_roberta \
    --validation_file /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/mlm/ee_obj_for_mlm_test.txt \
    --per_device_eval_batch_size 32 \
    --do_eval \
    --output_dir /home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/eval_logs \
    --line_by_line \
    --mlm_probability 0 \
    --pad_to_max_length \
    --max_seq_length 512

pip uninstall transformers -y
pip install transformers==4.5.0
