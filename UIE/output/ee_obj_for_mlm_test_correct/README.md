---
tags:
- generated_from_trainer
model-index:
- name: ee_obj_for_mlm_test_correct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ee_obj_for_mlm_test_correct

This model is a fine-tuned version of [/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_mask_40](https://huggingface.co//home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_mask_40) on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.1791
- eval_accuracy: 0.9341
- eval_runtime: 23.6915
- eval_samples_per_second: 254.268
- eval_steps_per_second: 7.978
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.27.3
- Pytorch 1.9.0+cu102
- Datasets 2.10.1
- Tokenizers 0.13.2
