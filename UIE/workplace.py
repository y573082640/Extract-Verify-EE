from transformers import BertTokenizer
bert_dir = "model_hub/chinese-roberta-wwm-ext/"
tokenizer = BertTokenizer.from_pretrained(bert_dir)
tokenizer.add_special_tokens({'additional_special_tokens':["[TGR]","[DEMO]","[ARG]"]})
tokenizer.save_pretrained('model_hub/tmp/')
