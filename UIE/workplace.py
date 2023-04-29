from transformers import BertTokenizer
bert_dir = "model_hub/chinese-roberta-wwm-ext/"
tokenizer = BertTokenizer.from_pretrained(bert_dir)
tokenizer.add_special_tokens({'additional_special_tokens':["[TGR]","[DEMO]","[ARG]"]})
tokenizer.save_pretrained('model_hub/tmp/')
import numpy as np
import pandas as pd
def apply_label_smoothing(label_list, factor):
    smoothed_labels = np.full((len(label_list), ), factor / (len(label_list) - 1))
    true_label_indices = np.nonzero(label_list)[0]
    smoothed_labels[true_label_indices] = 1 - factor
    return smoothed_labels.tolist()
import numpy as np

def apply_label_smoothing_sigmoid(label_list, factor):
    smoothed_labels = np.zeros(len(label_list))
    true_label_indices = np.nonzero(label_list)[0]
    smoothed_labels[true_label_indices] = 1 - factor
    non_true_label_indices = np.nonzero(label_list == 0)[0]
    smoothed_labels[non_true_label_indices] = factor / (len(label_list) - len(true_label_indices))
    smoothed_labels[non_true_label_indices] = 0.1 * np.exp(smoothed_labels[non_true_label_indices]) / \
        (1 + np.exp(smoothed_labels[non_true_label_indices]))
    return smoothed_labels.tolist()

label_list = [0] * 100
label_list[5] = 1
label_list[15] = 1
label_list[50] = 1
factor = 0.1
smoothed_labels = apply_label_smoothing(label_list, factor)
print(smoothed_labels)

train_df = pd.DataFrame(
    columns=["step", "train loss", "eval loss", "precision", "recall"]
)
train_df.to_csv(
    f"/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/log/train_log/train_loss_.csv",
    index=False,
)

import json
with open('/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/replace_set.json','r') as fp:
    d = json.load(fp)
    print(d.keys())