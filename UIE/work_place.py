from transformers import BertForMaskedLM, BertTokenizer, pipeline
import numpy as np
model_path = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/model_hub/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_path)
text = '蔚来创始人、CEO 李斌今天发布内部邮件'
# k = tokenizer.decode(3187)
# k = tokenizer.decode(3300)
# print(k)

# indices1 = np.argwhere(t1 == 3300).squeeze()
# indices2 = np.argwhere(t2 == 3300).squeeze()
# print(indices1)
# print(indices2)
# result = len(np.intersect1d(indices1, indices2))
# precision = result/len(indices1)
# recall = result/len(indices2)
# f1 = 2*recall*precision/(precision+recall)
# print(precision)
# print(recall)
# print(f1)

# indices1 = np.array([True,False])
# indices2 = np.array([False,True])
# labels = np.array([3300,3187])
# target_mask1 = labels == (tokenizer.convert_tokens_to_ids('有'))
# target_mask2 = labels == (tokenizer.convert_tokens_to_ids('无'))
# target_mask = target_mask1 | target_mask2
# indices = indices1 | indices2
print(tokenizer.tokenize(text))
print([b for b in text])