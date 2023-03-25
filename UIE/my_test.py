# from transformers import BertTokenizer
# import torch
# gaz_dict = {
#     50: "./data/embs/ctb.50d.vec",
#     100: "./data/embs/tencent-d100/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt",
#     200: "./data/embs/tencent-d200/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
# }

# bert_dir = "model_hub/chinese-bert-wwm-ext/"
# tokenizer = BertTokenizer.from_pretrained(
#     bert_dir, add_special_tokens=True, do_lower_case=False)

# text = '又一起 sptgr 坠机事故 sptgr 发生，美国居民区燃起熊熊大火，机上人员已经丧生。[SEP]问题：前文包含坠机事件发生的时间,包含年、月、日、天、周、时、分、秒等吗？答案： unused4 unused5 不unused6 unused7 。'
# encode_dict = tokenizer.encode(text)
# decode_dict = tokenizer.decode(encode_dict)
# word_id = tokenizer.convert_tokens_to_ids("hello")
# k = torch.full((3,5),0.15)
# msk = torch.zeros((3,5))
# msk[0,0] = 4
# msk[1,1] = 4
# msk[2,2] = 4
# msk = msk.eq(4)
# msk = torch.roll(msk,1,-1)
# k = k.masked_fill_(msk,1.0)
# print(k)
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import json
import tqdm

# load your model and tokenizer from a local directory or a URL
model = AutoModelForMaskedLM.from_pretrained("checkpoints/ee/mlm_label")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/ee/mlm_label")

# create an unmasker object with the model and tokenizer
unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=1)

# open the .txt file with the texts to inference
with open("/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/ee_obj_for_mlm_infer_exists.txt", "r") as f:
    # create an empty list to store the results
    results = []
    # create an empty list to store the current batch of texts
    batch = []
    # loop over each line in the file
    for line in tqdm.tqdm(f.readlines()):
        # strip the newline character
        line = line.strip()
        # append the line to the current batch
        if len(line) > 0 and '[MASK]' in line:
            batch.append(line)
        # check if the batch size is reached
        if len(batch) == 32:
            
            # call the unmasker on the batch
            result = unmasker(batch)
            # append the result to the list
            results.extend(result)
            # clear the current batch
            batch = []
    # check if there are any remaining texts in the batch
    if len(batch) > 0:
        # call the unmasker on the remaining batch
        result = unmasker(batch)
        # append the result to the list
        results.extend(result)

# optionally, write the results to another file
with open("output/result_ee_obj_for_mlm_label_test_exists.json", "w") as f:
    # loop over each result in the list
    cnt = 0
    total = 0
    for result in results:
        # write the result as a string to the file
        total += 1
        if result[0]['score'] < 0.7:
            cnt += 1
            result[0]['sequence'] = result[0]['sequence'].replace(" ", "").replace(
                "unused4unused5", "").replace("unused6unused7", "").replace("sptgr", "").replace("sparg", "")
            json.dump(result[0], f, ensure_ascii=False, separators=(',', ':'))
            f.write("\n")
    f.write("得分是：")
    f.write(str(cnt/total))
    f.write("\n")
