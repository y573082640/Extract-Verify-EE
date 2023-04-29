from transformers import BertForMaskedLM, BertTokenizer, pipeline
from torch.utils.data import DataLoader, Dataset
import tqdm
import json
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VerifyDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
all_evt = ['出售/收购', '跌停', '加息', '降价', '降息', '融资', '上市', '涨价', '涨停', '发布', '获奖', '上映', '下架', '召回', '道歉', '点赞', '感谢', '会见', '探班', '夺冠', '晋级', '禁赛', '胜负', '退赛', '退役', '产子/女', '出轨', '订婚', '分手', '怀孕', '婚礼', '结婚', '离婚', '庆生', '求婚', '失联', '死亡', '罚款', '拘捕', '举报', '开庭', '立案', '起诉', '入狱', '约谈', '爆炸', '车祸', '地震', '洪灾', '起火', '坍/垮塌', '袭击', '坠机', '裁员', '辞/离职', '加盟', '解雇', '解散', '解约', '停职', '退出', '罢工', '闭幕', '开幕', '游行']

def query(textb):
    string = ""
    for evt in all_evt:
        string += "unused5 [MASK] {} unused6 ".format(evt)
    return (textb + string)

def create_verified_dataset(file):
    datas,labels,texts = [],[],[]
    with open(file,'r') as f:
        f = f.read().strip().split("\n")
        for line in f:
            evt = json.loads(line)
            evt_set = set()
            for e in evt['event_list']:
                evt_set.add(e['event_type'].split("-")[-1])
            labels.append(list(evt_set))
            datas.append(query(evt['text']))
            texts.append(evt['text'])

    return labels,datas,texts

def calculate_metric(predict, gt):
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        if entity_predict in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    return tp, fp, fn

def get_prediction(sents,model,tokenizer):
    inputs = tokenizer(
            sents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            # return_special_tokens_mask=True,
        )
    
    all_masked_pos = []
    for token_ids in inputs['input_ids']:
        masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
        all_masked_pos.append([mask.item() for mask in masked_position])
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    last_hidden_state = output[0]
    last_hidden_state = last_hidden_state.argmax(-1)
    ## shape = (batch_size,max_len,vocab_size)
    print(last_hidden_state.shape)

    ret = []
    
    for input_idx,masked_pos in enumerate(all_masked_pos):
        list_of_list =[]
        for index,mask_index in enumerate(masked_pos):
            top_label = last_hidden_state[input_idx][mask_index]
            word = tokenizer.decode(top_label.item()).strip()
            if word == '有':
                list_of_list.append(all_evt[index])
        ret.append(list_of_list)
        
    return ret

def get_p_r_f(tp, fp, fn):
    # print(tp, fp, fn)
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return [p, r, f1]

def trigger_predict(file, batch_size=32, model_path="checkpoints/ee/mlm_tri_roberta"):
    
    labels,datas,texts = create_verified_dataset(file)
    # print(datas)
    # load your model and tokenizer from a local directory or a URL
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)

    infer_data = VerifyDataset(datas)
    data_loader = DataLoader(
        dataset=infer_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )
    ret = []
    for batch_data in tqdm.tqdm(data_loader):
        ret += get_prediction(batch_data,model,tokenizer)
    tp, fp, fn = 0,0,0
    
    for i in range(len(labels)):
        tp1, fp1, fn1 = calculate_metric(ret[i],labels[i])
        tp += tp1
        fp += fp1
        fn += fn1
    score = get_p_r_f(tp, fp, fn)
    print("precision:{},recall:{},f1_score:{}".format(score[0],score[1],score[2]))


if __name__ == '__main__':
    path = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_dev.json'
    ret = trigger_predict(path)
