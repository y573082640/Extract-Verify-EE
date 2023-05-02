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
evt_dict = {"出售/收购": "财经/交易-出售/收购", "跌停": "财经/交易-跌停", "加息": "财经/交易-加息", "降价": "财经/交易-降价", "降息": "财经/交易-降息", "融资": "财经/交易-融资", "上市": "财经/交易-上市", "涨价": "财经/交易-涨价", "涨停": "财经/交易-涨停", "发布": "产品行为-发布", "获奖": "产品行为-获奖", "上映": "产品行为-上映", "下架": "产品行为-下架", "召回": "产品行为-召回", "道歉": "交往-道歉", "点赞": "交往-点赞", "感谢": "交往-感谢", "会见": "交往-会见", "探班": "交往-探班", "夺冠": "竞赛行为-夺冠", "晋级": "竞赛行为-晋级", "禁赛": "竞赛行为-禁赛", "胜负": "竞赛行为-胜负", "退赛": "竞赛行为-退赛", "退役": "竞赛行为-退役", "产子/女": "人生-产子/女", "出轨": "人生-出轨", "订婚": "人生-订婚", "分手": "人生-分手", "怀孕": "人生-怀孕", "婚礼": "人生-婚礼", "结婚": "人生-结婚", "离婚": "人生-离婚", "庆生": "人生-庆生", "求婚": "人生-求婚", "失联": "人生-失联", "死亡": "人生-死亡", "罚款": "司法行为-罚款", "拘捕": "司法行为-拘捕", "举报": "司法行为-举报", "开庭": "司法行为-开庭", "立案": "司法行为-立案", "起诉": "司法行为-起诉", "入狱": "司法行为-入狱", "约谈": "司法行为-约谈", "爆炸": "灾害/意外-爆炸", "车祸": "灾害/意外-车祸", "地震": "灾害/意外-地震", "洪灾": "灾害/意外-洪灾", "起火": "灾害/意外-起火", "坍/垮塌": "灾害/意外-坍/垮塌", "袭击": "灾害/意外-袭击", "坠机": "灾害/意外-坠机", "裁员": "组织关系-裁员", "辞/离职": "组织关系-辞/离职", "加盟": "组织关系-加盟", "解雇": "组织关系-解雇", "解散": "组织关系-解散", "解约": "组织关系-解约", "停职": "组织关系-停职", "退出": "组织关系-退出", "罢工": "组织行为-罢工", "闭幕": "组织行为-闭幕", "开幕": "组织行为-开幕", "游行": "组织行为-游行"}

def query(textb):
    string = "[SEP]"
    for i,evt in enumerate(all_evt):
        string += " [MASK] {} ".format(evt)
        if i != len(all_evt) - 1:
            string += "[SEP] "
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
            labels.append([evt_dict[e] for e in list(evt_set)])
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
    # print(last_hidden_state.shape)

    ret = []
    
    for input_idx,masked_pos in enumerate(all_masked_pos):
        list_of_list =[]
        for index,mask_index in enumerate(masked_pos):
            top_label = last_hidden_state[input_idx][mask_index]
            word = tokenizer.decode(top_label.item()).strip()
            if word == '有':
                list_of_list.append(evt_dict[all_evt[index]])
        ret.append(list_of_list)
        
    return ret

def get_p_r_f(tp, fp, fn):
    # print(tp, fp, fn)
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return [p, r, f1]

def trigger_eval_file(input_file,label_file):
    
    preds,_,_ = create_verified_dataset(input_file)
    labels,_,_ = create_verified_dataset(label_file)
    # print(datas)
    # load your model and tokenizer from a local directory or a URL

    tp, fp, fn = 0,0,0
    
    for i in range(len(labels)):
        tp1, fp1, fn1 = calculate_metric(preds[i],labels[i])
        tp += tp1
        fp += fp1
        fn += fn1
    score = get_p_r_f(tp, fp, fn)
    print("precision:{},recall:{},f1_score:{}".format(score[0],score[1],score[2]))

def trigger_eval(file, batch_size=32, model_path="/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_tri_roberta/best_model"):
    
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

def trigger_predict(file,output, batch_size=32, model_path="/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_tri_roberta/best_model"):
    datas = []
    ids = []
    with open(file,'r') as f:
        f = f.read().strip().split("\n")
        for line in f:
            evt = json.loads(line)
            datas.append(evt['text'])
            ids.append(evt['id'])
    # print(datas)
    # load your model and tokenizer from a local directory or a URL
    model = BertForMaskedLM.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)

    infer_data = VerifyDataset([query(d) for d in datas])
    data_loader = DataLoader(
        dataset=infer_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )
    results = []
    for batch_data in tqdm.tqdm(data_loader):
        results += get_prediction(batch_data,model,tokenizer)
    # print(results)
    out_list = []
    for i in range(len(datas)):
        data_tuple = {
            'text':datas[i],
            "id":ids[i],
            'event_list':[]
        }
        for evt in results[i]:
            data_tuple['event_list'].append({
                'event_type':evt
            })
        out_list.append(data_tuple)
    with open(output,"w") as fp:
        for answer in out_list:
            json.dump(answer, fp, ensure_ascii=False, separators=(',', ':'))
            fp.write("\n")

if __name__ == '__main__':
    dev = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_dev.json"
    path = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/data/ee/duee/duee_test2.json'
    out = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/log/tmp.json"
    bio_pred = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/storage/bio_pred.json"
    ret = trigger_predict(file=path,output="log/event_detection_9575.txt",batch_size=32,model_path="/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/mlm_tri_roberta 9575")
