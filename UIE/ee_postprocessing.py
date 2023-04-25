import json
from utils.metrics import lcs,longest_common_substring

def remove_duplicates(data_list):
    # 遍历每个事件对象
    ret = []
    for data in data_list:
        for event in data['event_list']:
            # 遍历每个事件对象中的参数，构建字典
            filtered_arguments = {}
            for argument in event["arguments"]:
                role = argument["role"]
                del argument['event_id']
                # 如果当前role已经在字典中存在，比较当前参数的长度和字典中存储的参数的长度
                if role in filtered_arguments:
                    filtered_arguments[role].append(argument)
                # 如果当前role在字典中不存在，直接将参数添加到字典中
                else:
                    filtered_arguments[role] = [argument]

            tmp = []
            # 此处用于剔除相同的参数，并始终选择比较短的哪一个
            for key, role_list in filtered_arguments.items():
                results = []
                for i in range(len(role_list)):
                    is_duplicate = False
                    for j in range(len(role_list)):
                        if i == j:
                            continue
                        tmp_i = role_list[i]['argument']
                        tmp_j = role_list[j]['argument']
                        max_len = longest_common_substring(
                            tmp_i, tmp_j)
                        bound = min(
                            3, min(len(tmp_j), len(tmp_i)))
                        if max_len >= bound and len(tmp_i) > len(tmp_j):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        results.append(role_list[i])
                tmp.extend(results)
            event['arguments'] = tmp
        ret.append(data)
    return ret



# 新建一个字典，用于存放每个role对应的最短参数
if __name__ == "__main__":
    path = 'important log/推理 all exist 0.25 0.5.json'
    datalist = []
    with open(path,'r',encoding='utf-8') as fp:
        for d in fp:
            datalist.append(json.loads(d)) 
    ret = remove_duplicates(datalist)
    with open('log/测试融合规则.json','w') as fp:
            for answer in ret:
                json.dump(answer, fp, ensure_ascii=False, separators=(',', ':'))
                fp.write('\n')