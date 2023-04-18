import json


def longest_common_substring(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0] * n for _ in range(m)]
    max_len = 0
    for i in range(m):
        for j in range(n):
            if str1[i] == str2[j]:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0
    # print(str1, str2, max_len)
    return max_len


path = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/important log/推理 all exist 0.25 0.5.json'
output = 'log/剔除重复.json'


def check_dump(path):
    cnt = 0
    cnt2 = 0
    total = 0
    with open(path) as f:
        for line in f:
            event = json.loads(line.strip())
            for event_obj in event["event_list"]:
                count = {}
                for argument in event_obj["arguments"]:

                    role = argument["role"]
                    if role in count:
                        count[role].append(argument['argument'])
                    else:
                        count[role] = [argument['argument']]
                        total += 1

                for key, value in count.items():
                    if len(value) >= 2:
                        # print(key)
                        # print(value)
                        # print("--------------------------")
                        cnt += 1

    print(cnt, total)
    print(cnt/total)

# 新建一个字典，用于存放每个role对应的最短参数


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


# if __name__ == "__main__":
#     remove_duplicates(path, output)
#     check_dump(path)
#     check_dump(output)
