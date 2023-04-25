import json
def check_length(path):
    cnt = 0
    total = 0
    with open(path) as f:
        for line in f:
            event = json.loads(line.strip())
            text = event['text']
            e_len = len(event["event_list"])
            total += 1
            if len(text) <= 18 and e_len == 3:
                cnt += 1
                print("文本:%s" % text)
                print("长度:%d" %len(text))
                print('----------------------------')

    print(cnt, total)
    print(cnt/total)

def check_repeat(path):
    cnt = 0
    total = 0
    with open(path) as f:
        for line in f:
            event = json.loads(line.strip())
            text = event['text']
            for event_obj in event["event_list"]:
                for argument in event_obj["arguments"]:
                    argu = argument['argument']
                    role = argument['role']
                    total += 1
                    if text.count(argu) >= 2:
                        cnt += 1
                        print("文本:%s" % text)
                        print("论元:%s - %s" %(role,argu))
                        print("次数:%s" %text.count(argu))
                        print("长度:%d" %len(text))
                        print('----------------------------')

    print(cnt, total)
    print(cnt/total)


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
                    total += 1
                    
                    if role in count:
                        count[role].append(argument['argument'])
                    else:
                        count[role] = [argument['argument']]
                        

                for key, value in count.items():
                    if len(value) >= 2:
                        print(key)
                        print(value)
                        print("--------------------------")
                        cnt += len(value)

    print(cnt, total)
    print(cnt/total)

# 新建一个字典，用于存放每个role对应的最短参数
if __name__ == "__main__":
    path = '/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/log/output 2023-04-25 10:05:40.json'
    # path='/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/log/verified_result 2023-04-25 11:18:06.json'
    check_dump(path)

    ## 6613 69952
    ## 0.09453625343092406

    ## 3300 68250
    ## 0.04835164835164835