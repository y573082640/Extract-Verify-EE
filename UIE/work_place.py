import pickle
import json


def dump_sample(file_name):
    with open(file_name, "rb") as filehandle:
        # read the data as binary data stream
        train_data = pickle.load(filehandle)

    train_dict = {}
    for line in train_data:
        text, label = line[0], line[1]
        train_dict[text] = True

    output = []
    with open("data/ee/duee/duee_train.json", "r") as filehandle:
        # read the data as binary data stream
        for line in filehandle:
            d = json.loads(d)
            if d["text"] in train_data:
                output.append(d)

    with open("data/ee/duee/{}".format(file_name), "r") as filehandle:
        for line in output:
            json.dump(line, filehandle, ensure_ascii=False, separators=(",", ":"))
            filehandle.write("\n")


import json
import random

infer_path = 'data/ee/duee/duee_test2.json'
demo_path = "demo/event_extraction_data.json"
datas = []
with open(infer_path,"r") as fp:
    for line in fp:
        datas.append(json.loads(line))
    samples = random.sample(datas,50)
with open(demo_path,"a+") as fp:
    for sample in samples:
        json.dump(sample, fp, ensure_ascii=False, separators=(",", ":"))
        fp.write("\n")
