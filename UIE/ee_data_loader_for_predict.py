import json
import logging
from config import get_tokenizer
from ee_data_loader import ListDataset, convert_list_to_tensor
from utils.question_maker import (
    creat_demo,
    creat_argu_token,
)
from utils.lexicon_functions import (
    generate_instance_with_gaz,
    batchify_augment_ids,
)
from utils.question_maker import (
    create_role_tuple_for_predict,
)

NULLKEY = "-null-"


class EeDatasetPredictor(ListDataset):
    def process_list_data(self,data_list):
        """
        event = {
            "event_type":e_type,
            'text':text,
            'trigger':trg_tuple[0],
            'trigger_start_index':trg_tuple[1],
            'event_id':text_id+"——"+str(i), # 聚合事件论元 和 聚合事件列表
        }
        """
        ret = []
        if self.args.aug_mode=='demo':
            sim_scorer = self.args.sim_scorer
            text_embs, text_tuples = sim_scorer.create_embs_and_tuples(
                data_list, label2role=self.args.label2role
            )
            demo_embs = self.args.demo_embs
            demo_tuples = self.args.demo_tuples
            most_sim = sim_scorer.sim_match(
                text_embs, demo_embs, rank=0
            )  # {corpus_id,score} rank表示取最相似的还是次相似的
            for idx, text_tuple in enumerate(text_tuples):
                sim_id = most_sim[idx]["corpus_id"]
                demo = creat_demo(demo_tuples[sim_id])
                obj_tokens, token_type_ids = creat_argu_token(
                    text_tuple, demo, self.args.max_seq_len
                )
                if self.args.use_lexicon:
                    augment_Ids = generate_instance_with_gaz(
                        obj_tokens,
                        self.args.pos_alphabet,
                        self.args.word_alphabet,
                        self.args.biword_alphabet,
                        self.args.gaz_alphabet,
                        self.args.gaz_alphabet_count,
                        self.args.gaz,
                        self.args.max_seq_len,
                    )
                    augment_Ids = batchify_augment_ids(
                        [augment_Ids], self.args.max_seq_len
                    )
                else:
                    augment_Ids = []

                ret.append(
                    {
                        "obj_tokens": obj_tokens,
                        "token_type_ids": token_type_ids,
                        "augment_Ids": augment_Ids,
                        "event_id": text_tuple["event_id"],
                        "role": text_tuple["role"],
                    }
                )
        else:
            text_tuples, _ = create_role_tuple_for_predict(
                data_list, label2role=self.args.label2role
            )
            for idx, text_tuple in enumerate(text_tuples):
                obj_tokens, token_type_ids = creat_argu_token(
                    text_tuple, None, self.args.max_seq_len
                )
                if self.args.use_lexicon:
                    augment_Ids = generate_instance_with_gaz(
                        obj_tokens,
                        self.args.pos_alphabet,
                        self.args.word_alphabet,
                        self.args.biword_alphabet,
                        self.args.gaz_alphabet,
                        self.args.gaz_alphabet_count,
                        self.args.gaz,
                        self.args.max_seq_len,
                    )
                    augment_Ids = batchify_augment_ids(
                        [augment_Ids], self.args.max_seq_len
                    )
                else:
                    augment_Ids = []

                ret.append(
                    {
                        "obj_tokens": obj_tokens,
                        "token_type_ids": token_type_ids,
                        "augment_Ids": augment_Ids,
                        "event_id": text_tuple["event_id"],
                        "role": text_tuple["role"],
                    }
                )

        logging.info("data_list数据预处理完成")
        return ret

    def load_data(self, filename):
        ret = []
        if "ner" == self.args.task:
            with open(filename, encoding="utf-8") as f:
                f = f.read().strip().split("\n")
                for d in f:
                    d = json.loads(d)
                    text = d["text"]
                    id = d["id"]

                    # 简单预处理
                    text = text.replace("\n", "。")
                    if text[-1] != "。":
                        text += "。"

                    tokens_b = [i for i in text]

                    if self.args.use_lexicon:
                        augment_Ids = generate_instance_with_gaz(
                            ["CLS"] + tokens_b[: self.args.max_seq_len - 2] + ["SEP"],
                            self.args.pos_alphabet,
                            self.args.word_alphabet,
                            self.args.biword_alphabet,
                            self.args.gaz_alphabet,
                            self.args.gaz_alphabet_count,
                            self.args.gaz,
                            self.args.max_seq_len,
                        )
                    else:
                        augment_Ids = []

                    ret.append(
                        {
                            "ner_tokens": tokens_b,
                            "text_id": id,
                            "augment_Ids": augment_Ids,
                        }
                    )

        elif "obj" == self.args.task:
            # obj直接读取list了
            pass

        return ret


class EeCollatePredictor:
    def __init__(self, max_len,  task, args):
        self.maxlen = max_len
        self.tokenizer = get_tokenizer(args)
        self.task = task
        self.args = args

    def collate_fn(self, batch):
        batch_ner_token_ids = []
        batch_ner_attention_mask = []
        batch_ner_token_type_ids = []
        batch_obj_token_ids = []
        batch_obj_attention_mask = []
        batch_obj_token_type_ids = []
        batch_argu_roles = []  # 用于构建提交文件
        batch_raw_tokens = []  # 用于构建输出
        batch_text_ids = []  # 用于构建提交文件
        batch_augment_Ids = []

        for i, data in enumerate(batch):
            augment_Ids = data["augment_Ids"]
            batch_augment_Ids.append(augment_Ids)

            if "ner" == self.task:
                tokens = data["ner_tokens"]
                ids = data["text_id"]
                encode_dict = self.tokenizer.encode_plus(
                    text=tokens,
                    max_length=self.args.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                )
                ner_token_ids = encode_dict["input_ids"]
                ner_token_type_ids = encode_dict["token_type_ids"]
                ner_attention_mask = encode_dict["attention_mask"]

                batch_ner_token_ids.append(ner_token_ids)
                batch_ner_attention_mask.append(ner_attention_mask)
                batch_ner_token_type_ids.append(ner_token_type_ids)
                batch_raw_tokens.append(tokens)
                batch_text_ids.append(ids)

            elif "obj" == self.task:
                obj_tokens = data["obj_tokens"]
                role = data["role"]
                event_id = data["event_id"]
                raw_tokens = obj_tokens
                obj_tokens = self.tokenizer.convert_tokens_to_ids(obj_tokens)
                obj_token_type_ids = data["token_type_ids"]

                if len(obj_tokens) < self.maxlen:
                    obj_attention_mask = [1] * len(obj_tokens) + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_token_type_ids = obj_token_type_ids + [0] * (
                        self.maxlen - len(obj_tokens)
                    )
                    obj_tokens = obj_tokens + [0] * (self.maxlen - len(obj_tokens))
                else:
                    obj_attention_mask = [1] * self.maxlen

                batch_obj_token_ids.append(obj_tokens)
                batch_obj_attention_mask.append(obj_attention_mask)
                batch_obj_token_type_ids.append(obj_token_type_ids)
                batch_raw_tokens.append(raw_tokens)
                batch_argu_roles.append(role)
                batch_text_ids.append(event_id)

        res = {}

        if "ner" == self.task:
            try:
                batch_ner_token_ids = convert_list_to_tensor(batch_ner_token_ids)
                batch_ner_attention_mask = convert_list_to_tensor(
                    batch_ner_attention_mask
                )
                batch_ner_token_type_ids = convert_list_to_tensor(
                    batch_ner_token_type_ids
                )
            except Exception as e:
                logging.info(e)
                logging.info(batch_ner_token_ids)
                for i in batch_ner_token_ids:
                    logging.info(len(i))
                    logging.info(isinstance(i, list))
                exit(0)

            ner_res = {
                "ner_input_ids": batch_ner_token_ids,
                "ner_attention_mask": batch_ner_attention_mask,
                "ner_token_type_ids": batch_ner_token_type_ids,
            }

            res = ner_res

        elif "obj" == self.task:
            batch_obj_token_ids = convert_list_to_tensor(batch_obj_token_ids)
            batch_obj_attention_mask = convert_list_to_tensor(batch_obj_attention_mask)
            batch_obj_token_type_ids = convert_list_to_tensor(batch_obj_token_type_ids)

            sbj_obj_res = {
                "re_obj_input_ids": batch_obj_token_ids,
                "re_obj_attention_mask": batch_obj_attention_mask,
                "re_obj_token_type_ids": batch_obj_token_type_ids,
                "argu_roles": batch_argu_roles,
            }

            res = sbj_obj_res

        # 用于解码
        res["raw_tokens"] = batch_raw_tokens
        # 用于构建最终输出
        res["text_ids"] = batch_text_ids
        # 用于词典增强
        if self.args.use_lexicon:
            batch_augment_Ids = batchify_augment_ids(batch_augment_Ids, self.maxlen)
        res["batch_augment_Ids"] = batch_augment_Ids

        return res


if __name__ == "__main__":
    pass
