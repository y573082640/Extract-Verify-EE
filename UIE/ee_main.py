import sys

sys.path.append("..")
import tqdm
import logging
import time
from datetime import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import classification_report as cr
from transformers import AdamW, get_linear_schedule_with_warmup
from UIE.model import UIEModel
from UIE.config import EeArgs
from UIE.ee_data_loader import EeDataset, EeCollate
from UIE.ee_data_loader_for_predict import EeCollatePredictor, EeDatasetPredictor
from UIE.utils.decode import (
    ner_decode_batch,
    ner_decode2,
    bj_decode,
    bj_decode_label,
    ner_decode_label,
    obj_decode_batch,
    tri_decode_batch,
    depart_ner_output_batch,
)
from UIE.utils.metrics import (
    calculate_metric,
    word_level_calculate_metric,
    classification_report,
    get_p_r_f,
    get_argu_p_r_f,
)


class EePipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = self.model.module if hasattr(self.model, "module") else self.model

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split(".")
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {
                "params": [
                    p
                    for n, p in bert_param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.lr,
            },
            {
                "params": [
                    p
                    for n, p in bert_param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.args.lr,
            },
            # 其他模块，差分学习率
            {
                "params": [
                    p
                    for n, p in other_param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.other_lr,
            },
            {
                "params": [
                    p
                    for n, p in other_param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": self.args.other_lr,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.args.warmup_proportion * t_total),
            num_training_steps=t_total,
        )

        return optimizer, scheduler

    def bj_eval_forward(self, data_loader, label, id2label, return_report=False):
        """主体或客体"""
        s_logits, e_logits = None, None
        s_labels, e_labels = None, None
        raw_tokens = []
        argu_tuples = []
        masks = None
        loss = 0
        self.model.eval()
        print("...测试中")
        for batch_data in tqdm.tqdm(data_loader):
            for key in batch_data.keys():
                if key not in self.args.ignore_key:
                    batch_data[key] = batch_data[key].to(self.args.device)
            if "sbj" == self.args.task:
                output = self.model(
                    re_sbj_input_ids=batch_data["re_sbj_input_ids"],
                    re_sbj_token_type_ids=batch_data["re_sbj_token_type_ids"],
                    re_sbj_attention_mask=batch_data["re_sbj_attention_mask"],
                    re_sbj_start_labels=None,
                    re_sbj_end_labels=None,
                    augment_Ids=batch_data["batch_augment_Ids"],
                    sim_scores=batch_data["batch_sim_score"],
                )
                start_logits = output["re_output"]["sbj_start_logits"].detach().cpu()
                end_logits = output["re_output"]["sbj_end_logits"].detach().cpu()

                tmp_mask = batch_data["re_sbj_attention_mask"].detach().cpu()
            else:
                output = self.model(
                    re_obj_input_ids=batch_data["re_obj_input_ids"],
                    re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                    re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                    re_obj_start_labels=batch_data["re_obj_start_labels"],
                    re_obj_end_labels=batch_data["re_obj_end_labels"],
                    augment_Ids=batch_data["batch_augment_Ids"],
                    sim_scores=batch_data["batch_sim_score"],
                )
                start_logits = output["re_output"]["obj_start_logits"].detach().cpu()
                end_logits = output["re_output"]["obj_end_logits"].detach().cpu()
                tmp_start_labels = batch_data["re_obj_start_labels"].detach().cpu()
                tmp_end_labels = batch_data["re_obj_end_labels"].detach().cpu()
                tmp_mask = batch_data["re_obj_attention_mask"].detach().cpu()

            if s_logits is None:
                s_logits = start_logits
                e_logits = end_logits
                s_labels = tmp_start_labels
                e_labels = tmp_end_labels
                masks = tmp_mask
            else:
                s_logits = np.append(s_logits, start_logits, axis=0)
                e_logits = np.append(e_logits, end_logits, axis=0)
                s_labels = np.append(s_labels, tmp_start_labels, axis=0)
                e_labels = np.append(e_labels, tmp_end_labels, axis=0)
                masks = np.append(masks, tmp_mask, axis=0)
            
            loss += output["re_output"]["obj_loss"].item()
            raw_tokens += batch_data["raw_tokens"]
            argu_tuples += batch_data["argu_tuples"]

        bj_outputs = {
            "s_logits": s_logits,
            "e_logits": e_logits,
            "s_labels": s_labels,
            "e_labels": e_labels,            
            "masks": masks,
            "argu_tuples": argu_tuples,
            "raw_tokens": raw_tokens,
        }

        metrics = {}

        metrics = self.get_bj_metrics(
            bj_outputs, label=label, id2label=id2label, return_report=return_report
        )
        metrics["bj_metrics"] = metrics
        metrics["loss"] = loss
        return metrics

    def eval_forward(self, data_loader, label, id2label, return_report=False):
        ner_s_logits, ner_e_logits = [], []
        raw_tokens = []
        ner_masks = None
        ner_start_labels = None
        ner_end_labels = None
        loss = 0
        self.model.eval()
        print("...测试中")
        for batch_data in tqdm.tqdm(data_loader):
            for key in batch_data.keys():
                if key not in self.args.ignore_key:
                    batch_data[key] = batch_data[key].to(self.args.device)
            output = self.model(
                ner_input_ids=batch_data["ner_input_ids"],
                ner_token_type_ids=batch_data["ner_token_type_ids"],
                ner_attention_mask=batch_data["ner_attention_mask"],
                ner_start_labels=batch_data["ner_start_labels"],
                ner_end_labels=batch_data["ner_end_labels"],
                augment_Ids=batch_data["batch_augment_Ids"],
            )

            tmp_ner_mask = batch_data["ner_attention_mask"].detach().cpu()
            tmp_ner_start_labels = batch_data["ner_start_labels"].detach().cpu()
            tmp_ner_end_labels = batch_data["ner_end_labels"].detach().cpu()

            if ner_start_labels is None:
                ner_masks = tmp_ner_mask
                ner_start_labels = tmp_ner_start_labels
                ner_end_labels = tmp_ner_end_labels
            else:
                ner_masks = np.append(ner_masks, tmp_ner_mask, axis=0)
                ner_start_labels = np.append(
                    ner_start_labels, tmp_ner_start_labels, axis=0
                )
                ner_end_labels = np.append(ner_end_labels, tmp_ner_end_labels, axis=0)

            loss += output["ner_output"]["ner_loss"].item()
            ner_s_logits, ner_e_logits, raw_tokens = depart_ner_output_batch(
                output, batch_data, ner_s_logits, ner_e_logits, raw_tokens
            )

        ner_outputs = {
            "ner_s_logits": ner_s_logits,  # (1492,label_num,max_len)
            "ner_e_logits": ner_e_logits,
            "ner_masks": ner_masks,
            "ner_start_labels": ner_start_labels,  # (1492, 65, 256)
            "ner_end_labels": ner_end_labels,
            "raw_tokens": raw_tokens,
        }

        metrics = {}

        ner_metrics = self.get_ner_metrics(
            ner_outputs, label, id2label, return_report=return_report
        )
        metrics["ner_metrics"] = ner_metrics
        metrics["loss"] = loss
        return metrics

    def get_bj_metrics(self, bj_outputs, label, id2label, return_report=False):

        
        if 'tri' == self.args.task:
            ### 触发词识别，需要使用严格匹配
            role_metric, total_count = self.get_tri_metrics_helper(
                bj_outputs, id2label, return_report
            )
            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_p_r_f(
                mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        else:
            ### 论元识别，需要使用F1匹配
            role_metric, total_count = self.get_bj_metrics_helper(
                bj_outputs, id2label, return_report
            )
            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = get_argu_p_r_f(
                mirco_metrics[0], mirco_metrics[1], mirco_metrics[2], mirco_metrics[3]
            )            
        res = {
            "precision": mirco_metrics[0],
            "recall": mirco_metrics[1],
            "f1": mirco_metrics[2],
            "report": None,
        }
        if return_report:
            report = classification_report(
                role_metric, label, id2label, total_count, metrics_type=self.args.task
            )
            res["report"] = report
        return res

    def get_tri_metrics_helper(self, outputs, id2label, return_report):
        total_count = [0 for _ in range(len(id2label))]
        role_metric = np.zeros([len(id2label), 3])
        s_logits = outputs["s_logits"]
        e_logits = outputs["e_logits"]
        s_labels = outputs["s_labels"]
        e_labels = outputs["e_labels"]
        masks = outputs["masks"]
        raw_tokens = outputs["raw_tokens"]
        for s_logit, e_logit, s_label, e_label, mask, text in zip(
            s_logits, e_logits, s_labels, e_labels, masks, raw_tokens
        ):
            length = sum(mask)
            # input = (label_num, max_len)
            # output = {label:predict_index}
            true_entities = bj_decode_label(
                s_label, e_label, length, id2label
            )
            pred_entities = bj_decode(
                s_logit, e_logit, length, id2label, bound=0.5
            )

            for idx, _type in enumerate(list(id2label.values())):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                role_metric[idx] += calculate_metric(
                    pred_entities[_type], true_entities[_type], text
                )

        return role_metric, total_count

    def get_bj_metrics_helper(self, outputs, id2label, return_report):
        total_count = [0 for _ in range(len(id2label))]
        role_metric = np.zeros([len(id2label), 4])
        s_logits = outputs["s_logits"]
        e_logits = outputs["e_logits"]
        argu_tuples = outputs["argu_tuples"]
        masks = outputs["masks"]
        raw_tokens = outputs["raw_tokens"]
        for s_logit, e_logit, argu_tuple, mask, text in zip(
            s_logits, e_logits, argu_tuples, masks, raw_tokens
        ):
            length = sum(mask)
            pred_entities = bj_decode(s_logit, e_logit, length, id2label, bound=0.1)
            true_entities = {"答案": argu_tuple}

            # print("========================")
            # print(pred_entities)
            # print(true_entities)
            # print("========================")
            # if return_report:
            #     if str(pred_entities) != str(true_entities):
            #         logging.debug("========================")
            #         logging.debug(''.join(text))
            #         logging.debug('真实')
            #         for key in true_entities.keys():
            #             if len(true_entities[key]) > 0:
            #                 for s_t_tuple in true_entities[key]:
            #                     logging.debug(
            #                         key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

            #         logging.debug('预测')
            #         for key in pred_entities.keys():
            #             if len(pred_entities[key]) > 0:
            #                 for s_t_tuple in pred_entities[key]:
            #                     logging.debug(
            #                         key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))
            #         logging.debug("========================")

            for idx, _type in enumerate(list(id2label.values())):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                role_metric[idx] += word_level_calculate_metric(
                    pred_entities[_type], true_entities[_type], text
                )

        return role_metric, total_count

    def get_ner_metrics(self, ner_outputs, label, id2label, return_report=False):
        role_metric, total_count = self.get_ner_metrics_helper(
            ner_outputs, return_report
        )
        # print(role_metric)
        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        res = {
            "precision": mirco_metrics[0],
            "recall": mirco_metrics[1],
            "f1": mirco_metrics[2],
            "report": None,
        }
        if return_report:
            report = classification_report(
                role_metric, label, id2label, total_count, metrics_type=self.args.task
            )
            res["report"] = report
        return res

    def get_ner_metrics_helper(self, ner_outputs, return_report):
        total_count = [0 for _ in range(len(self.args.ent_id2label))]
        role_metric = np.zeros([len(self.args.ent_id2label), 3])
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        s_logits = ner_outputs["ner_s_logits"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        e_logits = ner_outputs["ner_e_logits"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        s_labels = ner_outputs["ner_start_labels"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        e_labels = ner_outputs["ner_end_labels"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        masks = ner_outputs["ner_masks"]
        raw_tokens = ner_outputs["raw_tokens"]

        with open("log/trigger_badcase.txt", "w") as file_object:
            file_object.write(str(time.time()) + "\n")
        for s_logit, e_logit, s_label, e_label, mask, text in zip(
            s_logits, e_logits, s_labels, e_labels, masks, raw_tokens
        ):
            length = sum(mask)
            # input = (label_num, max_len)
            # output = {label:predict_index}
            true_entities = ner_decode_label(
                s_label, e_label, length, self.args.ent_id2label
            )
            pred_entities = ner_decode2(
                s_logit, e_logit, length, self.args.ent_id2label
            )
            # logging.debug("========================")
            # logging.debug(s_logit)
            # logging.debug(e_logit)
            # logging.debug(pred_entities)
            # logging.debug(true_entities)
            # logging.debug(s_label)
            # logging.debug(e_label)
            # logging.debug("========================")
            # if return_report:
            #     if str(pred_entities) != str(true_entities):
            #         logging.debug("<========================>")
            #         logging.debug(''.join(text))
            #         logging.debug('=============>')
            #         for key in true_entities.keys():
            #             if len(true_entities[key]) > 0:
            #                 for s_t_tuple in true_entities[key]:
            #                     logging.debug(
            #                         key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

            #         logging.debug('<=============')
            #         for key in pred_entities.keys():
            #             if len(pred_entities[key]) > 0:
            #                 for s_t_tuple in pred_entities[key]:
            #                     logging.debug(
            #                         key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

            # 对第i条数据，计算每个事件类型的预测结果
            for idx, _type in enumerate(self.args.entity_label):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                # pred_entities[_type] = [(start_index,end_index+1)...]
                role_metric[idx] += calculate_metric(
                    pred_entities[_type], true_entities[_type], text
                )

        return role_metric, total_count

    def train(self, dev=True):
        train_dataset = EeDataset(file_path=self.args.train_path, args=self.args)
        collate = EeCollate(
            max_len=self.args.max_seq_len,
            task=self.args.task,
            args=self.args,
        )
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            num_workers=4,
            collate_fn=collate.collate_fn,
        )
        dev_loader = None
        dev_callback = None
        ### 保存loss和f1
        train_df = pd.DataFrame(
            columns=["step", "train loss", "eval loss", "precision", "recall"]
        )
        ###
        if dev:
            dev_dataset = EeDataset(
                file_path=self.args.dev_path, args=self.args, test=True
            )
            dev_dataset = dev_dataset
            dev_loader = DataLoader(
                dataset=dev_dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate.collate_fn,
            )

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.0
        early_stop_cnt = 0
        for epoch in range(1, self.args.train_epoch + 1):
            if early_stop_cnt >= 12:
                logging.warn('early stop before epoch{}'.format(epoch))
                break
            for batch_data in tqdm.tqdm((train_loader)):
                train_loss = 0
                self.model.train()
                for key in batch_data.keys():
                    if key not in self.args.ignore_key:
                        batch_data[key] = batch_data[key].to(self.args.device)
                if "ner" == self.args.task:
                    output = self.model(
                        ner_input_ids=batch_data["ner_input_ids"],
                        ner_token_type_ids=batch_data["ner_token_type_ids"],
                        ner_attention_mask=batch_data["ner_attention_mask"],
                        ner_start_labels=batch_data["ner_start_labels"],
                        ner_end_labels=batch_data["ner_end_labels"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                    )
                    loss = output["ner_output"]["ner_loss"]
                elif "obj" == self.args.task or "tri" == self.args.task:
                    output = self.model(
                        re_obj_input_ids=batch_data["re_obj_input_ids"],
                        re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                        re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                        re_obj_start_labels=batch_data["re_obj_start_labels"],
                        re_obj_end_labels=batch_data["re_obj_end_labels"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                        sim_scores=batch_data["batch_sim_score"],
                    )
                    loss = output["re_output"]["obj_loss"]

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                train_loss += loss.item() # 统计每次的loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print(
                    "【train】 Epoch: %d/%d Step: %d/%d loss: %.5f"
                    % (epoch, self.args.train_epoch, global_step, t_total, loss.item())
                )
                if dev and global_step % eval_step == 0:
                    if "ner" == self.args.task:
                        ret = self.eval_forward(
                            dev_loader, self.args.entity_label, self.args.ent_id2label
                        )
                        eval_loss = ret["loss"]
                        metrics = ret["ner_metrics"]

                    elif "obj" == self.args.task or "tri" == self.args.task:
                        label = ["答案"]
                        id2label = {0: "答案"}
                        ret = self.bj_eval_forward(dev_loader, label, id2label)
                        eval_loss = ret["loss"]
                        metrics = ret["bj_metrics"]

                    output_info = "【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f} train_loss={:.4f} eval_loss={:.4f}".format(
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        train_loss,
                        eval_loss,
                    )
                    logging.info(output_info)
                    print(output_info)
                    ### 保存loss和f1
                    train_result = {
                        "step": global_step,
                        "train loss": train_loss,
                        "eval loss": eval_loss,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                    }
                    train_df = train_df.append(train_result, ignore_index=True)
                    train_loss = 0
                    ###
                    if metrics["f1"] > best_f1:
                        early_stop_cnt = 0
                        best_f1 = metrics["f1"]
                        print("【best_f1】：{}".format(best_f1))
                        logging.info("【best_f1】：{}".format(best_f1))
                        self.save_model()
                    else:
                        early_stop_cnt += 1

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_df.to_csv(
            f"/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/log/train_log/train_loss_{current_time}.csv",
            index=False,
        )

    def test(self):
        test_dataset = EeDataset(
            file_path=self.args.test_path, args=self.args, test=True
        )
        collate = EeCollate(
            max_len=self.args.max_seq_len,
            task=self.args.task,
            args=self.args,
        )
        test_dataset = test_dataset
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate.collate_fn,
        )
        self.load_model()
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            if "ner" == self.args.task:
                metrics = self.eval_forward(
                    test_loader,
                    self.args.entity_label,
                    self.args.ent_id2label,
                    return_report=True,
                )
                metrics = metrics["ner_metrics"]

            elif "obj" == self.args.task or "tri" == self.args.task:
                label = ["答案"]
                id2label = {0: "答案"}
                metrics = self.bj_eval_forward(
                    test_loader, label, id2label, return_report=True
                )
                metrics = metrics["bj_metrics"]

            output_info = (
                "【test】 precision={:.4f} recall={:.4f} f1_score={:.4f}".format(
                    metrics["precision"], metrics["recall"], metrics["f1"]
                )
            )
            print(output_info)
            print(metrics["report"])
            logging.info(output_info)
            logging.info(metrics["report"])

    def predict(self, filepath=None, data=None):
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            if "ner" == self.args.task:
                text_ids, raw_tokens = [], []
                ner_s_logits, ner_e_logits = [], []
                logging.info("...构建测试数据集：" + filepath)
                test_dataset = EeDatasetPredictor(
                    file_path=filepath,
                    max_len=self.args.max_seq_len,
                    entity_label=self.args.entity_label,
                    task=self.args.task,
                    args=self.args,
                )

                collate = EeCollatePredictor(
                    max_len=self.args.max_seq_len,
                    task=self.args.task,
                    args=self.args,
                )

                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                    collate_fn=collate.collate_fn,
                )
                logging.info("...获取模型输出")
                for batch_data in tqdm.tqdm(test_loader):
                    for key in batch_data.keys():
                        if key not in self.args.ignore_key:
                            batch_data[key] = batch_data[key].to(self.args.device)
                    output = self.model(
                        ner_input_ids=batch_data["ner_input_ids"],
                        ner_token_type_ids=batch_data["ner_token_type_ids"],
                        ner_attention_mask=batch_data["ner_attention_mask"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                    )

                    ner_s_logits, ner_e_logits, raw_tokens = depart_ner_output_batch(
                        output, batch_data, ner_s_logits, ner_e_logits, raw_tokens
                    )
                    text_ids += batch_data["text_ids"]

                logging.info("...解码事件列表...")
                pred_entities = ner_decode_batch(
                    ner_s_logits, ner_e_logits, raw_tokens, self.args.ent_id2label
                )

                merged_ret = []
                logging.info("...映射事件和文本ID...")
                for event_dict, t_id, t_tokens in zip(
                    pred_entities, text_ids, raw_tokens
                ):
                    merged_ret.append(
                        {
                            "event_dict": event_dict,
                            "text_id": t_id,
                            "text": "".join(t_tokens),
                        }
                    )
                logging.info("...事件检测任务完成")
                return merged_ret

            # tokens = ['[CLS]'] + tokens + ['[SEP]'] + tokens + + ['[SEP]']
            elif "tri" == self.args.task:
                s_logits, e_logits = None, None
                raw_tokens, text_ids, event_types,text_bias = [],[], [], []

                logging.info("...构建测试数据集：" + filepath)
                test_dataset = EeDatasetPredictor(
                    file_path=filepath,
                    args=self.args,
                )

                collate = EeCollatePredictor(
                    max_len=self.args.max_seq_len,
                    task=self.args.task,
                    args=self.args,
                )

                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                    collate_fn=collate.collate_fn,
                )
                for batch_data in tqdm.tqdm(test_loader):
                    for key in batch_data.keys():
                        if key not in self.args.ignore_key:
                            batch_data[key] = batch_data[key].to(self.args.device)
                    output = self.model(
                        re_obj_input_ids=batch_data["re_obj_input_ids"],
                        re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                        re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                    )
                    start_logits = (
                        output["re_output"]["obj_start_logits"].detach().cpu()
                    )
                    end_logits = output["re_output"]["obj_end_logits"].detach().cpu()
                    tmp_mask = batch_data["re_obj_attention_mask"].detach().cpu()

                    if start_logits.dim() < 2:
                        start_logits = start_logits.unsqueeze(0)
                        end_logits = end_logits.unsqueeze(0)

                    if s_logits is None:
                        s_logits = start_logits
                        e_logits = end_logits
                        masks = tmp_mask
                    else:
                        s_logits = np.append(s_logits, start_logits, axis=0)
                        e_logits = np.append(e_logits, end_logits, axis=0)
                        masks = np.append(masks, tmp_mask, axis=0)

                    raw_tokens += batch_data["raw_tokens"]
                    text_ids += batch_data["text_ids"]
                    event_types += batch_data["event_types"]
                    text_bias += batch_data["text_bias"]
                logging.info("...进行触发词解码")
                ret = tri_decode_batch(
                    s_logits=s_logits,
                    e_logits=e_logits,
                    masks=masks,
                    id2label={0: "答案"},
                    raw_tokens=raw_tokens,
                    text_ids=text_ids,
                    event_types=event_types,
                    text_bias=text_bias
                )
                logging.info("...触发词预测完毕")

                return ret
            
            elif "obj" == self.args.task:
                """
                {
                event_type,
                textb,
                trigger,
                trigger_start_index,
                event_id,
                text_id
                }
                """
                s_logits, e_logits = None, None
                raw_tokens, event_ids, roles = [], [], []
                logging.info("...构建测试数据集：" + str(len(data)))
                test_dataset = EeDatasetPredictor(
                    data=data,
                    max_len=self.args.max_seq_len,
                    entity_label=self.args.entity_label,
                    task=self.args.task,
                    args=self.args,
                )

                collate = EeCollatePredictor(
                    max_len=self.args.max_seq_len,
                    task=self.args.task,
                    args=self.args,
                )

                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=self.args.eval_batch_size,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=collate.collate_fn,
                )

                for batch_data in tqdm.tqdm(test_loader):
                    for key in batch_data.keys():
                        if key not in self.args.ignore_key:
                            batch_data[key] = batch_data[key].to(self.args.device)
                    output = self.model(
                        re_obj_input_ids=batch_data["re_obj_input_ids"],
                        re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                        re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                    )
                    start_logits = (
                        output["re_output"]["obj_start_logits"].detach().cpu()
                    )
                    end_logits = output["re_output"]["obj_end_logits"].detach().cpu()
                    tmp_mask = batch_data["re_obj_attention_mask"].detach().cpu()

                    if start_logits.dim() < 2:
                        start_logits = start_logits.unsqueeze(0)
                        end_logits = end_logits.unsqueeze(0)

                    if s_logits is None:
                        s_logits = start_logits
                        e_logits = end_logits
                        masks = tmp_mask
                    else:
                        s_logits = np.append(s_logits, start_logits, axis=0)
                        e_logits = np.append(e_logits, end_logits, axis=0)
                        masks = np.append(masks, tmp_mask, axis=0)

                    raw_tokens += batch_data["raw_tokens"]
                    event_ids += batch_data["text_ids"]
                    roles += batch_data["argu_roles"]

                logging.info("...进行论元解码")
                # logging.info(raw_tokens)
                # logging.info(event_ids)
                # logging.info(roles)
                ret = obj_decode_batch(
                    s_logits=s_logits,
                    e_logits=e_logits,
                    masks=masks,
                    id2label={0: "答案"},
                    raw_tokens=raw_tokens,
                    event_ids=event_ids,
                    roles=roles,
                )
                logging.info("...论元预测完毕")

                return ret


if __name__ == "__main__":
    obj_weight = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/obj_duee_roberta_None_【论元抽取】多论元合并 No DEMO sample4.pt"
    ner_weright = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/ner_duee_roberta_no_lexicon_len256_bs32.pt"
    check_base = (
        "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/"
    )
    ner_path = check_base + "tri_duee_roberta_None_trigger_extraction.pt"

    path = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/checkpoints/ee/tri_duee_roberta_None_trigger_extraction.pt"
    # for t_path in ["0_1.json", "2_3.json", "4_5.json"]:

    #     args.test_path = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/experiment/train_only/split_by_evt_num/{}".format(
    #         t_path
    # )
    #     model = UIEModel(args)
    #     ee_pipeline = EePipeline(model, args)
    #     ee_pipeline.test()
    #     torch.cuda.empty_cache()

    args = EeArgs(
        "obj",
        log=True,
        aug_mode=None,
        model='roberta',
        add_trigger=False,
        # output_name="【论元抽取】多论元合并 No DEMO sample2"
        weight_path=obj_weight
    )
    model = UIEModel(args)
    ee_pipeline = EePipeline(model, args)
    ee_pipeline.test()
    torch.cuda.empty_cache()