import sys
sys.path.append('..')
import tqdm
import logging
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report as cr
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from UIE.model import UIEModel
from UIE.config import EeArgs
from UIE.ee_data_loader import EeDataset, EeCollate, EeCollatePredictor, EeDatasetPredictor
from UIE.utils.decode import ner_decode_batch, ner_decode2, bj_decode, obj_decode_batch, sigmoid, depart_ner_output_batch
from UIE.utils.metrics import calculate_metric, word_level_calculate_metric, classification_report, get_p_r_f, get_argu_p_r_f
from UIE.utils.lexicon_functions import generate_instance_with_gaz, batchify_augment_ids
from UIE.utils.question_maker import get_question_for_argument, creat_argu_labels, creat_argu_token, creat_demo



class EePipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(
            self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    def bj_eval_forward(self,
                        data_loader,
                        label,
                        id2label,
                        return_report=False):
        """主体或客体"""
        s_logits, e_logits = None, None
        raw_tokens = []
        masks = None
        start_labels = None
        end_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                if key not in self.args.ignore_key:
                    batch_data[key] = batch_data[key].to(self.args.device)
            if "sbj" in self.args.tasks:
                output = self.model(re_sbj_input_ids=batch_data["re_sbj_input_ids"],
                                    re_sbj_token_type_ids=batch_data["re_sbj_token_type_ids"],
                                    re_sbj_attention_mask=batch_data["re_sbj_attention_mask"],
                                    re_sbj_start_labels=batch_data["re_sbj_start_labels"],
                                    re_sbj_end_labels=batch_data["re_sbj_end_labels"],
                                    augment_Ids=batch_data["batch_augment_Ids"],
                                    sim_scores=batch_data["batch_sim_score"]
                                    )
                start_logits = output["re_output"]["sbj_start_logits"].detach(
                ).cpu()
                end_logits = output["re_output"]["sbj_end_logits"].detach(
                ).cpu()

                tmp_mask = batch_data["re_sbj_attention_mask"].detach().cpu()
                tmp_start_labels = batch_data["re_sbj_start_labels"].detach(
                ).cpu()
                tmp_end_labels = batch_data["re_sbj_end_labels"].detach().cpu()
            else:
                output = self.model(re_obj_input_ids=batch_data["re_obj_input_ids"],
                                    re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                                    re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                                    re_obj_start_labels=batch_data["re_obj_start_labels"],
                                    re_obj_end_labels=batch_data["re_obj_end_labels"],
                                    augment_Ids=batch_data["batch_augment_Ids"],
                                    sim_scores=batch_data["batch_sim_score"]
                                    )
                start_logits = output["re_output"]["obj_start_logits"].detach(
                ).cpu()
                end_logits = output["re_output"]["obj_end_logits"].detach(
                ).cpu()

                tmp_mask = batch_data["re_obj_attention_mask"].detach().cpu()
                tmp_start_labels = batch_data["re_obj_start_labels"].detach(
                ).cpu()
                tmp_end_labels = batch_data["re_obj_end_labels"].detach().cpu()

            if start_labels is None:
                s_logits = start_logits
                e_logits = end_logits
                masks = tmp_mask
                start_labels = tmp_start_labels
                end_labels = tmp_end_labels
            else:
                s_logits = np.append(s_logits, start_logits, axis=0)
                e_logits = np.append(e_logits, end_logits, axis=0)
                masks = np.append(masks, tmp_mask, axis=0)
                start_labels = np.append(
                    start_labels, tmp_start_labels, axis=0)
                end_labels = np.append(end_labels, tmp_end_labels, axis=0)

            raw_tokens += batch_data['raw_tokens']

        bj_outputs = {
            "s_logits": s_logits,
            "e_logits": e_logits,
            "masks": masks,
            "start_labels": start_labels,
            "end_labels": end_labels,
            'raw_tokens': raw_tokens
        }

        metrics = {}

        metrics = self.get_bj_metrics(bj_outputs,
                                      label=label,
                                      id2label=id2label,
                                      return_report=return_report)
        metrics["bj_metrics"] = metrics

        return metrics

    def eval_forward(self,
                     data_loader,
                     label,
                     id2label,
                     return_report=False):
        ner_s_logits, ner_e_logits = [], []
        raw_tokens = []
        ner_masks = None
        ner_start_labels = None
        ner_end_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                if key not in self.args.ignore_key:
                    batch_data[key] = batch_data[key].to(self.args.device)
            output = self.model(ner_input_ids=batch_data["ner_input_ids"],
                                ner_token_type_ids=batch_data["ner_token_type_ids"],
                                ner_attention_mask=batch_data["ner_attention_mask"],
                                ner_start_labels=batch_data["ner_start_labels"],
                                ner_end_labels=batch_data["ner_end_labels"],
                                augment_Ids=batch_data["batch_augment_Ids"]
                                )

            tmp_ner_mask = batch_data["ner_attention_mask"].detach().cpu()
            tmp_ner_start_labels = batch_data["ner_start_labels"].detach(
            ).cpu()
            tmp_ner_end_labels = batch_data["ner_end_labels"].detach().cpu()

            if ner_start_labels is None:
                ner_masks = tmp_ner_mask
                ner_start_labels = tmp_ner_start_labels
                ner_end_labels = tmp_ner_end_labels
            else:
                ner_masks = np.append(ner_masks, tmp_ner_mask, axis=0)
                ner_start_labels = np.append(
                    ner_start_labels, tmp_ner_start_labels, axis=0)
                ner_end_labels = np.append(
                    ner_end_labels, tmp_ner_end_labels, axis=0)

            ner_s_logits, ner_e_logits, raw_tokens = depart_ner_output_batch(
                output, batch_data, ner_s_logits, ner_e_logits, raw_tokens)

        ner_outputs = {
            "ner_s_logits": ner_s_logits,  # (1492,label_num,max_len)
            "ner_e_logits": ner_e_logits,
            "ner_masks": ner_masks,
            "ner_start_labels": ner_start_labels,  # (1492, 65, 256)
            "ner_end_labels": ner_end_labels,
            'raw_tokens': raw_tokens
        }

        metrics = {}

        ner_metrics = self.get_ner_metrics(ner_outputs,
                                           label,
                                           id2label,
                                           return_report=return_report)

        metrics["ner_metrics"] = ner_metrics

        return metrics

    def get_bj_metrics(self,
                       bj_outputs,
                       label,
                       id2label,
                       return_report=False):
        role_metric, total_count = self.get_bj_metrics_helper(
            bj_outputs, id2label, return_report)
        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = get_argu_p_r_f(
            mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        res = {
            "precision": mirco_metrics[0],
            "recall": mirco_metrics[1],
            "f1": mirco_metrics[2],
            "report": None,
        }
        if return_report:
            report = classification_report(
                role_metric, label, id2label, total_count)
            res["report"] = report
        return res

    def get_bj_metrics_helper(self, outputs, id2label, return_report):
        total_count = [0 for _ in range(len(id2label))]
        role_metric = np.zeros([len(id2label), 3])
        s_logits = outputs["s_logits"]
        e_logits = outputs["e_logits"]
        s_label = outputs["start_labels"]
        e_label = outputs["end_labels"]
        masks = outputs["masks"]
        raw_tokens = outputs["raw_tokens"]
        for s_logit, e_logit, s_label, e_label, mask, text in zip(s_logits, e_logits, s_label, e_label, masks, raw_tokens):
            length = sum(mask)
            pred_entities = bj_decode(s_logit, e_logit, length, id2label)
            true_entities = bj_decode(s_label, e_label, length, id2label)
            # print("========================")
            # print(pred_entities)
            # print(true_entities)
            # print("========================")
            if return_report:
                if str(pred_entities) != str(true_entities):
                    logging.debug("========================")
                    logging.debug(''.join(text))
                    logging.debug('真实')
                    for key in true_entities.keys():
                        if len(true_entities[key]) > 0:
                            for s_t_tuple in true_entities[key]:
                                logging.debug(
                                    key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

                    logging.debug('预测')
                    for key in pred_entities.keys():
                        if len(pred_entities[key]) > 0:
                            for s_t_tuple in pred_entities[key]:
                                logging.debug(
                                    key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))
                    logging.debug("========================")

            for idx, _type in enumerate(list(id2label.values())):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                role_metric[idx] += word_level_calculate_metric(
                    pred_entities[_type], true_entities[_type], text)

        return role_metric, total_count

    def get_ner_metrics(self,
                        ner_outputs,
                        label,
                        id2label,
                        return_report=False):
        role_metric, total_count = self.get_ner_metrics_helper(
            ner_outputs, return_report)
        # print(role_metric)
        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = get_p_r_f(
            mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
        res = {
            "precision": mirco_metrics[0],
            "recall": mirco_metrics[1],
            "f1": mirco_metrics[2],
            "report": None,
        }
        if return_report:
            report = classification_report(
                role_metric, label, id2label, total_count)
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
        s_label = ner_outputs["ner_start_labels"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        e_label = ner_outputs["ner_end_labels"]
        # (1492, 65, 256) = (num_cases,num_labels,max_len)
        masks = ner_outputs["ner_masks"]
        raw_tokens = ner_outputs["raw_tokens"]
        # with open('log/trigger_badcase.txt', 'w') as file_object:
        #     file_object.write(str(time.time()) + "\n")
        #     for i in range(19):
        #         file_object.write(str(s_logits[i][53]) + "\n") # 组织关系-裁员
        #         file_object.write(str(e_logits[i][53]) + "\n")
        #         file_object.write(str("*"*50) + "\n")
        #         file_object.write(str(s_label[i][53]) + "\n")
        #         file_object.write(str(e_label[i][53]) + "\n")
        #         file_object.write(str("="*100) + "\n")
        #     for i in range(19,28):
        #         file_object.write(str(s_logits[i][50]) + "\n") # 组织关系-裁员
        #         file_object.write(str(e_logits[i][50]) + "\n")
        #         file_object.write(str("*"*50) + "\n")
        #         file_object.write(str(s_label[i][50]) + "\n")
        #         file_object.write(str(e_label[i][50]) + "\n")
        #         file_object.write(str("="*100) + "\n")
        #     exit(0)
        with open('log/trigger_badcase.txt', 'w') as file_object:
            file_object.write(str(time.time()) + "\n")
        for s_logit, e_logit, s_label, e_label, mask, text in zip(s_logits, e_logits, s_label, e_label, masks, raw_tokens):
            length = sum(mask)
            # input = (label_num, max_len)
            pred_entities = ner_decode2(
                s_logit, e_logit, length, self.args.ent_id2label)
            # output = {label:predict_index}
            true_entities = ner_decode2(
                s_label, e_label, length, self.args.ent_id2label)
            # print("========================")
            # print(pred_entities)
            # print(true_entities)
            # print("========================")
            if return_report:
                if str(pred_entities) != str(true_entities):
                    logging.debug("<========================>")
                    logging.debug(''.join(text))
                    logging.debug('=============>')
                    for key in true_entities.keys():
                        if len(true_entities[key]) > 0:
                            for s_t_tuple in true_entities[key]:
                                logging.debug(
                                    key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

                    logging.debug('<=============')
                    for key in pred_entities.keys():
                        if len(pred_entities[key]) > 0:
                            for s_t_tuple in pred_entities[key]:
                                logging.debug(
                                    key + ":" + str(text[s_t_tuple[0]:s_t_tuple[1]]))

            # 对第i条数据，计算每个事件类型的预测结果
            for idx, _type in enumerate(self.args.entity_label):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                total_count[idx] += len(true_entities[_type])
                # pred_entities[_type] = [(start_index,end_index+1)...]
                role_metric[idx] += calculate_metric(
                    pred_entities[_type], true_entities[_type], text)

        return role_metric, total_count

    def train(self, dev=True):
        train_dataset = EeDataset(file_path=self.args.train_path,
                                  tokenizer=self.args.tokenizer,
                                  max_len=self.args.max_seq_len,
                                  entity_label=self.args.entity_label,
                                  tasks=self.args.tasks,
                                  args=self.args)
        collate = EeCollate(max_len=self.args.max_seq_len,
                            tokenizer=self.args.tokenizer, tasks=self.args.tasks, args=self.args)
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  sampler=train_sampler,
                                  collate_fn=collate.collate_fn)
        dev_loader = None
        dev_callback = None
        if dev:
            dev_dataset = EeDataset(file_path=self.args.dev_path,
                                    tokenizer=self.args.tokenizer,
                                    max_len=self.args.max_seq_len,
                                    entity_label=self.args.entity_label,
                                    tasks=self.args.tasks,
                                    args=self.args)
            dev_dataset = dev_dataset
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.args.eval_batch_size,
                                    shuffle=False,
                                    collate_fn=collate.collate_fn)

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    if key not in self.args.ignore_key:
                        batch_data[key] = batch_data[key].to(self.args.device)
                if "ner" in self.args.tasks:
                    output = self.model(ner_input_ids=batch_data["ner_input_ids"],
                                        ner_token_type_ids=batch_data["ner_token_type_ids"],
                                        ner_attention_mask=batch_data["ner_attention_mask"],
                                        ner_start_labels=batch_data["ner_start_labels"],
                                        ner_end_labels=batch_data["ner_end_labels"],
                                        augment_Ids=batch_data["batch_augment_Ids"]
                                        )
                    loss = output["ner_output"]["ner_loss"]
                elif "obj" in self.args.tasks:
                    output = self.model(
                        re_obj_input_ids=batch_data["re_obj_input_ids"],
                        re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                        re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                        re_obj_start_labels=batch_data["re_obj_start_labels"],
                        re_obj_end_labels=batch_data["re_obj_end_labels"],
                        augment_Ids=batch_data["batch_augment_Ids"],
                        sim_scores=batch_data['batch_sim_score']
                    )
                    loss = output["re_output"]["obj_loss"]

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】 Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev and global_step % eval_step == 0:
                    if "ner" in self.args.tasks:
                        metrics = self.eval_forward(
                            dev_loader, self.args.entity_label, self.args.ent_id2label)
                        metrics = metrics["ner_metrics"]

                    elif "obj" in self.args.tasks:
                        label = ["答案"]
                        id2label = {0: "答案"}
                        metrics = self.bj_eval_forward(
                            dev_loader, label, id2label)
                        metrics = metrics["bj_metrics"]

                    output_info = '【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(metrics["precision"],
                                                                                                 metrics["recall"],
                                                                                                 metrics["f1"])
                    logging.info(output_info)
                    print(output_info)
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        print("【best_f1】：{}".format(best_f1))
                        self.save_model()

    def test(self):
        test_dataset = EeDataset(file_path=self.args.test_path,
                                 tokenizer=self.args.tokenizer,
                                 max_len=self.args.max_seq_len,
                                 entity_label=self.args.entity_label,
                                 tasks=self.args.tasks,
                                 args=self.args)
        collate = EeCollate(max_len=self.args.max_seq_len,
                            tokenizer=self.args.tokenizer, tasks=self.args.tasks, args=self.args)
        test_dataset = test_dataset
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate.collate_fn)
        self.load_model()
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            if "ner" in self.args.tasks:
                metrics = self.eval_forward(
                    test_loader, self.args.entity_label, self.args.ent_id2label, return_report=True)
                metrics = metrics["ner_metrics"]

            elif "obj" in self.args.tasks:
                label = ["答案"]
                id2label = {0: "答案"}
                metrics = self.bj_eval_forward(
                    test_loader, label, id2label, return_report=True)
                metrics = metrics["bj_metrics"]

            output_info = '【test】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(metrics["precision"],
                                                                                         metrics["recall"],
                                                                                         metrics["f1"])
            logging.info(output_info)
            logging.info(metrics["report"])
            print(output_info)
            print(metrics["report"])

    def predict(self, filepath=None, data=None):

        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            if "ner" in self.args.tasks:
                text_ids, raw_tokens = [], []
                ner_s_logits, ner_e_logits = [], []
                logging.info("构建测试数据集：" + filepath)
                test_dataset = EeDatasetPredictor(file_path=filepath,
                                                  tokenizer=self.args.tokenizer,
                                                  max_len=self.args.max_seq_len,
                                                  entity_label=self.args.entity_label,
                                                  tasks=self.args.tasks,
                                                  args=self.args)

                collate = EeCollatePredictor(max_len=self.args.max_seq_len,
                                             tokenizer=self.args.tokenizer,
                                             tasks=self.args.tasks,
                                             args=self.args)

                test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=self.args.eval_batch_size,
                                         shuffle=False,
                                         collate_fn=collate.collate_fn)
                logging.info("获取模型输出...")
                for batch_data in tqdm.tqdm(test_loader):
                    for key in batch_data.keys():
                        if key not in self.args.ignore_key:
                            batch_data[key] = batch_data[key].to(
                                self.args.device)
                    output = self.model(ner_input_ids=batch_data["ner_input_ids"],
                                        ner_token_type_ids=batch_data["ner_token_type_ids"],
                                        ner_attention_mask=batch_data["ner_attention_mask"],
                                        augment_Ids=batch_data["batch_augment_Ids"]
                                        )

                    ner_s_logits, ner_e_logits, raw_tokens = depart_ner_output_batch(
                        output, batch_data, ner_s_logits, ner_e_logits, raw_tokens)
                    text_ids += batch_data['text_ids']

                logging.info("解码事件列表...")
                pred_entities = ner_decode_batch(
                    ner_s_logits, ner_e_logits, raw_tokens, self.args.ent_id2label)

                merged_ret = []
                logging.info("映射事件和文本ID...")
                for event_dict, t_id, t_tokens in zip(pred_entities, text_ids, raw_tokens):
                    merged_ret.append({
                        'event_dict': event_dict,
                        'text_id': t_id,
                        'text': ''.join(t_tokens)
                    })
                logging.info("事件检测任务完成")
                return merged_ret

            # tokens = ['[CLS]'] + tokens + ['[SEP]'] + tokens + + ['[SEP]']
            elif "obj" in self.args.tasks:
                '''
                {
                event_type,
                textb,
                trigger,
                trigger_start_index,
                event_id,
                text_id
                }                
                '''
                s_logits, e_logits = None, None
                raw_tokens, event_ids, roles = [], [], []
                logging.info("构建测试数据集：" + str(len(data)))
                test_dataset = EeDatasetPredictor(data=data,                                        tokenizer=self.args.tokenizer,
                                                  max_len=self.args.max_seq_len,
                                                  entity_label=self.args.entity_label,
                                                  tasks=self.args.tasks,
                                                  args=self.args)

                collate = EeCollatePredictor(max_len=self.args.max_seq_len,
                                             tokenizer=self.args.tokenizer,
                                             tasks=self.args.tasks,
                                             args=self.args)

                test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=self.args.eval_batch_size,
                                         shuffle=False,
                                         collate_fn=collate.collate_fn)


                for batch_data in tqdm.tqdm(test_loader):
                    for key in batch_data.keys():
                        if key not in self.args.ignore_key:
                            batch_data[key] = batch_data[key].to(
                                self.args.device)
                    output = self.model(re_obj_input_ids=batch_data["re_obj_input_ids"],
                                        re_obj_token_type_ids=batch_data["re_obj_token_type_ids"],
                                        re_obj_attention_mask=batch_data["re_obj_attention_mask"],
                                        augment_Ids=batch_data["batch_augment_Ids"],
                                        )
                    start_logits = output["re_output"]["obj_start_logits"].detach(
                    ).cpu()
                    end_logits = output["re_output"]["obj_end_logits"].detach(
                    ).cpu()
                    tmp_mask = batch_data["re_obj_attention_mask"].detach(
                    ).cpu()

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

                    raw_tokens += batch_data['raw_tokens']
                    event_ids += batch_data['text_ids']
                    roles += batch_data['argu_roles']

                logging.info("进行论元解码")
                ret = obj_decode_batch(
                    s_logits=s_logits,
                    e_logits=e_logits,
                    masks=masks,
                    id2label={0: "答案"},
                    raw_tokens=raw_tokens,
                    event_ids=event_ids,
                    roles=roles
                )
                logging.info("论元预测完毕")

                return ret

if __name__ == '__main__':
    args = EeArgs('ner',use_lexicon=True)
    model = UIEModel(args)
    ee_pipeline = EePipeline(model, args)
    ee_pipeline.train()
    ee_pipeline.test()

    args = EeArgs('obj',log=False)
    model = UIEModel(args)
    ee_pipeline = EePipeline(model, args)
    ee_pipeline.train()
    ee_pipeline.test()



