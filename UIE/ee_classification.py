import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

train_file = "data/ee/classification/train.csv" # 数据文件路径，数据需要提前下载
dev_file = "data/ee/classification/dev.csv"
model_name = "/home/ubuntu/PointerNet_Chinese_Information_Extraction/UIE/model_hub/chinese-roberta-wwm-ext" # 所使用模型

# 加载数据集
data_files = {"train": train_file, "test": dev_file}
datasets = load_dataset("csv", data_files=data_files,delimiter="\t")
print('done')
# 数据集处理
tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_function(examples):
  tokenized_examples = tokenizer(examples["review"], max_length=512, truncation=True)
  tokenized_examples["labels"] = examples["label"]
  return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True)

# 构建评估函数
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = predictions.argmax(axis=-1)
  return accuracy_metric.compute(predictions=predictions, references=labels)

# 训练器配置
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
  learning_rate=5e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=10,
  weight_decay=0.01,
  warmup_ratio=0.01,
  output_dir="model_for_seqclassification",
  logging_steps=100,
  save_steps=100,
  eval_steps=100,
  save_total_limit=3,
  evaluation_strategy = "steps",
  save_strategy = "steps",
  load_best_model_at_end=True,
  gradient_accumulation_steps=2,
)

trainer = Trainer(
  model,
  args,
  train_dataset=tokenized_datasets["train"],
  eval_dataset=tokenized_datasets["test"],
  tokenizer=tokenizer,
  compute_metrics=compute_metrics,
  data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# 训练与评估
trainer.train()

trainer.evaluate()