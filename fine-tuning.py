# 导入必要的库
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np

# 1. 加载数据集
dataset = load_dataset("AlbertHatsuki/codeforces-question-solution-label")
cpp_dataset = dataset.filter(lambda example: example["language"] == "cpp")

# 2. 确认数据集拆分
assert isinstance(cpp_dataset, DatasetDict), "cpp_dataset 应该是一个 DatasetDict"
assert "train" in cpp_dataset and "validation" in cpp_dataset and "test" in cpp_dataset, \
    "cpp_dataset 应该包含 'train', 'validation' 和 'test' 拆分"
print("数据集拆分确认：", cpp_dataset.keys())

# 3. 获取所有唯一标签
all_tags = set()
for example in cpp_dataset["train"]:
    all_tags.update(example["tags"])
all_tags = sorted(list(all_tags))  # 排序以确保一致性
print(f"唯一标签: {all_tags}")

# 4. 创建标签到 ID 的映射
tag_to_id = {tag: i for i, tag in enumerate(all_tags)}

# 5. 编码标签为多标签格式
def encode_tags(examples):
    labels = [0.0] * len(all_tags)  # 使用 float 类型，与多标签分类兼容
    for tag in examples["tags"]:
        labels[tag_to_id[tag]] = 1.0
    return {"labels": labels}

cpp_dataset = cpp_dataset.map(encode_tags, batched=False)

# 6. 加载 UnixCoder 分词器
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")

# 7. 编码代码
def encode_code(examples):
    encodings = tokenizer(
        examples["code"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"  # 返回 PyTorch 张量
    )
    return {
        "input_ids": encodings["input_ids"].squeeze(0),
        "attention_mask": encodings["attention_mask"].squeeze(0)
    }

cpp_dataset = cpp_dataset.map(encode_code, batched=False)

# 8. 设置数据集格式为 PyTorch
cpp_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 9. 检查设备并加载模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/unixcoder-base",
    num_labels=len(all_tags),
    problem_type="multi_label_classification"
).to(device)

# 10. 定义 Jaccard 相似度计算函数
def compute_jaccard(y_true, y_pred):
    jaccard_scores = []
    for true_labels, pred_labels in zip(y_true, y_pred):
        true_set = set([i for i, val in enumerate(true_labels) if val == 1])
        pred_set = set([i for i, val in enumerate(pred_labels) if val == 1])
        if len(true_set.union(pred_set)) == 0:  # 处理并集为空的情况
            jaccard = 1.0 if len(true_set) == 0 and len(pred_set) == 0 else 0.0
        else:
            jaccard = len(true_set.intersection(pred_set)) / len(true_set.union(pred_set))
        jaccard_scores.append(jaccard)
    return np.mean(jaccard_scores)

# 11. 定义 compute_metrics 函数，将 Jaccard 作为 precision
def compute_metrics(p):
    y_true = p.label_ids
    y_pred = (p.predictions > 0.5).astype(int)  # 使用 0.5 作为阈值
    jaccard = compute_jaccard(y_true, y_pred)
    return {
        "precision": jaccard,  # 将 Jaccard 相似度作为 precision
        "jaccard": jaccard     # 保留 Jaccard 指标以便参考
    }

# 12. 设置训练参数
training_args = TrainingArguments(
    warmup_steps=500,
    output_dir="./cpp_results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=50,
    weight_decay=0.1,
    logging_dir="./cpp_logs",
    logging_steps=50,
    use_mps_device=torch.backends.mps.is_available(),
)

# 13. 创建 Trainer 并传入 compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cpp_dataset["train"],
    eval_dataset=cpp_dataset["validation"],
    compute_metrics=compute_metrics,
)

# 14. 开始训练
trainer.train()

# 15. 模型评估（在测试集上）
predictions = trainer.predict(cpp_dataset["test"])
y_true = [example["labels"].cpu().numpy() for example in cpp_dataset["test"]]
y_pred = (predictions.predictions > 0.5).astype(int)

# 计算 Jaccard 相似度
mean_jaccard = compute_jaccard(y_true, y_pred)
print(f"测试集平均 Jaccard 相似度 (Precision): {mean_jaccard:.4f}")

# 16. 保存模型
trainer.save_model("./cpp_fine_tuned_unixcoder")
