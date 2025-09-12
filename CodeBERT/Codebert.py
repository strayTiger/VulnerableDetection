# -*- coding: utf-8 -*-

# ---------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os, json, time, datetime, pickle, random, argparse, shutil, statistics
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, WeightedRandomSampler,
                              TensorDataset)

from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report)
from numpy import trapz

# Hugging Face
from transformers import (RobertaTokenizer, RobertaModel,
                          AdamW, get_linear_schedule_with_warmup)

# 自己的数据预处理函数（从第一段脚本里拷贝）
from DataLoader import SplitCharacters, ListToCSV

# ---------------------------------------------------------
# 科普 原始 Transformer 使用基于正弦和余弦函数的位置编码（Positional Encoding）
# 关键特性：这种编码可以动态生成，无需预定义最大长度（如 512）。理论上，它可以处理无限长的序列。
#BERT 的 512 长度限制是开发者权衡效率、资源和任务需求后的结果，而非技术上限。BERT使用的是绝对位置编码
# 2. 全局配置 这些值提到文件最前面统一管理，方便后期微调；完全不会改变算法逻辑。
# ---------------------------------------------------------
#SEED          = 42    # 1. 可复现性——同一机器或不同机器，多次跑实验能得出几乎一致的结果；2. 对比公平——改其他超参时，避免因为“刚好抽到好/坏初值”带来噪声
MAX_LEN       = 512   # CodeBERT base 最大长度
BATCH_SIZE    = 12
NUM_EPOCHS    = 2
LR            = 2e-5  # 学习率
WEIGHT_DECAY  = 0.01  # L2 正则项系数，AdamW 会在每步更新时额外乘以 (1 − lr*weight_decay)，抑制权重过大,减少过拟合，特别是参数量大的 Transformer；也能在一定程度上稳定训练
GRAD_NORM     = 1.0   # 梯度裁剪上限：反向传播完毕后，把所有参数梯度的 L2 范数缩放到不超过这一数值.防止出现 梯度爆炸 —— 特别是大模型、大学习率或 LoRA 等微调时，某一小批次可能让梯度异常大，导致权重发散
MODEL_DIR     = "./codebert_model"       # 你的 CodeBERT 权重目录
RESULT_DIR    = "./result"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#random.seed(SEED)
#np.random.seed(SEED)
#torch.manual_seed(SEED)
#torch.cuda.manual_seed_all(SEED)

Path(RESULT_DIR).mkdir(exist_ok=True)

# ---------------------------------------------------------
# 3. 读取 JSONL → id / label / code
# ---------------------------------------------------------
def generate_id_label_code(json_path):
    ids, labels, codes = [], [], []
    with open(json_path, encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            ids.append(js["func_name"])
            labels.append(js["target"])
            elements = [SplitCharacters(tok) for tok in js["func"].split()]
            codes.append(" ".join(elements))
    return ids, labels, codes

data_dir = "./"
train_ids, train_y, train_code = generate_id_label_code(Path(data_dir, "train.jsonl"))
valid_ids, valid_y, valid_code = generate_id_label_code(Path(data_dir, "valid.jsonl"))
test_ids,  test_y,  test_code  = generate_id_label_code(Path(data_dir, "test.jsonl"))

print(f"Train {len(train_y)}  | positive {np.sum(train_y)}")
print(f"Valid {len(valid_y)}  | positive {np.sum(valid_y)}")
print(f"Test  {len(test_y)}   | positive {np.sum(test_y)}")

# ---------------------------------------------------------
# 4. Tokenizer & Padding
# ---------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)

def encode_codes(code_list, max_len=MAX_LEN):
    enc = tokenizer(code_list,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="np")
    return enc["input_ids"], enc["attention_mask"]

train_ids_np, train_masks_np = encode_codes(train_code)
valid_ids_np, valid_masks_np = encode_codes(valid_code)
test_ids_np,  test_masks_np  = encode_codes(test_code)

train_inputs  = torch.tensor(train_ids_np,   dtype=torch.long)
train_masks   = torch.tensor(train_masks_np, dtype=torch.float)
train_labels  = torch.tensor(train_y,        dtype=torch.long)

valid_inputs  = torch.tensor(valid_ids_np,   dtype=torch.long)
valid_masks   = torch.tensor(valid_masks_np, dtype=torch.float)
valid_labels  = torch.tensor(valid_y,        dtype=torch.long)

test_inputs   = torch.tensor(test_ids_np,    dtype=torch.long)
test_masks    = torch.tensor(test_masks_np,  dtype=torch.float)
test_labels   = torch.tensor(test_y,         dtype=torch.long)

# ---------------------------------------------------------
# 5. 处理类别不平衡 → WeightedRandomSampler
# ---------------------------------------------------------
cls_sample_count = np.unique(train_y, return_counts=True)[1]
cls_weights      = 1. / cls_sample_count
samples_weight   = torch.from_numpy(cls_weights[train_y]).double()
weighted_sampler = WeightedRandomSampler(samples_weight,
                                         len(samples_weight),
                                         replacement=True)

train_dataset  = TensorDataset(train_inputs, train_masks, train_labels)
valid_dataset  = TensorDataset(valid_inputs, valid_masks, valid_labels)
test_dataset   = TensorDataset(test_inputs,  test_masks,  test_labels)

train_loader   = DataLoader(train_dataset, sampler=weighted_sampler,
                            batch_size=BATCH_SIZE)
valid_loader   = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                            batch_size=BATCH_SIZE)
test_loader    = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),
                            batch_size=BATCH_SIZE)

# ---------------------------------------------------------
# 6. CodeBERT 分类模型
# ---------------------------------------------------------
class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(MODEL_DIR)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]     # <s> token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        logits = logits[:, 1] - logits[:, 0]            # 二分类差值逻辑

        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            return loss
        return logits

model = CodeBERTClassifier().to(DEVICE)

# ---------------------------------------------------------
# 7. 优化器 + 调度器
# ---------------------------------------------------------
no_decay = ["bias", "LayerNorm.weight"]
opt_group = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0}
]
optimizer = AdamW(opt_group, lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(0.1*total_steps),
                                            num_training_steps=total_steps)

def topk_metrics(pred_probs, labels, ks=(50,100,150,200,250,300,350,400)):
    scores = np.asarray(pred_probs).flatten()
    labels = np.asarray(labels).flatten()
    total_pos = labels.sum()
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    res = {}
    for k in ks:
        if k > len(sorted_labels): continue
        tp_k = sorted_labels[:k].sum()
        res[k] = (tp_k/k, tp_k/total_pos if total_pos else 0.)
    return res

# ---------------------------------------------------------
# 8. 训练函数
# ---------------------------------------------------------
def format_time(sec): return str(datetime.timedelta(seconds=int(round(sec))))

def train(model):
    best_val_loss = float("inf")
    stats = []
    patience=5
    min_delta=1e-4
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} ========")
        t0, tot_loss, n_samples = time.time(), 0.0, 0

        model.train()

        # 假如我们脚本里把 BATCH_SIZE 设成 4，表示一次前向/反向会同时送进 4 个样本。打印的显存就是处理这 4 条样本所需的峰值
        torch.cuda.reset_peak_memory_stats() # 把峰值计数器清零，便于下一次测量.每 40 个 batch 的当前最大 GPU 占用

        for step, batch in enumerate(train_loader):
            if step % 40 == 0 and step:
                elapsed = format_time(time.time() - t0)
                peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"训练峰值显存: {peak_mem:.2f} GB  Batch {step}/{len(train_loader)} | Elapsed {elapsed}")
                torch.cuda.reset_peak_memory_stats()

            b_ids, b_masks, b_y = (t.to(DEVICE) for t in batch)
            optimizer.zero_grad()
            loss = model(b_ids, attention_mask=b_masks, labels=b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step(); scheduler.step()

            tot_loss += loss.item(); n_samples += b_y.size(0)

        avg_train_loss = tot_loss / n_samples
        print(f"  Train loss: {avg_train_loss:.4f} | epoch took {format_time(time.time()-t0)}")

        # ---------- Validation ----------
        model.eval();
        val_loss, n_eval = 0.0, 0
        t0=time.time()
        
        with torch.no_grad():
            for b_ids, b_masks, b_y in valid_loader:
                b_ids, b_masks, b_y = b_ids.to(DEVICE), b_masks.to(DEVICE), b_y.to(DEVICE)
                loss = model(b_ids, attention_mask=b_masks, labels=b_y)
                val_loss += loss.item(); n_eval += b_y.size(0)

        avg_val_loss = val_loss / n_eval
        print(f"  Valid loss: {avg_val_loss:.4f}")
        elapsed = format_time(time.time() - t0)     
        print(f" validation elapsed {elapsed}")
        stats.append({"epoch": epoch,
                      "train_loss": avg_train_loss,
                      "val_loss": avg_val_loss})
	
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss  # 更新最好的验证集损失
            epochs_no_improve = 0
            path = Path(RESULT_DIR, f"codebert_best.bin")
            torch.save(model.state_dict(), path)
            print(f"  >> Saved best model to {path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print(f"\n Early stopping at epoch {epoch+1} (patience = {patience})")
            break

    return stats

training_stats = train(model)

# ---------------------------------------------------------
# 9. 评估 + 绘图
# ---------------------------------------------------------
def get_predictions(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for b_ids, b_masks, _ in loader:
            b_ids, b_masks = b_ids.to(DEVICE), b_masks.to(DEVICE)
            p = model(b_ids, attention_mask=b_masks).sigmoid().cpu().numpy()
            probs.extend(p.tolist())
    return np.asarray(probs)

t0=time.time()
pred_probs = get_predictions(model, test_loader)
elapsed = format_time(time.time() - t0)   
print(f" test elapsed {elapsed}")
auc = roc_auc_score(test_y, pred_probs)
print(f"\n========== TEST ==========\nAUC = {auc:.4f}")

def evaluate(y_true, p, thr=0.5):
    cm = confusion_matrix(y_true, p>thr)
    TN, FP, FN, TP = cm.ravel()
    print(cm)
    print(f"TNs {TN} | FPs {FP} | FNs {FN} | TPs {TP}")
    print("precision", precision_score(y_true, p>thr))
    print("recall   ", recall_score(y_true,  p>thr))
    print("f1       ", f1_score(y_true,     p>thr))
    print("bal_acc  ", balanced_accuracy_score(y_true, p>thr))
    print("kappa    ", cohen_kappa_score(y_true, p>thr))
    prec, rec, _ = precision_recall_curve(y_true, p)
    print("AP (trapz)", trapz(rec, prec))
    print(classification_report(y_true, p>thr,target_names=["Non-vuln","Vuln"]))
    pq_rq = topk_metrics(p, y_true, ks=[50,100,150,200,250,300,350,400])
    for k,(pq,rq) in pq_rq.items():
        print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

evaluate(test_y, pred_probs, 0.5)

# ---------------------------------------------------------
# 10. 保存结果 + Loss 曲线
# ---------------------------------------------------------
result_df = pd.DataFrame({"Func_id": test_ids,
                          "prob":    pred_probs,
                          "Label":   test_y})
csv_name = f"CodeBERT_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
ListToCSV(result_df, Path(RESULT_DIR, csv_name))

# Loss 曲线
df_stats = pd.DataFrame(training_stats).set_index("epoch")
sns.set(style="darkgrid", font_scale=1.3)
plt.figure(figsize=(10,5))
plt.plot(df_stats["train_loss"], "b-", label="Train")
plt.plot(df_stats["val_loss"],   "g-", label="Valid")
plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout()
plt.savefig(Path(RESULT_DIR, f"loss_curve_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf"))
plt.show()