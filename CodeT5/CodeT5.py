# -*- coding: utf-8 -*-

# ---------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os, json, time, datetime, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              WeightedRandomSampler, TensorDataset)

from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report)
from numpy import trapz

# Hugging Face
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)

# 你自己的数据预处理函数（与之前保持一致）
from DataLoader import SplitCharacters, ListToCSV

# ---------------------------------------------------------
# 2. 全局配置
# ---------------------------------------------------------
SEED          = 42
MAX_LEN       = 512
BATCH_SIZE    = 12
NUM_EPOCHS    = 1
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
GRAD_NORM     = 1.0

# 使用 HuggingFace Hub 模型；若你有本地权重，把它改成本地目录路径
MODEL_DIR     = "./codeT5_model"
RESULT_DIR    = "./result"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 可复现性（如不需要可注释）
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            labels.append(int(js["target"]))                 # 确保是 int
            codes.append(SplitCharacters(js["func"]))        # 对整段代码做拆分
    return ids, labels, codes

data_dir = "./"
train_ids, train_y, train_code = generate_id_label_code(Path(data_dir, "train.jsonl"))
valid_ids, valid_y, valid_code = generate_id_label_code(Path(data_dir, "valid.jsonl"))
test_ids,  test_y,  test_code  = generate_id_label_code(Path(data_dir, "test.jsonl"))

print(f"Train {len(train_y)}  | positive {np.sum(train_y)}")
print(f"Valid {len(valid_y)}  | positive {np.sum(valid_y)}")
print(f"Test  {len(test_y)}   | positive {np.sum(test_y)}")

# ---------------------------------------------------------
# 4. Tokenizer & 编码
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

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
train_masks   = torch.tensor(train_masks_np, dtype=torch.long)   # 注意：用 long/bool 更稳
train_labels  = torch.tensor(train_y,        dtype=torch.long)

valid_inputs  = torch.tensor(valid_ids_np,   dtype=torch.long)
valid_masks   = torch.tensor(valid_masks_np, dtype=torch.long)
valid_labels  = torch.tensor(valid_y,        dtype=torch.long)

test_inputs   = torch.tensor(test_ids_np,    dtype=torch.long)
test_masks    = torch.tensor(test_masks_np,  dtype=torch.long)
test_labels   = torch.tensor(test_y,         dtype=torch.long)

# ---------------------------------------------------------
# 5. 类别不平衡 → WeightedRandomSampler
# ---------------------------------------------------------
train_y_np = np.asarray(train_y, dtype=int)
cls_sample_count = np.unique(train_y_np, return_counts=True)[1]
cls_weights      = 1. / cls_sample_count
samples_weight   = torch.from_numpy(cls_weights[train_y_np]).double()
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
# 6. CodeT5 分类模型（CE 二分类）
# ---------------------------------------------------------
# 直接用 HF 自带的分类头，num_labels=2 -> CrossEntropyLoss
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2).to(DEVICE)

# ---------------------------------------------------------
# 7. 优化器 + 调度器
# ---------------------------------------------------------
no_decay_keywords = ["bias", "LayerNorm.weight", "layer_norm.weight"]
opt_group = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_keywords)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_keywords)],
     "weight_decay": 0.0}
]
optimizer = AdamW(opt_group, lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(WARMUP_RATIO*total_steps),
                                            num_training_steps=total_steps)

# ---------------------------------------------------------
# 8. 工具函数
# ---------------------------------------------------------
def format_time(sec): return str(datetime.timedelta(seconds=int(round(sec))))

def topk_metrics(pred_probs, labels, ks=(10,20,30,40,50,100,150,200)):
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
# 9. 训练（含：按样本加权平均 Loss、早停、保存最优）
# ---------------------------------------------------------
def train(model):
    best_val_loss = float("inf")
    stats = []
    patience = 5
    min_delta = 1e-4
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} ========")
        t0 = time.time()
        model.train()
        tot_loss, n_samples = 0.0, 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_loader, start=1):
            b_ids, b_masks, b_y = (t.to(DEVICE) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=b_ids, attention_mask=b_masks, labels=b_y)  # CE loss
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step(); scheduler.step()

            # —— 按样本数加权累计，避免“再除一次 batch_size”的错误 ——
            bs = b_y.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

            if step % 1000 == 0 and step:
                elapsed = format_time(time.time() - t0)
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                    print(f"训练峰值显存: {peak_mem:.2f} GB  Batch {step}/{len(train_loader)} | Elapsed {elapsed}")
                    torch.cuda.reset_peak_memory_stats()

        avg_train_loss = tot_loss / max(1, n_samples)
        print(f"  Train loss: {avg_train_loss:.4f} | epoch took {format_time(time.time()-t0)}")

        # ---------- Validation ----------
        model.eval()
        val_loss, n_eval = 0.0, 0
        v0 = time.time()
        with torch.no_grad():
            for b_ids, b_masks, b_y in valid_loader:
                b_ids, b_masks, b_y = b_ids.to(DEVICE), b_masks.to(DEVICE), b_y.to(DEVICE)
                out = model(input_ids=b_ids, attention_mask=b_masks, labels=b_y)
                # —— 验证集同样做“按样本加权平均” ——
                val_loss += out.loss.item() * b_y.size(0)
                n_eval   += b_y.size(0)

        avg_val_loss = val_loss / max(1, n_eval)
        print(f"  Valid loss: {avg_val_loss:.4f}")
        print(f"  validation elapsed {format_time(time.time()-v0)}")

        stats.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # 早停 + 保存最优
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            path = Path(RESULT_DIR, "codet5_best.bin")
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
# 10. 评估
# ---------------------------------------------------------
def get_predictions(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for b_ids, b_masks, _ in loader:
            b_ids, b_masks = b_ids.to(DEVICE), b_masks.to(DEVICE)
            logits = model(input_ids=b_ids, attention_mask=b_masks).logits  # [B,2]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy() # 正类概率
            probs.extend(p.tolist())
    return np.asarray(probs)

def evaluate(y_true, p, thr=0.5):
    print("\n========== TEST ==========")
    auc = roc_auc_score(y_true, p)
    print(f"AUC = {auc:.4f}")
    print(f"The mean of predicted probabilities: {p.mean():.6f}")

    y_pred = (p > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_   = recall_score(y_true,  y_pred, zero_division=0)
    f1        = f1_score(y_true,     y_pred, zero_division=0)
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap_trapz  = trapz(prec, rec)                         # 修正：y=precision, x=recall
    ap_avg    = average_precision_score(y_true, p)

    print(f"(True Negatives): {TN}")
    print(f"(False Positives): {FP}")
    print(f"(False Negatives): {FN}")
    print(f"(True Positives): {TP}")
    print(f"precision : {precision:.6f}")
    print(f"recall    : {recall_:.6f}")
    print(f"f1        : {f1:.6f}")
    print(f"kappa     : {kappa:.6f}")
    print(f"balanced_accuracy : {bal_acc:.6f}")
    print(f"AP (trapz): {ap_trapz:.6f}")
    print(f"AP (avg)  : {ap_avg:.6f}")
    print("\nConfusion Matrix:\n", cm, "\n")
    print(classification_report(y_true, y_pred, target_names=["Non-vuln","Vuln"]))

    pq_rq = topk_metrics(p, y_true, ks=[10,20,30,40,50,100,150,200])
    for k,(pq,rq) in pq_rq.items():
        print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

# —— 测试前：加载最佳权重 ——
best_path = Path(RESULT_DIR, "codet5_best.bin")
if best_path.exists():
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded best checkpoint from {best_path}")
else:
    print("[WARN] Best checkpoint not found. Using last-epoch weights.")

t0 = time.time()
pred_probs = get_predictions(model, test_loader)
print(f" test elapsed {format_time(time.time() - t0)}")
evaluate(test_y, pred_probs, thr=0.5)

# ---------------------------------------------------------
# 11. 保存结果 + Loss 曲线
# ---------------------------------------------------------
result_df = pd.DataFrame({"Func_id": test_ids,
                          "prob":    pred_probs,
                          "Label":   test_y})
csv_name = f"CodeT5_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
ListToCSV(result_df, Path(RESULT_DIR, csv_name))
print(f"结果保存到: {Path(RESULT_DIR, csv_name)}")

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
