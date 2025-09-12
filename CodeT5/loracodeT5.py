# -*- coding: utf-8 -*-

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
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler, TensorDataset

from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report)
from numpy import trapz

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW as TorchAdamW

from peft import LoraConfig, get_peft_model, TaskType

# 你的数据预处理工具
from DataLoader import SplitCharacters, ListToCSV
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== 全局配置 ====================
SEED          = 42
MAX_LEN       = 512
BATCH_SIZE    = 12
NUM_EPOCHS    = 4
LR_MAIN       = 2e-5        # 基座其它可训练参数
LR_LORA_HEAD  = 1e-4        # LoRA & 分类头更大学习率
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
GRAD_NORM     = 1.0

# 可选：是否使用 Focal Loss（用它时建议关闭 WeightedRandomSampler，避免双重补偿）
USE_FOCAL     = False
FOCAL_GAMMA   = 2.0
FOCAL_ALPHA   = 0.25        # 正类权重（可 None）

MODEL_DIR     = "./codeT5_model"   # 或 "Salesforce/codet5-base"
RESULT_DIR    = "./result"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
Path(RESULT_DIR).mkdir(exist_ok=True)

# 可复现
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 数据读取 ====================
def generate_id_label_code(json_path):
    ids, labels, codes = [], [], []
    with open(json_path, encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            ids.append(js["func_name"])
            labels.append(int(js["target"]))
            codes.append(SplitCharacters(js["func"]))  # 对整段代码处理
    return ids, labels, codes

data_dir = "./"
train_ids, train_y, train_code = generate_id_label_code(Path(data_dir, "train.jsonl"))
valid_ids, valid_y, valid_code = generate_id_label_code(Path(data_dir, "valid.jsonl"))
test_ids,  test_y,  test_code  = generate_id_label_code(Path(data_dir, "test.jsonl"))

print(f"Train {len(train_y)}  | positive {np.sum(train_y)}")
print(f"Valid {len(valid_y)}  | positive {np.sum(valid_y)}")
print(f"Test  {len(test_y)}   | positive {np.sum(test_y)}")

# ==================== 编码 ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # T5 家族兜底

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
train_masks   = torch.tensor(train_masks_np, dtype=torch.long)   # long/bool 更稳
train_labels  = torch.tensor(train_y,        dtype=torch.long)

valid_inputs  = torch.tensor(valid_ids_np,   dtype=torch.long)
valid_masks   = torch.tensor(valid_masks_np, dtype=torch.long)
valid_labels  = torch.tensor(valid_y,        dtype=torch.long)

test_inputs   = torch.tensor(test_ids_np,    dtype=torch.long)
test_masks    = torch.tensor(test_masks_np,  dtype=torch.long)
test_labels   = torch.tensor(test_y,         dtype=torch.long)

# ==================== 采样器/Loader ====================
# 默认：WeightedRandomSampler（若 USE_FOCAL=True，建议改成普通 RandomSampler）
train_y_np = np.asarray(train_y, dtype=int)
cls_sample_count = np.unique(train_y_np, return_counts=True)[1]
cls_weights      = 1. / cls_sample_count
samples_weight   = torch.from_numpy(cls_weights[train_y_np]).double()
weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_labels)
test_dataset  = TensorDataset(test_inputs,  test_masks,  test_labels)

train_loader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),  batch_size=BATCH_SIZE)

# ==================== 模型 + LoRA ====================
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)

# LoRA 目标层（T5 家族）：显存够可覆盖注意力 + FFN
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q","v"],
    modules_to_save=["classification_head"]  # 确保分类头可训练&保存
)
model = get_peft_model(base_model, lora_cfg).to(DEVICE)
model.print_trainable_parameters()

# 保险起见：显式解冻分类头
for n,p in model.named_parameters():
    if "classification_head" in n:
        p.requires_grad = True

# ==================== 优化器/调度器 ====================
# 只收集 requires_grad=True 的参数；LoRA&头给更大学习率
lora_params, head_params, other_params = [], [], []
for n,p in model.named_parameters():
    if not p.requires_grad: 
        continue
    if "classification_head" in n:
        head_params.append(p)
    elif "lora_" in n:
        lora_params.append(p)
    else:
        other_params.append(p)

no_decay_kw = ["bias", "LayerNorm.weight", "layer_norm.weight"]
def split_decay(params):
    decay, nodecay = [], []
    for p in params:
        name = ""  # 无法直接拿到名字，这里简单用全部 decay
        decay.append(p)
    return decay, nodecay  # 简化处理；若需更细分可改为带名字分组

decay_lora, _  = split_decay(lora_params)
decay_head, _  = split_decay(head_params)
decay_other, _ = split_decay(other_params)

optimizer = TorchAdamW([
    {"params": decay_lora,  "lr": LR_LORA_HEAD, "weight_decay": WEIGHT_DECAY},
    {"params": decay_head,  "lr": LR_LORA_HEAD, "weight_decay": WEIGHT_DECAY},
    {"params": decay_other, "lr": LR_MAIN,      "weight_decay": WEIGHT_DECAY},
])

total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps
)

# ==================== 可选 Focal Loss ====================
import torch.nn.functional as F
def focal_loss_ce(logits, targets, gamma=2.0, alpha=None):
    # logits: [B,2]; targets: Long [B]
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    ce = F.nll_loss(logp, targets, reduction='none')
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = torch.where(targets==1, torch.full_like(p_t, alpha),
                                         torch.full_like(p_t, 1-alpha))
        loss = alpha_t * loss
    return loss.mean()

# ==================== 工具函数 ====================
def format_time(sec): return str(datetime.timedelta(seconds=int(round(sec))))

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

def get_predictions(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for b_ids, b_masks, _ in loader:
            b_ids, b_masks = b_ids.to(DEVICE), b_masks.to(DEVICE)
            logits = model(input_ids=b_ids, attention_mask=b_masks).logits  # [B,2]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.asarray(probs)

def pick_best_threshold(y, probs):
    prec, rec, thr = precision_recall_curve(y, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.nanargmax(f1s)
    best_f1 = float(np.nanmax(f1s))
    best_thr = 0.5 if idx >= len(thr) else float(thr[idx])
    return best_thr, best_f1

def evaluate(y_true, p, thr=0.5, title="TEST"):
    print(f"\n========== {title} ==========")
    auc = roc_auc_score(y_true, p)
    y_pred = (p > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_   = recall_score(y_true,  y_pred, zero_division=0)
    f1        = f1_score(y_true,     y_pred, zero_division=0)
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap_trapz  = trapz(prec, rec)                         # 修正方向
    ap_avg    = average_precision_score(y_true, p)

    print(f"AUC={auc:.4f}  AP(trapz)={ap_trapz:.4f}  AP(avg)={ap_avg:.4f}  meanP={p.mean():.6f}")
    print(f"P={precision:.4f}  R={recall_:.4f}  F1={f1:.4f}  Kappa={kappa:.4f}  BalAcc={bal_acc:.4f}")
    print("Confusion Matrix:\n", cm)
    pq_rq = topk_metrics(p, y_true, ks=[50,100,150,200,250,300,350,400])
    for k,(pq,rq) in pq_rq.items():
        print(f"Top-{k}: PQ={pq*100:5.2f}% | RQ={rq*100:5.2f}%")
    print(classification_report(y_true, y_pred, target_names=["Non-vuln","Vuln"]))

# ==================== 训练 ====================
def train(model):
    best_metric = -1.0
    patience, min_delta = 5, 1e-4
    bad_epochs = 0
    stats = []

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
            out = model(input_ids=b_ids, attention_mask=b_masks, labels=b_y)
            # 默认 CE；若启用 Focal，用 logits 计算
            if USE_FOCAL:
                loss = focal_loss_ce(out.logits, b_y, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
            else:
                loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step(); scheduler.step()

            bs = b_y.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

            if torch.cuda.is_available() and step % 1000 == 0 and step:
                elapsed = format_time(time.time() - t0)
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"训练峰值显存: {peak:.2f} GB  Batch {step}/{len(train_loader)} | Elapsed {elapsed}")
                torch.cuda.reset_peak_memory_stats()

        avg_train_loss = tot_loss / max(1, n_samples)
        print(f"  Train loss: {avg_train_loss:.4f} | took {format_time(time.time()-t0)}")

        # ----- 验证：用 F1@best_thr 做早停/保存 -----
        model.eval()
        with torch.no_grad():
            val_probs = get_predictions(model, valid_loader)
        best_thr, f1_val = pick_best_threshold(np.asarray(valid_y), val_probs)
        # 同时算 val_loss（仅记录用，不作为早停判据）
        val_loss, n_eval = 0.0, 0
        with torch.no_grad():
            for b_ids, b_masks, b_y in valid_loader:
                b_ids, b_masks, b_y = b_ids.to(DEVICE), b_masks.to(DEVICE), b_y.to(DEVICE)
                out = model(input_ids=b_ids, attention_mask=b_masks, labels=b_y)
                val_loss += out.loss.item() * b_y.size(0)
                n_eval   += b_y.size(0)
        avg_val_loss = val_loss / max(1, n_eval)

        print(f"  Valid loss: {avg_val_loss:.4f} | Val best-F1: {f1_val:.4f} at thr={best_thr:.4f}")

        stats.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                      "val_best_f1": f1_val, "val_best_thr": best_thr})

        # 以 F1 为主判据
        if f1_val > best_metric + min_delta:
            best_metric = f1_val
            bad_epochs = 0
            best_path = Path(RESULT_DIR, "codet5_lora_best.bin")
            torch.save(model.state_dict(), best_path)   # 保存 LoRA+头 可训练权重
            # 同时保存阈值（便于一键评测）
            with open(Path(RESULT_DIR, "codet5_lora_best_thr.txt"), "w") as wf:
                wf.write(f"{best_thr:.6f}\n")
            print(f"  >> Saved best model to {best_path} (F1={best_metric:.4f}, thr={best_thr:.4f})")
        else:
            bad_epochs += 1
            print(f"  No improvement for {bad_epochs} epoch(s)")
            if bad_epochs >= patience:
                print(f"\n Early stopping at epoch {epoch+1} (patience={patience})")
                break

    return stats

training_stats = train(model)

# ==================== 测试：加载最优权重 + 用验证集最优阈值 ====================
best_path = Path(RESULT_DIR, "codet5_lora_best.bin")
if best_path.exists():
    missing, unexpected = model.load_state_dict(torch.load(best_path, map_location=DEVICE), strict=False)
    print(f"[INFO] Loaded best LoRA checkpoint from {best_path}")
else:
    print("[WARN] Best checkpoint not found. Using last-epoch weights.")

# 取保存的最佳阈值；若缺失就现场再算一遍
thr_path = Path(RESULT_DIR, "codet5_lora_best_thr.txt")
if thr_path.exists():
    with open(thr_path, "r") as rf:
        best_thr = float(rf.readline().strip())
else:
    with torch.no_grad():
        val_probs = get_predictions(model, valid_loader)
    best_thr, _ = pick_best_threshold(np.asarray(valid_y), val_probs)

t0 = time.time()
test_probs = get_predictions(model, test_loader)
print(f" test elapsed {format_time(time.time() - t0)}")
#evaluate(np.asarray(test_y), test_probs, thr=best_thr, title="TEST (F1-opt threshold)")
evaluate(np.asarray(test_y), test_probs, thr=0.5, title="TEST (F1-opt threshold)")

# ==================== 保存结果 + Loss 曲线 ====================
out_csv = Path(RESULT_DIR, f"CodeT5_LoRA_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
pd.DataFrame({"Func_id": test_ids, "prob": test_probs, "Label": test_y}).to_csv(out_csv, index=False)
print(f"结果保存到: {out_csv}")

df_stats = pd.DataFrame(training_stats).set_index("epoch")
sns.set(style="darkgrid", font_scale=1.2)
plt.figure(figsize=(10,5))
plt.plot(df_stats["train_loss"], "b-", label="Train")
plt.plot(df_stats["val_loss"],   "g-", label="Valid")
plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout()
plt.savefig(Path(RESULT_DIR, f"loss_curve_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf"))
plt.show()
