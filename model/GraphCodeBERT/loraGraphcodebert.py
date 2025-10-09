import os
import json
import time
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report)
from numpy import trapz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 基础配置 =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#MODEL_NAME      = "microsoft/graphcodebert-base"
MODEL_NAME      = "./graphcodebert_model"
MAX_LEN         = 512      # 显存紧张可改 384/320/256
BATCH_SIZE      = 12
NUM_EPOCHS      = 1
LR              = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
GRAD_NORM       = 1.0
LOG_EVERY       = 40
PATIENCE        = 5
MIN_DELTA       = 1e-4

SEED            = 42
RESULT_DIR      = "./result"
BEST_DIR_NAME   = "graphcodebert_best"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 速度/显存权衡开关
# USE_GC = True → 训练时不保存中间激活，在反传阶段重新计算，从而大幅省显存；代价是每步更慢
USE_GC          = False                         # 是否启用梯度检查点（省显存但变慢）
NUM_WORKERS     = 0 if os.name == "nt" else 2   # Windows 上 0 往往更稳更快
PIN_MEMORY      = torch.cuda.is_available()
PROFILE_MEM     = True                          # 打开显存监控（会触发 GPU 同步，略慢；追速度就改 False）

# ===== LoRA 配置（省显存默认）=====
LORA_RANK       = 8
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LORA_TARGETS    = ["query", "value", "key"]            # 需要更强再加 "key"
MODULES_TO_SAVE = ["classifier"]                # 保存分类头，保证“最佳”可复现

os.makedirs(RESULT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Ampere+ 打开 TF32，加速 matmul/conv（不影响数值正确性）
    try:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

# ===================== 兼容 DataLoader.py =====================
def _fallback_split_chars(s: str) -> str:
    s = s.replace("->", " -> ").replace(">>", " >> ").replace("<<", " << ")
    for t in ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',', '[', ']', '>', '<', '"']:
        s = s.replace(t, f" {t} ")
    return ' '.join(s.split())

def _fallback_list_to_csv(df_like, path):
    pd.DataFrame(df_like).to_csv(path, index=False)

try:
    from DataLoader import SplitCharacters as SplitCharacters_ext
    SplitCharacters = SplitCharacters_ext
except Exception:
    SplitCharacters = _fallback_split_chars

try:
    from DataLoader import ListToCSV as ListToCSV_ext
    ListToCSV = ListToCSV_ext
except Exception:
    ListToCSV = _fallback_list_to_csv

# ===================== 数据读取与编码 =====================
def generate_id_label_code(json_path, use_split=True):
    ids, labels, codes = [], [], []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            js = json.loads(line)
            ids.append(js["func_name"])
            labels.append(int(js["target"]))
            code = js["func"]
            if use_split:
                code = SplitCharacters(code)
            codes.append(code)
    return ids, np.array(labels, dtype=np.int64), codes

def encode_codes(tokenizer, code_list, max_len=MAX_LEN):
    enc = tokenizer(code_list,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

# ===================== LoRA =====================
def add_lora_to_model(model, rank=LORA_RANK):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        use_rslora=True,
        modules_to_save=MODULES_TO_SAVE,  # 确保分类头一起保存
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ===================== 工具函数 =====================
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

def plot_loss_curve(training_stats, out_dir):
    df = pd.DataFrame(training_stats).set_index("epoch")
    plt.figure(figsize=(10,5))
    plt.plot(df["train_loss"], label="Train")
    plt.plot(df["val_loss"],   label="Valid")
    plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.tight_layout()
    out = Path(out_dir, f"loss_curve_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf")
    plt.savefig(out); plt.close()
    print("loss曲线生成完成")

def get_predictions(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
    return np.asarray(probs)

def evaluate(y_true, p, thr=0.5, title="Test"):
    auc = roc_auc_score(y_true, p)
    y_pred = (p > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true,  y_pred, zero_division=0)
    f1        = f1_score(y_true,     y_pred, zero_division=0)
    bal_acc   = balanced_accuracy_score(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap_trapz  = trapz(rec, prec)
    ap_avg    = average_precision_score(y_true, p)

    print(f"\n===== {title} (thr={thr:.4f}) =====")
    print(f"Mean prob: {p.mean():.4f}")
    print(f"AUC: {auc:.4f} | balanced_acc: {bal_acc:.4f} | kappa: {kappa:.4f}")
    print(f"precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")
    print(f"AP(trapz): {ap_trapz:.4f} | AP: {ap_avg:.4f}")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["Non-vulnerable","Vulnerable"]))

    pq_rq = topk_metrics(p, y_true, ks=[10,20,30,40,50,100,150,200])
    for k,(pq,rq) in pq_rq.items():
        print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

def best_f1_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    idx = int(np.nanargmax(f1s))
    return float(thr[idx]), float(f1s[idx])

# ---- 显存日志工具 ----
def log_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    alloc = torch.cuda.max_memory_allocated(dev)
    reserv = torch.cuda.max_memory_reserved(dev)
    GB = 1024 ** 3
    print(
        f"[GPU MEM]{' ' + tag if tag else ''} | "
        f"peak allocated: {alloc/GB:.2f} GB ({alloc/total*100:.1f}%) | "
        f"peak reserved:  {reserv/GB:.2f} GB ({reserv/total*100:.1f}%)"
    )

# ===================== 主流程 =====================
def main():
    total_t0 = time.time()
    data_dir = "./"
    train_json = Path(data_dir, "train.jsonl")
    valid_json = Path(data_dir, "valid.jsonl")
    test_json  = Path(data_dir, "test.jsonl")

    # 读取数据
    train_ids, train_y, train_code = generate_id_label_code(train_json, use_split=True)
    valid_ids, valid_y, valid_code = generate_id_label_code(valid_json, use_split=True)
    test_ids,  test_y,  test_code  = generate_id_label_code(test_json,  use_split=True)

    print(f"Train {len(train_y)}  | positive {int(train_y.sum())}")
    print(f"Valid {len(valid_y)}  | positive {int(valid_y.sum())}")
    print(f"Test  {len(test_y)}   | positive {int(test_y.sum())}")

    # Tokenizer & 编码
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ids_np, train_masks_np = encode_codes(tokenizer, train_code, MAX_LEN)
    valid_ids_np, valid_masks_np = encode_codes(tokenizer, valid_code, MAX_LEN)
    test_ids_np,  test_masks_np  = encode_codes(tokenizer, test_code,  MAX_LEN)

    train_inputs  = train_ids_np.long()
    train_masks   = train_masks_np.long()   # 用 long/bool 更稳
    train_labels  = torch.tensor(train_y, dtype=torch.long)

    valid_inputs  = valid_ids_np.long()
    valid_masks   = valid_masks_np.long()
    valid_labels  = torch.tensor(valid_y, dtype=torch.long)

    test_inputs   = test_ids_np.long()
    test_masks    = test_masks_np.long()
    test_labels   = torch.tensor(test_y, dtype=torch.long)

    # 采样器（类别不平衡）
    cls_sample_count = np.unique(train_y, return_counts=True)[1]
    cls_weights      = 1. / cls_sample_count
    samples_weight   = torch.from_numpy(cls_weights[train_y]).double()
    weighted_sampler = WeightedRandomSampler(samples_weight,
                                             len(samples_weight),
                                             replacement=True)

    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_labels)
    test_dataset  = TensorDataset(test_inputs,  test_masks,  test_labels)

    train_loader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)
    test_loader  = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),  batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)

    # 加载模型 + LoRA
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    if USE_GC:
        base.config.use_cache = False
        if torch.cuda.is_available():
            try:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base.gradient_checkpointing_enable()
    model = add_lora_to_model(base, rank=LORA_RANK).to(DEVICE)

    # 优化器 + 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(WARMUP_RATIO * total_steps),
                                                num_training_steps=total_steps)

    # 训练与验证
    best_val_loss = float("inf")
    bad_epochs = 0
    training_stats = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} ========")
        t0 = time.time()
        model.train()
        tot_loss, n_samples = 0.0, 0

        # 每个 epoch 统计一次峰值（若想看全程最大值，可把这行去掉）
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, labels = [t.to(DEVICE, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

            if step % LOG_EVERY == 0 or step == len(train_loader):
                elapsed = format_time(time.time() - t0)
                avg_loss_so_far = tot_loss / max(1, n_samples)
                print(f"  Batch {step:,}/{len(train_loader):,} | Elapsed: {elapsed} | Avg loss: {avg_loss_so_far:.4f}")
                # 区间峰值（可选）：打印并重置，便于观察本区间最高显存
                if PROFILE_MEM and torch.cuda.is_available():
                    log_gpu_mem(tag=f"epoch{epoch+1} step{step}")
                    torch.cuda.reset_peak_memory_stats()

        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        if PROFILE_MEM and torch.cuda.is_available():
            # 本 epoch 全程峰值
            log_gpu_mem(tag=f"epoch{epoch+1} total")

        # 验证
        print("\nRunning Validation...")
        model.eval()
        val_loss, n_eval = 0.0, 0
        v0 = time.time()
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = [t.to(DEVICE, non_blocking=True) for t in batch]
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item() * labels.size(0)
                n_eval += labels.size(0)

        avg_train_loss = tot_loss / max(1, n_samples)
        avg_val_loss   = val_loss / max(1, n_eval)
        print(f"  Validation Loss: {avg_val_loss:.4f} | took {format_time(time.time() - v0)}")

        training_stats.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # 早停 + 保存最优（LoRA 适配器 + 分类头）
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            save_dir = Path(RESULT_DIR, BEST_DIR_NAME)
            save_dir.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  >> Saved BEST to {save_dir}")
        else:
            bad_epochs += 1
            print(f"  No improvement for {bad_epochs} epoch(s)")
            if bad_epochs >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")
    if PROFILE_MEM and torch.cuda.is_available():
        # 注意：若每个 epoch 都 reset 了，这里展示的是最后一个 epoch 的峰值
        log_gpu_mem(tag="final")

    # Loss 曲线
    plot_loss_curve(training_stats, RESULT_DIR)

    # ==== 加载“验证集最优”（适配器+分类头）评测 ====
    best_dir = Path(RESULT_DIR, BEST_DIR_NAME)
    if best_dir.exists():
        base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
        model = PeftModel.from_pretrained(base, best_dir).to(DEVICE)
        print(">> Loaded TRUE best (adapter + classifier) for evaluation.")
    else:
        print("[Warn] BEST dir not found. Using last-epoch model for evaluation.")

    # ==== 验证集上找最佳阈值 ====
    val_loader_for_pred = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    t0 = time.time()
    val_probs = get_predictions(model, val_loader_for_pred)
    thr_best, f1_best = best_f1_threshold(valid_y, val_probs)
    print(f"\n>> Best F1 threshold on VALID: thr={thr_best:.4f} | F1={f1_best:.4f} | took {format_time(time.time() - t0)}")

    # 测试
    t0 = time.time()
    test_loader_for_pred = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_probs = get_predictions(model, test_loader_for_pred)
    print(f" test elapsed {format_time(time.time() - t0)}")
    evaluate(test_y, test_probs, thr=0.5,      title="Test @ thr=0.5")
    evaluate(test_y, test_probs, thr=thr_best, title="Test @ thr=bestF1(valid)")

    # 结果保存
    out_csv = Path(RESULT_DIR, f"GraphCodeBERT_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    df_out = pd.DataFrame({"Func_id": test_ids, "prob": test_probs, "Label": test_y})
    ListToCSV(df_out, out_csv)
    print(f"结果保存到: {out_csv}")

if __name__ == "__main__":
    main()

