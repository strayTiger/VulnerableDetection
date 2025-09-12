
import os, json, time, datetime, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score,
                             average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, classification_report)
from numpy import trapz

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ===================== 基础配置 =====================
MODEL_NAME      = "./graphcodebert_model"
MAX_LEN         = 512
BATCH_SIZE      = 12
NUM_EPOCHS      = 2
LR              = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
GRAD_NORM       = 1.0
LOG_EVERY       = 40           # 每多少个batch打印一次中间日志
PATIENCE        = 5            # 早停
MIN_DELTA       = 1e-4

SEED            = 42
RESULT_DIR      = "./result"
BEST_DIR_NAME   = "graphcodebert_best"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
            if not line:
                continue
            js = json.loads(line)
            ids.append(js["func_name"])
            labels.append(int(js["target"]))
            code = js["func"]
            if use_split:
                code = SplitCharacters(code)  # 先用DataLoader风格拆符号
            codes.append(code)
    return ids, np.array(labels, dtype=np.int64), codes

def encode_codes(tokenizer, code_list, max_len=MAX_LEN):
    enc = tokenizer(code_list,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

# ===================== 训练/验证/测试工具 =====================
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
    plt.savefig(out)
    plt.close()
    print("loss曲线生成完成")

def get_predictions(model, loader):
    model.eval(); probs = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B,2]
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()             # 正类概率
            probs.extend(p.tolist())
    return np.asarray(probs)

def evaluate(y_true, p, thr=0.5):
    print("开始测试")
    print("Testing...")
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

    print(f" The mean of predicted probabilities: {p.mean():.4f}")
    print(f"auc  on test {bal_acc}")  
    print(f" (True Negatives): {TN}")
    print(f" (False Negatives):  {FN}")
    print(f" (True Positives): {TP}")
    print(f"(False Positives):{FP}")
    print(f"Total positive :  {y_true.sum()}")
    print(f"auc :{auc}")
    print(f"precision :{precision}")
    print(f"recall :{recall}")
    print(f"f1 :{f1}")
    print(f"Area Under Precision Recall Curve(AP): {ap_trapz:.4f}")
    print(f"average precision: {ap_avg:.4f}")
    print(f"kappa :{kappa}")
    print(f"balanced_accuracy :{bal_acc}")
    print(cm); print()
    print(classification_report(y_true, y_pred, target_names=["Non-vulnerable","Vulnerable"]))

    pq_rq = topk_metrics(p, y_true, ks=[10,20,30,40,50,100,150,200])
    for k,(pq,rq) in pq_rq.items():
        print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

# ===================== 主流程 =====================
def main():
    data_dir = "./"
    train_json = Path(data_dir, "train.jsonl")
    valid_json = Path(data_dir, "valid.jsonl")
    test_json  = Path(data_dir, "test.jsonl")

    # 读取数据（默认先做符号拆分，再Tokenizer）
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
    train_masks   = train_masks_np.float()
    train_labels  = torch.tensor(train_y, dtype=torch.long)

    valid_inputs  = valid_ids_np.long()
    valid_masks   = valid_masks_np.float()
    valid_labels  = torch.tensor(valid_y, dtype=torch.long)

    test_inputs   = test_ids_np.long()
    test_masks    = test_masks_np.float()
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

    train_loader = DataLoader(train_dataset, sampler=weighted_sampler, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),  batch_size=BATCH_SIZE)

    # 模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    # 优化器 + 调度器
    no_decay = ["bias", "LayerNorm.weight"]
    opt_group = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(opt_group, lr=LR)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(WARMUP_RATIO * total_steps),
                                                num_training_steps=total_steps)

    # 训练循环（含详细日志 + 早停 + 保存最好模型）
    training_stats = []
    best_val_loss = float("inf")
    bad_epochs = 0
    total_t0 = time.time()

    epoch_bar = tqdm(range(NUM_EPOCHS), desc="Epoch", total=NUM_EPOCHS)
    for epoch in epoch_bar:
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} ========")
        t0 = time.time()
        model.train()
        tot_loss, n_samples = 0.0, 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step(); scheduler.step()

            bs = labels.size(0)
            tot_loss += loss.item() * bs
            n_samples += bs

            if step % LOG_EVERY == 0 or step == len(train_loader):
                elapsed = format_time(time.time() - t0)
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"训练峰值显存: {peak_mem:5.2f} GB")
                    torch.cuda.reset_peak_memory_stats()
                avg_loss_so_far = tot_loss / max(1, n_samples)
                print(f"  Batch {step:,}  of  {len(train_loader):,}.    Elapsed: {elapsed}. "
                      f"Average training loss: {avg_loss_so_far:.2f}")

        print(f"  Training epcoh took: {format_time(time.time() - t0)}")  

        # 验证
        print("\nRunning Validation...")
        model.eval()
        val_loss, n_eval = 0.0, 0
        v0 = time.time()
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item() * labels.size(0)
                n_eval += labels.size(0)

        avg_train_loss = tot_loss / max(1, n_samples)
        avg_val_loss   = val_loss / max(1, n_eval)
        print(f"Valid loss: {avg_val_loss}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {format_time(time.time() - v0)}")

        training_stats.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # 早停 + 保存最优
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            bad_epochs = 0
            save_dir = Path(RESULT_DIR, BEST_DIR_NAME)
            save_dir.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(save_dir)
            print(f"  >> Saved best model to {save_dir}")
        else:
            bad_epochs += 1
            print(f"  No improvement for {bad_epochs} epoch(s)")
            if bad_epochs >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (patience = {PATIENCE})")
                break

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

    # Loss 曲线
    plot_loss_curve(training_stats, RESULT_DIR)

    # 测试
    t0 = time.time()
    pred_probs = get_predictions(model, test_loader)
    print(f" test elapsed {format_time(time.time() - t0)}")
    evaluate(test_y, pred_probs, thr=0.5)

    # 结果保存
    out_csv = Path(RESULT_DIR, f"GraphCodeBERT_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    df_out = pd.DataFrame({"Func_id": test_ids, "prob": pred_probs, "Label": test_y})
    ListToCSV(df_out, out_csv)
    print(f"结果保存到: {out_csv}")

if __name__ == "__main__":
    main()
