# -*- coding: utf-8 -*-
"""
GloVe + XGBoost baseline (with AUC / Loss curve / optional GPU memory)
Author: (your name)
"""

import json, time, datetime, re, os, random, psutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, precision_recall_curve
)
# ---------- CodeBERT 同款工具 ----------
from DataLoader import SplitCharacters, ListToCSV

# ================= 1. 载入 GloVe =================
def load_glove(path, dim=100):
    print(f"Loading GloVe: {path}")
    emb = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, ascii=True):
            parts = line.rstrip().split()
            if len(parts) == dim + 1:
                emb[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return emb

# ================= 2. 数据 & 分词 =================
def load_jsonl(fp): return [json.loads(l) for l in open(fp, encoding="utf-8")]

def tokenize_code(code: str):
    split_str = " ".join(SplitCharacters(tok) for tok in code.split())
    return split_str.split()

def sent_vec(tokens, glove, dim):
    vecs = [glove[w] for w in tokens if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)

def embed_dataset(data, glove, dim):
    X, y = [], []
    for item in tqdm(data, desc="Vectorizing", ascii=True):
        X.append(sent_vec(tokenize_code(item["func"]), glove, dim))
        y.append(item["target"])
    return np.asarray(X, np.float32), np.asarray(y, np.int64)

# ================= 3. Top-k 评估 =================
def eval_topk(y_true, y_prob, k):
    idx = np.argsort(y_prob)[::-1][:k]
    tp = y_true[idx].sum()
    return tp / k, tp / y_true.sum()

# ================= 4. 训练与评估 =================
def main():
    # ---------- 路径 ----------
    train_fp, valid_fp, test_fp = "train.jsonl", "valid.jsonl", "test.jsonl"
    glove_fp = "glove.6B.100d.txt"      # ← 替换
    dim = 100
    out_dir = Path("./result"); out_dir.mkdir(exist_ok=True)

    random.seed(42); np.random.seed(42)

    # ---------- 数据 ----------
    train_d = load_jsonl(train_fp)
    valid_d = load_jsonl(valid_fp)
    test_d  = load_jsonl(test_fp)
    glove = load_glove(glove_fp, dim)

    X_tr, y_tr = embed_dataset(train_d,  glove, dim)
    X_va, y_va = embed_dataset(valid_d,  glove, dim)
    X_te, y_te = embed_dataset(test_d,   glove, dim)

    # ---------- 模型 ----------
    print("Training XGBoost ...")
    clf = XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        # booster="gbtree",            # 若想 GPU 训练改成:
        # tree_method="gpu_hist",      # booster 默认一样
    )

    # 记录 eval_metric
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],  # train 0, valid 1
        verbose=False
    )

    evals_result = clf.evals_result()

    # ---------- 显存（仅 GPU） ----------
    # 若启用 gpu_hist，可用下列代码打印峰值 (需安装 pynvml)
    """
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    mem = nvmlDeviceGetMemoryInfo(handle)
    print(f'GPU Mem Used after training: {mem.used/1024**3:.2f} GB')
    """

    # ---------- 测试 ----------
    print("Testing ...")
    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    auc  = roc_auc_score(y_te, y_prob)
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("Report:\n", classification_report(y_te, y_pred))

    for k in [50, 100, 200, 300, 400, 500]:
        p, r = eval_topk(y_te, y_prob, k)
        print(f"P@{k:<3d} = {p:6.2%} | R@{k:<3d} = {r:6.2%}")

    # ---------- 保存预测 ----------
    df_out = {
        "Func_id": [d["func_name"] for d in test_d],
        "prob":    y_prob,
        "Label":   y_te
    }
    csv_name = out_dir / f"GloVe_XGB_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    ListToCSV(df_out, csv_name)
    print(f"Saved CSV: {csv_name}")

    # ---------- Loss 曲线 ----------
    tr_loss = evals_result["validation_0"]["logloss"]  # 训练集
    va_loss = evals_result["validation_1"]["logloss"]  # 验证集

    import matplotlib.pyplot as plt, seaborn as sns
    sns.set(style="darkgrid"); plt.figure(figsize=(10,5))
    plt.plot(tr_loss, label="Train"); plt.plot(va_loss, label="Valid")
    plt.xlabel("Boosting Round"); plt.ylabel("Logloss"); plt.title("XGBoost Logloss")
    plt.legend(); plt.tight_layout()
    img_name = out_dir / f"xgb_loss_curve_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf"
    plt.savefig(img_name); plt.close()
    print(f"Saved loss curve: {img_name}")

if __name__ == "__main__":
    main()
