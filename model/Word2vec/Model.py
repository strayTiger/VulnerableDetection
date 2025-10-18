# -*- coding: utf-8 -*-
"""
Word2Vec + XGBoost baseline (aligned with GloVe & CodeBERT metrics)
Author: (your name)
"""

import json, time, datetime, random, os
from pathlib import Path
import re, numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt, seaborn as sns

# —— 与其他版本同用的工具 ——
from DataLoader import SplitCharacters, ListToCSV

# ---------------------------------------------------
# 1. 数据加载 + 分词
# ---------------------------------------------------
def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def tokenize_code(code: str):
    split_str = " ".join(SplitCharacters(tok) for tok in code.split())
    return split_str.split()

def prepare_sentences(dataset):
    return [tokenize_code(item["func"]) for item in dataset]

# ---------------------------------------------------
# 2. 句向量 = token 向量平均
# ---------------------------------------------------
def sent_vec(tokens, w2v, dim):
    vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)

def embed_dataset(dataset, w2v, dim):
    X, y = [], []
    for item in tqdm(dataset, desc="Vectorizing", ascii=True):
        X.append(sent_vec(tokenize_code(item["func"]), w2v, dim))
        y.append(item["target"])
    return np.asarray(X, np.float32), np.asarray(y, np.int64)

# ---------------------------------------------------
# 3. Top-k 评估
# ---------------------------------------------------
def eval_topk(y_true, y_prob, k):
    idx = np.argsort(y_prob)[::-1][:k]
    tp = y_true[idx].sum()
    p = tp / k
    r = tp / y_true.sum()
    return p, r

# ---------------------------------------------------
# 4. 主流程
# ---------------------------------------------------
def main():
    # ---------- 路径 ----------
    train_fp, valid_fp, test_fp = "train.jsonl", "valid.jsonl", "test.jsonl"
    dim = 100
    out_dir = Path("./result"); out_dir.mkdir(exist_ok=True)

    random.seed(42); np.random.seed(42)

    # ---------- 加载 ----------
    train_d = load_jsonl(train_fp)
    valid_d = load_jsonl(valid_fp)
    test_d  = load_jsonl(test_fp)

    # ---------- 训练 Word2Vec ----------
    print("Training Word2Vec ...")
    sentences = prepare_sentences(train_d + valid_d + test_d)
    w2v = Word2Vec(
        sentences,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )

    # ---------- 向量化 ----------
    X_tr, y_tr = embed_dataset(train_d,  w2v, dim)
    X_va, y_va = embed_dataset(valid_d,  w2v, dim)
    X_te, y_te = embed_dataset(test_d,   w2v, dim)

    # ---------- XGBoost ----------
    print("Training XGBoost ...")
    clf = XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        # tree_method="gpu_hist",  # 若需 GPU 加速，解除注释
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=False
    )
    evals_result = clf.evals_result()

    # ---------- 测试 ----------
    print("Testing ...")
    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("Report:\n", classification_report(y_te, y_pred))

    for k in [50, 100, 200, 300, 400, 500]:
        p, r = eval_topk(y_te, y_prob, k)
        print(f"P@{k:<3d} = {p:6.2%} | R@{k:<3d} = {r:6.2%}")

    # ---------- 保存 CSV ----------
    df_out = {
        "Func_id": [d["func_name"] for d in test_d],
        "prob":    y_prob,
        "Label":   y_te
    }
    csv_name = out_dir / f"Word2Vec_XGB_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    ListToCSV(df_out, csv_name)
    print(f"Saved CSV: {csv_name}")

    # ---------- Loss 曲线 ----------
    tr_loss = evals_result["validation_0"]["logloss"]
    va_loss = evals_result["validation_1"]["logloss"]
    sns.set(style="darkgrid"); plt.figure(figsize=(10,5))
    plt.plot(tr_loss, label="Train"); plt.plot(va_loss, label="Valid")
    plt.xlabel("Boosting Round"); plt.ylabel("Logloss"); plt.title("W2V-XGB Logloss")
    plt.legend(); plt.tight_layout()
    img_name = out_dir / f"w2v_loss_curve_{datetime.datetime.now():%Y%m%d%H%M%S}.pdf"
    plt.savefig(img_name); plt.close()
    print(f"Saved loss curve: {img_name}")

if __name__ == "__main__":
    main()
