# -*- coding: utf-8 -*-
r"""
GraphCodeBERT 漏洞二分类（与 XLNet/CodeBERT 版功能对齐）

用法示例（推荐基线）：
python cwe_graphcodebert.py --data-root "E:\graphCodebert" --model graphcodebert_model --strip-comments --strip-strings --mask-sard-hints --epochs 3
python cwe_graphcodebert.py --data-root "E:\graphCodebert" --model graphcodebert_model --strip-comments --strip-strings --mask-all-identifiers --epochs 3
"""

import os
import re
import time
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler
from torch.optim import AdamW
from torch import amp  # torch.amp.autocast / GradScaler（新 API）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

# -------------------- 通用工具 --------------------
KS = (10, 20, 30, 40, 50, 100, 150, 200)  # Top-K 评价点

def set_seed(seed):
    """固定随机种子，便于复现"""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    """秒 -> hh:mm:ss"""
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def topk_metrics(pred_probs, labels, ks=KS):
    """
    排序型 Top-K Precision/Recall（不依赖阈值）
    pred_probs: 模型对“正类”的分数/概率
    labels    : 真实 0/1 标签
    """
    scores = np.asarray(pred_probs).flatten()
    labels = np.asarray(labels).flatten()
    total_pos = labels.sum() if len(labels) else 0
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    res = {}
    for k in ks:
        if k > len(sorted_labels): 
            continue
        tp = int(sorted_labels[:k].sum())
        pq = tp / k
        rq = tp / total_pos if total_pos else 0.0
        res[k] = (pq, rq)
    return res

# -------------------- 命令行参数 --------------------
def get_args():
    ap = argparse.ArgumentParser(description="GraphCodeBERT CWE 漏洞检测训练/评估")

    # 数据与模型
    ap.add_argument("--data-root", required=True, help="数据根目录，里面有 train/val/test 三个子目录")
    ap.add_argument("--model", default="graphcodebert_model",
                    help="HuggingFace 模型名或本地 GraphCodeBERT 权重路径")
    ap.add_argument("--max-len", type=int, default=512, help="序列最大长度（超过会截断）")

    # 训练超参
    ap.add_argument("--batch-size", type=int, default=28)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=5, help="早停耐心值（验证集无显著下降的轮数）")
    ap.add_argument("--threshold", type=float, default=0.5, help="将概率转为 0/1 的默认阈值")

    # 训练策略
    ap.add_argument("--shuffle", action="store_true", help="读取 train 文本时先随机打乱")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-weighted-sampler", action="store_true", help="训练集使用类权重采样（类别不平衡时可用）")
    ap.add_argument("--random-init", action="store_true", help="不加载预训练权重，随机初始化（sanity check）")

    # 代码清洗（防泄露）
    ap.add_argument("--strip-comments", action="store_true",
                    help="移除 C/C++ 注释（/*...*/ 与 //...），降低注释提示造成的泄露")
    ap.add_argument("--strip-strings", action="store_true",
                    help="把字符串/字符字面量统一替换为占位符 \"STR\"（字符改为 'C' 的形态）")
    ap.add_argument("--mask-sard-hints", action="store_true",
                    help="掩蔽 good/bad/G2B/B2G/Sink/Source 等提示性标识符为 ID（防止靠名字猜标签）")
    ap.add_argument("--mask-sard-scope", choices=["id", "func"], default="id",
                    help="id: 替换所有标识符(默认); func: 仅替换疑似函数/函数指针名")
    ap.add_argument("--mask-all-identifiers", action="store_true",
                    help="将所有类似标识符的词统一替换为 ID（用于泄露路径自检）")

    ap.add_argument("--result-dir", default="./result", help="输出目录（模型/图表/CSV）")
    return ap.parse_args()

# -------------------- 从文件名解析标签/CWE --------------------
PAT_OK     = re.compile(r'0[_-]0\.txt$', re.IGNORECASE)   # ……0_0.txt / 0-0.txt -> 正常(0)
PAT_VULN   = re.compile(r'0[_-]1\.txt$', re.IGNORECASE)   # ……0_1.txt / 0-1.txt -> 漏洞(1)
PAT_LAST01 = re.compile(r'([01])\.txt$', re.IGNORECASE)   # 兜底：……0.txt / ……1.txt
PAT_CWE    = re.compile(r'(CWE-\d+)', re.IGNORECASE)      # 提取文件名中的 CWE-xxx

def infer_label_from_name(name: str) -> int:
    """根据文件名推断标签（0=正常, 1=漏洞）"""
    if PAT_OK.search(name):   return 0
    if PAT_VULN.search(name): return 1
    m = PAT_LAST01.search(name)
    if m: return 0 if m.group(1) == '0' else 1
    raise ValueError(f"无法从文件名推断标签: {name}")

def parse_cwe_from_name(name: str) -> str:
    """从文件名解析 CWE 编号"""
    m = PAT_CWE.search(name)
    return m.group(1).upper() if m else "UNKNOWN"

# -------------------- 代码清洗（可选） --------------------
# 正则：注释、字符串、字符
C_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
C_LINE_COMMENT  = re.compile(r"//.*?$", re.MULTILINE)
STR_RE = re.compile(r'(?P<prefix>L|u8|u|U)?(?P<body>"(?:\\.|[^"\\])*")')
CHR_RE = re.compile(r"(?P<prefix>L|u8|u|U)?(?P<body>'(?:\\.|[^'\\])*')")

# SARD 提示词（大小写不敏感，带后缀）
SARD_HINTS = re.compile(r"(?i)\b(?:good|bad|g2b|b2g|sink|source)\w*\b")

def _mask_sard_identifiers(s: str, scope: str):
    if scope == "id":
        return SARD_HINTS.sub("ID", s)
    elif scope == "func":
        # 仅疑似函数名（name(...)）与函数指针名（(*name)(...)）
        s = re.sub(r"(?i)\b(?:good|bad|g2b|b2g|sink|source)\w*(?=\s*\()", "ID", s)
        s = re.sub(r"(?i)(\(\s*\*\s*)(?:good|bad|g2b|b2g|sink|source)\w*(?=\s*\))", r"\1ID", s)
        return s
    return s

def sanitize_c_code(src: str,
                    strip_comments=True, strip_strings=True,
                    mask_sard_hints=False, mask_sard_scope="id",
                    mask_all_identifiers=False,
                    str_placeholder="STR", char_placeholder="C"):
    """
    清洗 C/C++ 源码：
    1) strip_strings=True  → 字符串/字符字面量替换为 "STR" / 'C'（保持外观）
    2) strip_comments=True → 删除 /*...*/ 与 //... 注释
    3) mask_sard_hints     → 将 good/bad/G2B/B2G/Sink/Source 等提示性标识符替换为 'ID'
       mask_sard_scope=id  → 任意标识符
       mask_sard_scope=func→ 仅函数名/函数指针名
    4) mask_all_identifiers→ 所有疑似标识符（\b[A-Za-z_]\w*\b）替换为 'ID'
    5) 压缩空白
    """
    s = src
    # 1) 先处理字符串/字符
    if strip_strings:
        s = STR_RE.sub(lambda m: f'{m.group("prefix") or ""}"{str_placeholder}"', s)
        s = CHR_RE.sub(lambda m: f"{m.group('prefix') or ''}'{char_placeholder}'", s)
    # 2) 删注释
    if strip_comments:
        s = C_BLOCK_COMMENT.sub(" ", s)
        s = C_LINE_COMMENT.sub(" ", s)
    # 3) 掩蔽 SARD 提示词
    if mask_sard_hints:
        s = _mask_sard_identifiers(s, mask_sard_scope)
    # 4) （大锤）掩蔽所有“像标识符”的词（用于泄露自检）
    if mask_all_identifiers:
        s = re.sub(r"\b[A-Za-z_]\w*\b", "ID", s)
    # 5) 压缩空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------- 读取数据 --------------------
def load_split_from_dir(split_dir: Path,
                        strip_comments=True, strip_strings=True,
                        mask_sard_hints=False, mask_sard_scope="id",
                        mask_all_identifiers=False,
                        shuffle=False, seed=42):
    """
    读取某个 split（train/val/test）目录下所有 .txt 文件：
    返回 (ids, labels, texts, cwes, paths)
    """
    files = list(split_dir.glob("*.txt"))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)

    ids, labels, texts, cwes, paths = [], [], [], [], []
    for p in files:
        try:
            y = infer_label_from_name(p.name)
        except ValueError:
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        txt = sanitize_c_code(
            raw,
            strip_comments=strip_comments,
            strip_strings=strip_strings,
            mask_sard_hints=mask_sard_hints,
            mask_sard_scope=mask_sard_scope,
            mask_all_identifiers=mask_all_identifiers
        )
        ids.append(p.stem)
        labels.append(y)
        texts.append(txt)
        cwes.append(parse_cwe_from_name(p.name))
        paths.append(str(p))
    return ids, labels, texts, cwes, paths

# -------------------- 模型：GraphCodeBERT + 掩码均值池化 --------------------
class GraphCodeBERTVD(nn.Module):
    """
    使用 GraphCodeBERT（RoBERTa-like）提取最后一层隐向量，
    对 padding 做掩码均值，再接线性层输出单一 logit（二分类）。
    说明：本实现为“文本输入”模式，未显式引入 DFG；如需 DFG 可在此基础上扩展。
    """
    def __init__(self, model_name="microsoft/graphcodebert-base", random_init=False):
        super().__init__()
        if random_init:
            cfg = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(cfg)
        else:
            self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size  # 通常 768
        self.classifier = nn.Linear(hidden, 1)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        last_hidden = out[0]  # [bs, L, H]
        if attention_mask is None:
            pooled = last_hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()      # [bs, L, 1]
            summed = (last_hidden * mask).sum(dim=1)         # [bs, H]
            denom = mask.sum(dim=1).clamp_min(1e-6)          # [bs, 1]
            pooled = summed / denom
        logit = self.classifier(pooled).squeeze(-1)          # [bs]

        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logit, labels.float())
            return loss, logit
        return logit

# -------------------- DataLoader 构建 --------------------
def encode_batch(tokenizer, texts, max_len):
    enc = tokenizer.batch_encode_plus(
        texts, add_special_tokens=True,
        padding='max_length', truncation=True, max_length=max_len,
        return_attention_mask=True, return_tensors=None
    )
    return enc["input_ids"], enc["attention_mask"]

def build_loaders(tokenizer, args,
                  train_texts, train_labels,
                  val_texts, val_labels,
                  test_texts, test_labels):
    train_ids, train_masks = encode_batch(tokenizer, train_texts, args.max_len)
    val_ids,   val_masks   = encode_batch(tokenizer, val_texts, args.max_len)
    test_ids,  test_masks  = encode_batch(tokenizer, test_texts, args.max_len)

    t_inputs  = torch.tensor(train_ids, dtype=torch.long)
    v_inputs  = torch.tensor(val_ids,   dtype=torch.long)
    te_inputs = torch.tensor(test_ids,  dtype=torch.long)

    t_masks  = torch.tensor(train_masks, dtype=torch.float32)
    v_masks  = torch.tensor(val_masks,   dtype=torch.float32)
    te_masks = torch.tensor(test_masks,  dtype=torch.float32)

    t_labels  = torch.tensor(train_labels, dtype=torch.long)
    v_labels  = torch.tensor(val_labels,   dtype=torch.long)
    te_labels = torch.tensor(test_labels,  dtype=torch.long)

    train_ds = TensorDataset(t_inputs, t_masks, t_labels)
    val_ds   = TensorDataset(v_inputs, v_masks, v_labels)
    test_ds  = TensorDataset(te_inputs, te_masks, te_labels)

    if args.use_weighted_sampler:
        cls_cnt = np.unique(train_labels, return_counts=True)[1]
        weight = 1.0 / cls_cnt
        sample_w = torch.tensor(weight[train_labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, len(sample_w))
        train_loader = DataLoader(train_ds, sampler=sampler, batch_size=args.batch_size, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=True)

    val_loader  = DataLoader(val_ds,  sampler=SequentialSampler(val_ds),  batch_size=args.batch_size, pin_memory=True)
    test_loader = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=args.batch_size, pin_memory=True)
    return train_loader, val_loader, test_loader

# -------------------- 训练 / 验证 / 测试 --------------------
class EarlyStopping:
    """简单早停：验证集 loss 在 min_delta 内无改进连续 patience 次就停止"""
    def __init__(self, patience=5, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False
    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

@torch.no_grad()
def eval_once(model, data_loader, device, name="Eval"):
    """不训练，快速评估一次（冷启动观测泄露风险）"""
    model.eval()
    tot_loss, tot_n = 0.0, 0
    probs, ys = [], []
    for b_ids, b_mask, b_y in data_loader:
        b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
        with amp.autocast('cuda'):
            loss, logits = model(b_ids, attention_mask=b_mask, labels=b_y)
        tot_loss += float(loss.item()) * len(b_y)
        tot_n += len(b_y)
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(p)
        ys.extend(b_y.detach().cpu().numpy().tolist())
    y = np.asarray(ys, dtype=int); p = np.asarray(probs, dtype=float)
    yhat = (p > 0.5).astype(int)
    auc = roc_auc_score(y, p)
    f1  = f1_score(y, yhat, zero_division=0)
    print(f"\n[{name}] Loss={tot_loss/max(1,tot_n):.6f}  AUC={auc:.4f}  F1={f1:.4f}  N={tot_n}")

def train_loop(model, args, train_loader, val_loader, device, result_dir):
    """混合精度 + 线性学习率调度 + 早停 + 保存最好模型"""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = amp.GradScaler('cuda')

    early = EarlyStopping(patience=args.patience, min_delta=0.005)
    best_loss = None
    train_hist, val_hist = [], []

    model_save_path = os.path.join(result_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_graphcodebert.bin')

    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\n======== Epoch {epoch}/{args.epochs} ========\nTraining...")
        t0 = time.time()
        model.train()
        tr_loss = 0.0; n_train = 0

        for step, batch in enumerate(train_loader, start=1):
            b_ids, b_mask, b_y = [t.to(device) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast('cuda'):
                loss, _ = model(b_ids, attention_mask=b_mask, labels=b_y)

            tr_loss += loss.item() * b_y.size(0)
            n_train += b_y.size(0)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 1000 == 0 or step == len(train_loader):
                elapsed = format_time(time.time() - t0)
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"  Batch {step:>5}/{len(train_loader):>5} | Elapsed {elapsed} | Peak CUDA {peak:.2f} GB")

        ep_tr = tr_loss / max(1, n_train)
        train_hist.append(ep_tr)
        print(f"  Average training loss: {ep_tr:.6f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")

        # ---------- 验证 ----------
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        val_loss = 0.0; n_val = 0
        val_scores = []

        with torch.no_grad():
            for b_ids, b_mask, b_y in val_loader:
                b_ids, b_mask, b_y = b_ids.to(device), b_mask.to(device), b_y.to(device)
                with amp.autocast('cuda'):
                    loss, logits = model(b_ids, attention_mask=b_mask, labels=b_y)
                val_loss += loss.item() * b_y.size(0)
                n_val += b_y.size(0)
                scores = torch.sigmoid(logits).detach().cpu().numpy().tolist()
                val_scores.extend(scores)

        ep_val = val_loss / max(1, n_val)
        val_hist.append(ep_val)
        print(f"  Validation Loss: {ep_val:.6f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")

        # 保存最好模型（以验证集 loss 为准）
        if best_loss is None or ep_val < best_loss:
            best_loss = ep_val
            ckpt = {
                "epoch": epoch,
                "val_loss": best_loss,
                "state_dict": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                "train_loss_hist": train_hist,
                "val_loss_hist": val_hist,
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(ckpt, model_save_path)
            print(f"  >>> Saved best model to {model_save_path}")

        # 验证集 Top-K
        y_val = val_loader.dataset.tensors[2].cpu().numpy()
        topk = topk_metrics(val_scores, y_val, ks=KS)
        for k, (pq, rq) in topk.items():
            print(f"  [Valid] PQ{k:>3}={pq*100:5.2f}% | RQ{k:>3}={rq*100:5.2f}%")

        # 早停
        early.step(ep_val)
        if early.stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nTotal training took {format_time(time.time() - t0_all)}")
    return model_save_path, train_hist, val_hist

@torch.no_grad()
def test_loop(model, test_loader, device):
    """测试阶段：只前向得到概率"""
    print("\nTesting...")
    model.eval()
    probs = []
    for b_ids, b_mask, _ in test_loader:
        b_ids, b_mask = b_ids.to(device), b_mask.to(device)
        with amp.autocast('cuda'):
            logits = model(b_ids, attention_mask=b_mask)
        p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(p)
    return np.asarray(probs, dtype=float)

def evaluate_overall(labels, probs, thr=0.5):
    """整体评估（混淆矩阵、ROC-AUC、Precision/Recall/F1、AP、Balanced Acc、Top-K）"""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    yhat = (p > thr).astype(int)

    CM = confusion_matrix(y, yhat, labels=[0,1])
    TN, FP, FN, TP = CM[0,0], CM[0,1], CM[1,0], CM[1,1]
    auc_prob = roc_auc_score(y, p)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)
    f1   = f1_score(y, yhat, zero_division=0)
    ap   = average_precision_score(y, p)
    bal  = balanced_accuracy_score(y, yhat)

    print("\nConfusion Matrix:\n", CM)
    print(f'TN={TN}  FP={FP}  FN={FN}  TP={TP}')
    print(f'ROC-AUC(prob): {auc_prob:.6f}')
    print(f'Precision: {prec:.6f}  Recall: {rec:.6f}  F1: {f1:.6f}')
    print(f'Average Precision (AP): {ap:.6f}')
    print(f'Balanced Accuracy: {bal:.6f}')

    tk = topk_metrics(p, y, ks=KS)
    for k, (pq, rq) in tk.items():
        print(f'PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%')

def per_cwe_metrics(test_cwes, labels, probs, thr, result_dir):
    """
    逐 CWE 计算 Accuracy/Precision/Recall/F1，并生成：
    - CSV：每个 CWE 的四项指标 + 样本数
    - PNG：2×2 子图（Accuracy / Precision / Recall / F1）
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    yhat = (p > thr).astype(int)

    def safe(y, yhat):
        acc = float((y == yhat).mean()) if len(y) else 0.0
        pr  = precision_score(y, yhat, zero_division=0)
        rc  = recall_score(y, yhat, zero_division=0)
        f1  = f1_score(y, yhat, zero_division=0)
        return acc, pr, rc, f1

    rows = []
    # Overall
    acc, pr, rc, f1 = safe(y, yhat)
    rows.append({"CWE": "Overall", "Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1, "N": len(y)})

    bucket = defaultdict(lambda: {"y": [], "yhat": []})
    for yy, yh, c in zip(y, yhat, test_cwes):
        bucket[c]["y"].append(yy)
        bucket[c]["yhat"].append(yh)

    def cwe_key(c):
        m = re.search(r"\d+", c)
        return 10**9 if not m else int(m.group())

    for cwe in sorted(bucket.keys(), key=cwe_key):
        yy  = np.asarray(bucket[cwe]["y"])
        yh  = np.asarray(bucket[cwe]["yhat"])
        acc, pr, rc, f1 = safe(yy, yh)
        rows.append({"CWE": cwe, "Accuracy": acc, "Precision": pr, "Recall": rc, "F1": f1, "N": len(yy)})

    df = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_path = os.path.join(result_dir, f"{ts}_perCWE_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nPer-CWE metrics saved -> {csv_path}")
    print(df)

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    mets = ["Accuracy", "Precision", "Recall", "F1"]
    titles = ["(a) Accuracy", "(b) Precision", "(c) Recall", "(d) F1-Score"]
    xticks = df["CWE"].tolist()
    x = np.arange(len(xticks))
    for ax, m, t in zip(axes.ravel(), mets, titles):
        ax.bar(x, df[m].values)
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(m)
        ax.set_title(t)
    png_path = os.path.join(result_dir, f"{ts}_perCWE_4panels.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved -> {png_path}")

def best_threshold_from_val(model, val_loader, device):
    """
    用验证集扫阈值，返回 F1 最优阈值（可选，但通常能让测试集 F1 更稳）
    """
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_y in val_loader:
            b_ids, b_mask = b_ids.to(device), b_mask.to(device)
            with amp.autocast('cuda'):
                logits = model(b_ids, attention_mask=b_mask)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(b_y.numpy().tolist())
    ps = np.array(probs); ys = np.array(labels)
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.1, 0.9, 33):
        f1 = f1_score(ys, (ps > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f"[Val] best threshold={best_t:.3f}, F1={best_f1:.4f}")
    return best_t

def dump_errors(paths, labels, probs, thr, out_csv):
    """导出误判样本（FP/FN）以便人工检查"""
    y = np.array(labels); p = np.array(probs); yhat = (p > thr).astype(int)
    rows = []
    for path, yy, pp, yh in zip(paths, y, p, yhat):
        if yy != yh:
            rows.append({"ErrType": "FP" if yh==1 else "FN", "Path": path, "Label": int(yy), "Prob": float(pp)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Misclassifications saved -> {out_csv} (n={len(df)})")

# -------------------- 主函数 --------------------
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)

    # 1) 读取数据（可选清洗、可选打乱）
    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"
    assert train_dir.exists() and val_dir.exists() and test_dir.exists(), "train/val/test 目录不存在"

    tr_ids, tr_y, tr_txt, tr_cwe, tr_paths = load_split_from_dir(
        train_dir,
        strip_comments=args.strip_comments,
        strip_strings=args.strip_strings,
        mask_sard_hints=args.mask_sard_hints,
        mask_sard_scope=args.mask_sard_scope,
        mask_all_identifiers=args.mask_all_identifiers,
        shuffle=args.shuffle, seed=args.seed
    )
    va_ids, va_y, va_txt, va_cwe, va_paths = load_split_from_dir(
        val_dir,
        strip_comments=args.strip_comments,
        strip_strings=args.strip_strings,
        mask_sard_hints=args.mask_sard_hints,
        mask_sard_scope=args.mask_sard_scope,
        mask_all_identifiers=args.mask_all_identifiers,
        shuffle=False, seed=args.seed
    )
    te_ids, te_y, te_txt, te_cwe, te_paths = load_split_from_dir(
        test_dir,
        strip_comments=args.strip_comments,
        strip_strings=args.strip_strings,
        mask_sard_hints=args.mask_sard_hints,
        mask_sard_scope=args.mask_sard_scope,
        mask_all_identifiers=args.mask_all_identifiers,
        shuffle=False, seed=args.seed
    )

    print(f"Train: {len(tr_y)} (pos={sum(tr_y)})")
    print(f"Val  : {len(va_y)} (pos={sum(va_y)})")
    print(f"Test : {len(te_y)} (pos={sum(te_y)})")

    # 2) 分词编码 + DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_loader, val_loader, test_loader = build_loaders(
        tokenizer, args, tr_txt, tr_y, va_txt, va_y, te_txt, te_y)

    # 3) 构建模型并初始评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphCodeBERTVD(model_name=args.model, random_init=args.random_init).to(device)
    print(f"[Info] GraphCodeBERT loaded from: {args.model}")

    # 冷启动评估（未训练前观测泄露风险）
    eval_once(model, val_loader,  device, name="Init Val")
    eval_once(model, test_loader, device, name="Init Test")

    # 4) 训练
    model_path, train_hist, val_hist = train_loop(
        model, args, train_loader, val_loader, device, args.result_dir
    )

    # 5) 加载“验证集最优”权重再测试
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # （可选）用验证集调最优阈值
    best_t = best_threshold_from_val(model, val_loader, device)
    thr = best_t if best_t is not None else args.threshold

    # 6) 测试与评估（总体 + 逐 CWE）
    probs = test_loop(model, test_loader, device)
    evaluate_overall(te_y, probs, thr=thr)
    per_cwe_metrics(te_cwe, te_y, probs, thr=thr, result_dir=args.result_dir)

    # 7) 保存逐样本结果 + 误判样本
    ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    df_out = pd.DataFrame({"FuncID": te_ids, "Prob": probs, "Label": te_y, "CWE": te_cwe, "Path": te_paths})
    out_csv = os.path.join(args.result_dir, f"{ts}_perSample_results.csv")
    df_out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nPer-sample results saved -> {out_csv}")

    dump_errors(
        te_paths, te_y, probs, thr=thr,
        out_csv=os.path.join(args.result_dir, f"{ts}_test_errors.csv")
    )

if __name__ == "__main__":
    main()

"""
--mask-all-identifiers --epochs 3
        CWE  Accuracy  Precision    Recall        F1     N
0   Overall  0.908110   0.848214  0.994114  0.915387  3058
1    CWE-74  0.862500   0.787129  0.993750  0.878453   320
2   CWE-190  0.943750   0.898876  1.000000  0.946746   320
3   CWE-191  0.853125   0.772947  1.000000  0.871935   320
4   CWE-369  0.950000   0.909091  1.000000  0.952381   320
5   CWE-400  0.906250   0.853261  0.981250  0.912791   320
6   CWE-404  0.800000   0.714286  1.000000  0.833333   320
7   CWE-573  0.968750   0.957317  0.981250  0.969136   320
8   CWE-665  0.946667   0.903614  1.000000  0.949367   300
9   CWE-670  0.929293   0.876106  1.000000  0.933962   198
10  CWE-704  0.931250   0.887640  0.987500  0.934911   320

--mask-sard-identifiers --epochs 1
Per-CWE metrics saved -> ./result\2025-09-25_16-01-49_perCWE_metrics.csv
        CWE  Accuracy  Precision    Recall        F1     N
0   Overall  0.977436   0.962025  0.994114  0.977806  3058
1    CWE-74  0.915625   0.855615  1.000000  0.922190   320
2   CWE-190  0.987500   0.975610  1.000000  0.987654   320
3   CWE-191  0.981250   0.963855  1.000000  0.981595   320
4   CWE-369  0.996875   1.000000  0.993750  0.996865   320
5   CWE-400  0.981250   0.981250  0.981250  0.981250   320
6   CWE-404  0.962500   0.940476  0.987500  0.963415   320
7   CWE-573  0.996875   0.993789  1.000000  0.996885   320
8   CWE-665  0.996667   0.993377  1.000000  0.996678   300
9   CWE-670  1.000000   1.000000  1.000000  1.000000   198
10  CWE-704  0.965625   0.951515  0.981250  0.966154   320
Figure saved -> ./result\2025-09-25_16-01-49_perCWE_4panels.png
"""
