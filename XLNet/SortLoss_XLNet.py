# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os, time, datetime, json, random, contextlib
import numpy as np
import pandas as pd
from tqdm import trange

# ========= 环境 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Torch / Transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, Sampler
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch import amp
from transformers import XLNetModel, XLNetTokenizer, get_linear_schedule_with_warmup

# 评估
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)

# 你的工具
from DataLoader import SplitCharacters, ListToCSV

# ========= 配置 =========
MAX_LEN = 768
BATCH_SIZE = 28           
POS_PER_BATCH = 6         # 每个 batch 固定正样本数（3~8 之间可调）
REPEAT_PER_EPOCH = 8      # BalancedBatchSampler 每个 epoch 复用不同子采样轮数
NUM_EPOCHS = 20

MAX_GRAD_NORM = 1.0
PATH_MODEL = './xlnet_model'
RESULT_DIR = './result'
os.makedirs(RESULT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_xlnet.bin')

# 排序损失
USE_RANKING_LOSS = True
RANK_MARGIN = 0.5
RANK_WEIGHT = 0.25         # 目标权重（训练前两轮线性升到这里）
HNM_TOPK_NEG = 22         # 每批仅用 top-K 难负样本（<= BATCH_SIZE - POS_PER_BATCH）
TOP_POS_FOR_RANK = 3      # 每批仅用“最像正”的前 T 个正样本

# AMP
USE_AMP = True
AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

# PQ@K 与聚合权重
KS = (10, 20, 30, 40, 50, 100, 150, 200)
PQ_WEIGHTS = {10:1.3, 20:1.2, 30:1.0, 40:1.0, 50:1.0, 100:1.2, 150:1.3, 200:1.4}   # 作用不大，不同的PQ_WEIGHTS跑出来的结果一致

# ========= 工具函数 =========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def topk_metrics(pred_scores, labels, ks=KS):
    scores = np.asarray(pred_scores).flatten()
    labels = np.asarray(labels).flatten()
    total_pos = int(labels.sum())
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    res = {}
    for k in ks:
        if k > len(sorted_labels): continue
        tp_k = int(sorted_labels[:k].sum())
        pq_k = tp_k / k
        rq_k = tp_k / total_pos if total_pos else 0.0
        res[k] = (pq_k, rq_k)
    return res

def pq_aggregate(pred_scores, labels, ks=KS, weights=PQ_WEIGHTS):
    res = topk_metrics(pred_scores, labels, ks)
    score = 0.0
    for k, (pq, _) in res.items():
        score += weights.get(k, 1.0) * pq
    return float(score)

# ========= 读取数据 =========
def generateIdCodeLabels(json_file_path):
    id_list, label_list, func_body_list = [], [], []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            id_list.append(js['func_name'])
            label_list.append(js['target'])
            tokens = [SplitCharacters(t) for t in js['func'].split()]
            func_body_list.append(' '.join(tokens))
    return id_list, label_list, func_body_list

set_seed(42)
data_dir = './'
train_id_list, train_label_list, train_func_list = generateIdCodeLabels(os.path.join(data_dir, "libpng_train.jsonl"))
validation_id_list, validation_label_list, validation_func_list = generateIdCodeLabels(os.path.join(data_dir, "libpng_val.jsonl"))
test_id_list, test_label_list, test_func_list = generateIdCodeLabels(os.path.join(data_dir, "libpng_test.jsonl"))

print(f"The length of the training set is: {len(train_label_list)}, there are {np.count_nonzero(train_label_list)} vulnerable samples.")
print(f"The length of the validation set is: {len(validation_label_list)}, there are {np.count_nonzero(validation_label_list)} vulnerable samples.")
print(f"The length of the test set is: {len(test_label_list)}, there are {np.count_nonzero(test_label_list)} vulnerable samples.")

# ========= 分词（XLNet 自动加特殊符号） =========
tokenizer = XLNetTokenizer.from_pretrained(PATH_MODEL)

def codeTokenization_Padding(code_list):
    enc = tokenizer(
        list(code_list),
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_tensors=None
    )
    return enc['input_ids'], enc['attention_mask']

train_func_pad, train_masks = codeTokenization_Padding(train_func_list)
valid_func_pad, validation_masks = codeTokenization_Padding(validation_func_list)
test_func_pad, test_masks = codeTokenization_Padding(test_func_list)

# tensors
train_inputs = torch.tensor(train_func_pad, dtype=torch.long)
validation_inputs = torch.tensor(valid_func_pad, dtype=torch.long)
test_inputs = torch.tensor(test_func_pad, dtype=torch.long)

train_labels = torch.tensor(train_label_list, dtype=torch.long)
validation_labels = torch.tensor(validation_label_list, dtype=torch.long)
test_labels = torch.tensor(test_label_list, dtype=torch.long)

train_masks = torch.tensor(train_masks, dtype=torch.float32)
validation_masks = torch.tensor(validation_masks, dtype=torch.float32)
test_masks = torch.tensor(test_masks, dtype=torch.float32)

# ========= BalancedBatchSampler =========
class BalancedBatchSampler(Sampler):
    """
    每个 batch 固定 P 个正样本 + (B-P) 个负样本；一个 epoch 内重复 repeat 轮，覆盖更多负样本。
    """
    def __init__(self, labels, pos_per_batch, batch_size, drop_last=True, seed=42, repeat=6):
        labels = np.asarray(labels)
        self.pos_idx = np.where(labels == 1)[0].tolist()
        self.neg_idx = np.where(labels == 0)[0].tolist()
        assert len(self.pos_idx) > 0 and len(self.neg_idx) > 0, "正/负样本不能为空"
        self.P = int(pos_per_batch); self.B = int(batch_size); self.N = self.B - self.P
        assert 1 <= self.P < self.B, "POS_PER_BATCH 必须在 [1, BATCH_SIZE-1]"
        self.drop_last = drop_last
        self.repeat = int(repeat)
        self.rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.repeat):  # 重复多轮
            self.rng.shuffle(self.pos_idx)     # 打乱样本索引
            self.rng.shuffle(self.neg_idx)
            pos_ptr = neg_ptr = 0
            num_pos, num_neg = len(self.pos_idx), len(self.neg_idx)
            max_full_batches = min(num_pos // self.P, num_neg // self.N)
            for _ in range(max_full_batches):
                if pos_ptr + self.P > num_pos:
                    self.rng.shuffle(self.pos_idx); pos_ptr = 0
                batch_pos = self.pos_idx[pos_ptr:pos_ptr + self.P]; pos_ptr += self.P

                if neg_ptr + self.N > num_neg:    # 当负样本用完时，采样器会再次打乱负样本列表并重置指针
                    self.rng.shuffle(self.neg_idx); neg_ptr = 0
                batch_neg = self.neg_idx[neg_ptr:neg_ptr + self.N]; neg_ptr += self.N    #从当前打乱后的列表里顺序抽取N个负样本，然后移动指针，确保下次从新的位置开始

                batch = batch_pos + batch_neg
                self.rng.shuffle(batch)
                yield batch

    def __len__(self):
        base = min(len(self.pos_idx) // self.P, len(self.neg_idx) // self.N)
        return base * self.repeat

# 数据集与 DataLoader
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
valid_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
test_dataset  = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=BalancedBatchSampler(train_label_list, POS_PER_BATCH, BATCH_SIZE, drop_last=True, repeat=REPEAT_PER_EPOCH), 
    pin_memory=True
)
valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=BATCH_SIZE, pin_memory=True)   # 顺序采样
test_loader  = DataLoader(test_dataset,  sampler=SequentialSampler(test_dataset),  batch_size=BATCH_SIZE, pin_memory=True)

print("Steps per epoch (train):", len(train_loader))

# ========= 模型（最后有效 token 池化 + 排序损失 + 难负样本挖掘） =========
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(PATH_MODEL)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        self.rank_weight = RANK_WEIGHT  # 可动态调整

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        h = out[0]  # [B, L, H]

        # 取每条样本最后一个有效 token（XLNet 的 <cls> 在尾部）
        last_idx = attention_mask.long().sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        pooled = h[torch.arange(h.size(0), device=h.device), last_idx, :]  # [B, H]
        pooled = self.dropout(pooled)
        logit  = self.classifier(pooled.float()).squeeze(-1)               # [B]

        if labels is None:
            return logit

        # BCE 主损失
        bce_loss = BCEWithLogitsLoss()(logit.float(), labels.float())

        # 只用“最像正”的正样本 × “最难负样本”，并用 softplus 平滑 hinge
        if USE_RANKING_LOSS:
            y = labels.view(-1).float()
            pos_logits = logit[y == 1]
            neg_logits = logit[y == 0]
            if pos_logits.numel() and neg_logits.numel():
                t = min(TOP_POS_FOR_RANK, pos_logits.numel())
                tpos, _ = torch.topk(pos_logits, t, largest=True, sorted=False)

                k = min(HNM_TOPK_NEG, neg_logits.numel())
                hneg, _ = torch.topk(neg_logits, k, largest=True, sorted=False)

                diff = RANK_MARGIN - tpos.unsqueeze(1) + hneg.unsqueeze(0)  # [t, k]
                rank_loss = torch.nn.functional.softplus(diff).mean()       # 平滑

                return bce_loss + self.rank_weight * rank_loss

        return bce_loss

# 设备与模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Model().to(device)

# 先验 bias 初始化（按训练集正例率，避免初期全判负）
prior = (np.count_nonzero(train_label_list) + 1e-8) / (len(train_label_list) + 1e-8)
bias_init = float(np.log(prior / (1.0 - prior)))
with torch.no_grad():
    model.classifier.bias.fill_(bias_init)

# 优化器 & 调度器（头部更大学习率 + 10% warmup）
optimizer = AdamW([
    {'params': model.xlnet.parameters(),      'lr': 2e-5, 'weight_decay': 0.01},
    {'params': model.classifier.parameters(), 'lr': 6e-4, 'weight_decay': 0.0},
], eps=1e-6)

total_steps = len(train_loader) * NUM_EPOCHS
num_warmup_steps = int(0.10 * total_steps)  # 10% warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

use_amp_runtime = USE_AMP and torch.cuda.is_available()
scaler = amp.GradScaler('cuda') if use_amp_runtime else None

# 早停（监控“加权 PQ 聚合”，越大越好）
class EarlyStoppingMax:
    def __init__(self, patience=5, min_delta=0.003):  
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best = -1e9; self.early_stop = False
    def __call__(self, metric_value):
        if metric_value < self.best + self.min_delta:
            self.counter += 1
            print(f"Validation PQ metric no significant improvement for {self.counter} epoch(s).")
            if self.counter >= self.patience:
                self.early_stop = True; print("Early stopping triggered.")
        else:
            self.best = metric_value; self.counter = 0

def save_model(model, save_path, epoch, monitor_value, tr_hist, va_hist, opt_state):
    mdl = model.module if hasattr(model, 'module') else model
    torch.save({
        'epochs': epoch, 'monitor': monitor_value,
        'state_dict': mdl.state_dict(),
        'train_loss_hist': tr_hist, 'valid_loss_hist': va_hist,
        'optimizer_state_dict': opt_state
    }, save_path)
    print(f"Saving model at epoch {epoch} with Val-PQ metric = {monitor_value:.6f}")

def get_probs(model, loader, device):
    model.eval()
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            b_input_ids, b_input_mask, _ = [t.to(device) for t in batch]
            logits = model(b_input_ids, attention_mask=b_input_mask)
            probs  = torch.sigmoid(logits)
            probs_all.extend(probs.detach().cpu().numpy().tolist())
    return np.array(probs_all, dtype=np.float64)

def train(model, num_epochs, optimizer, train_loader, valid_loader, model_save_path, device=device, patience=5):
    training_stats, tr_hist, va_hist = [], [], []
    stopper = EarlyStoppingMax(patience=patience, min_delta=0.003)
    best_metric = -1e9
    total_t0 = time.time()

    for epoch in trange(num_epochs, desc="Epoch"):
        print(f"\n======== Epoch {epoch} / {num_epochs} ========")
        print("Training...")
        t0 = time.time(); model.train()
        tr_loss, n_train = 0.0, 0

        # rank_weight 热启动（前 warm_ep 轮线性升到目标值）
        warm_ep = 2
        curr = RANK_WEIGHT * min(1.0, (epoch + 1) / float(warm_ep))
        if hasattr(model, 'rank_weight'):
            model.rank_weight = curr
        print(f"  [Train] rank_weight now = {curr:.3f}")

        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            if use_amp_runtime:
                with amp.autocast('cuda', dtype=AMP_DTYPE):
                    loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                tr_loss += loss.item() * b_labels.size(0); n_train += b_labels.size(0)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
            else:
                loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                tr_loss += loss.item() * b_labels.size(0); n_train += b_labels.size(0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()

        epoch_tr = tr_loss / max(1, n_train); tr_hist.append(epoch_tr)
        print(f"  Avg training loss: {epoch_tr:.6f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")

        # ---- 验证（禁用 autocast，保证排序稳定） ----
        print("\nRunning Validation...")
        t0 = time.time(); model.eval()
        ev_loss, n_valid = 0.0, 0
        valid_probs = []

        with torch.no_grad():
            for batch in valid_loader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                ev_loss += loss.item() * b_labels.size(0); n_valid += b_labels.size(0)
                logits = model(b_input_ids, attention_mask=b_input_mask)
                valid_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())

        epoch_val_loss = ev_loss / max(1, n_valid); va_hist.append(epoch_val_loss)
        print(f"  Validation Loss: {epoch_val_loss:.6f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")

        valid_probs = np.array(valid_probs, dtype=np.float64)
        print(f"  [Valid] Predicted Positive Rate @0.5: {float((valid_probs>0.5).mean()*100.0):.2f}%")

        v_topk = topk_metrics(valid_probs, validation_labels.numpy(), ks=KS)
        for k, (pq, rq) in v_topk.items():
            print(f'  [Valid] PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%')

        monitor = pq_aggregate(valid_probs, validation_labels.numpy(), ks=KS, weights=PQ_WEIGHTS)
        print(f"  [Valid] Aggregated PQ score = {monitor:.6f}")

        training_stats.append({'epoch': epoch, 'Training Loss': epoch_tr, 'Valid. Loss': epoch_val_loss, 'Valid PQ agg': monitor})

        if monitor > best_metric + 1e-6:
            best_metric = monitor
            save_model(model, MODEL_SAVE_PATH, epoch, best_metric, tr_hist, va_hist, optimizer.state_dict())

        stopper(monitor)
        if stopper.early_stop:
            print(f"\nEarly stopped at epoch {epoch}.")
            break

    print(f"\nTotal training took {format_time(time.time() - total_t0)}")
    return model, tr_hist, va_hist, training_stats

# ========= 训练 =========
model, train_loss_set, valid_loss_set, training_stats = train(
    model=model,
    num_epochs=NUM_EPOCHS,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    model_save_path=MODEL_SAVE_PATH,
    device=device,
    patience=5
)

df_stats = pd.DataFrame(training_stats).set_index('epoch')
print(df_stats)

# ========= 测试与评估（含 PQ@K） =========
def Evaluate(labels, pred_scores, p=0.5):
    y = np.asarray(labels).astype(int); s = np.asarray(pred_scores).astype(float)
    yhat = (s > p).astype(int)

    CM = confusion_matrix(y, yhat, labels=[0,1])
    TN, FP, FN, TP = CM[0,0], CM[0,1], CM[1,0], CM[1,1]
    auc_prob = roc_auc_score(y, s)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)
    f1   = f1_score(y, yhat, zero_division=0)
    ap   = average_precision_score(y, s)
    bal_acc = balanced_accuracy_score(y, yhat)

    print('Confusion Matrix:\n', CM)
    print(f'TN={TN}  FP={FP}  FN={FN}  TP={TP}')
    print(f'ROC-AUC(prob): {auc_prob:.6f}')
    print(f'Precision: {prec:.6f}  Recall: {rec:.6f}  F1: {f1:.6f}  AP: {ap:.6f}  BalancedAcc: {bal_acc:.6f}')
    print(f'[Test] Predicted Positive Rate @0.5: {float((s>0.5).mean()*100.0):.2f}%')
    print("\nClassification report:")
    print(classification_report(y, yhat, target_names=["Non-vulnerable","Vulnerable"], digits=4))

    t_res = topk_metrics(s, y, ks=KS)
    for k, (pq, rq) in t_res.items():
        print(f'PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%')

# 方案 A：测试前加载“验证集最优”checkpoint
print("\nLoading BEST checkpoint for testing...")
assert os.path.isfile(MODEL_SAVE_PATH), f"Checkpoint not found: {MODEL_SAVE_PATH}"
ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)

# 加载保存的“最优轮次”权重（注意：save_model 里已经处理了 model.module 的情况）
model.load_state_dict(ckpt["state_dict"])
print("Loaded best raw weights from:", MODEL_SAVE_PATH)

# 正式测试（不再使用 EMA 的 apply_shadow/restore）
print("\nTesting (best checkpoint)...")
test_probs = get_probs(model, test_loader, device)
Evaluate(test_labels.numpy(), test_probs, p=0.5)

# 导出逐样本
zippedlist = list(zip(test_id_list, test_probs.tolist(), test_label_list))
result_set = pd.DataFrame(zippedlist, columns=['Func_id', 'prob', 'Label'])
ListToCSV(result_set, os.path.join('.', 'XLNet' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_result.csv'))
print("\nPer-sample results saved.")



"""

"""