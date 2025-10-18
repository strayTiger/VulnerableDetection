# -*- coding: utf-8 -*-
"""
"""
from __future__ import absolute_import, division, print_function
import os, argparse, glob, time, datetime, random, json, pickle, shutil, re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # 你自己的 GPU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
#from scipy.integrate import simps
from numpy import trapz
import torch
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler, WeightedRandomSampler)
from torch.nn import BCEWithLogitsLoss
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, balanced_accuracy_score,
                             precision_recall_curve, average_precision_score,
                             cohen_kappa_score)

# ==== 你的工具函数（SplitCharacters、ListToCSV 等） ====
from DataLoader import SplitCharacters, ListToCSV          # 保持不变

# ---------------------------------------------------------
#                  超参数与路径
# ---------------------------------------------------------
MAX_LEN        = 512           # RoBERTa 建议 512
BATCH_SIZE     = 16            
NUM_EPOCHS     = 2
LR             = 2e-5
max_grad_norm  = 1.0

result_dir = './result'
model_dir      = r'./roberta_model'
model_save_path = os.path.join(result_dir ,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_roberta.bin')

# ---------------------------------------------------------
#                1. 数据读取与预处理
# ---------------------------------------------------------
def generateIdCodeLabels(json_file_path):
    ids, labels, code_list = [], [], []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            ids.append(js['func_name'])
            labels.append(js['target'])
            code_tokens = []
            for tok in js['func'].split():
                code_tokens.append(SplitCharacters(tok))
            code_list.append(' '.join(code_tokens))
    return ids, labels, code_tokens if False else code_list   # 防 IDE 报错 ;)

train_id, train_y, train_code = generateIdCodeLabels('train.jsonl')
val_id,   val_y,   val_code   = generateIdCodeLabels('valid.jsonl')
test_id,  test_y,  test_code  = generateIdCodeLabels('test.jsonl')

print(f"Train  set : {len(train_y)}  | positives = {np.sum(train_y)}")
print(f"Valid  set : {len(val_y)}    | positives = {np.sum(val_y)}")
print(f"Test   set : {len(test_y)}    | positives = {np.sum(test_y)}")

# ---------------------------------------------------------
#                 2. Tokenizer / Padding
# ---------------------------------------------------------
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained(model_dir)   # 可换 codebert-base

def code_tokenize_pad(code_list, max_len=MAX_LEN):
    # RoBERTa tokenizer 会自己加 <s>...</s> 特殊符号
    encodings = [tokenizer.encode(text,
                                  add_special_tokens=True,
                                  truncation=True,
                                  max_length=max_len)
                 for text in code_list]
    padded = pad_sequences(encodings, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(tok_id > 0) for tok_id in seq] for seq in padded]
    return padded, attention_masks

train_inputs, train_masks = code_tokenize_pad(train_code)
val_inputs,   val_masks   = code_tokenize_pad(val_code)
test_inputs,  test_masks  = code_tokenize_pad(test_code)

# 转 torch.Tensor
train_inputs = torch.tensor(train_inputs)
val_inputs   = torch.tensor(val_inputs)
test_inputs  = torch.tensor(test_inputs)

train_masks  = torch.tensor(train_masks)
val_masks    = torch.tensor(val_masks)
test_masks   = torch.tensor(test_masks)

train_labels = torch.tensor(train_y)
val_labels   = torch.tensor(val_y)
test_labels  = torch.tensor(test_y)

# 处理类别不平衡——按样本权重随机采样
class_sample_count = np.unique(train_y, return_counts=True)[1]
weight = 1. / class_sample_count
samples_weight = torch.from_numpy(weight[train_y]).double()
weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_data = TensorDataset(train_inputs, train_masks, train_labels)
val_data   = TensorDataset(val_inputs,   val_masks,   val_labels)
test_data  = TensorDataset(test_inputs,  test_masks,  test_labels)

train_loader = DataLoader(train_data, sampler=weighted_sampler,
                          batch_size=BATCH_SIZE)
val_loader   = DataLoader(val_data,   sampler=SequentialSampler(val_data),
                          batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_data,  sampler=SequentialSampler(test_data),
                          batch_size=BATCH_SIZE)

# ---------------------------------------------------------
#                 3. 定义模型
# ---------------------------------------------------------
class Model(torch.nn.Module):
    """
    RoBERTa + mean-pooling + Linear → 1 logit (vuln vs non-vuln)
    """
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_dir)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 1)
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               return_dict=False)  # last_hidden_state, pooled
        hidden = outputs[0]                           # [B, L, H]
        pooled = hidden.mean(dim=1)                  # mean-pool
        logits = self.classifier(pooled).squeeze(-1) # [B]

        if labels is not None:
            loss = BCEWithLogitsLoss()(logits, labels.float())
            return loss
        return logits            # logits (not sigmoid)

    def freeze_roberta(self):
        for p in self.roberta.parameters():
            p.requires_grad = False
    def unfreeze_roberta(self):
        for p in self.roberta.parameters():
            p.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Model().to(device)
print("Loaded RoBERTa with", sum(p.numel() for p in model.parameters())/1e6, "M parameters")

# ---------------------------------------------------------
#                 4. 优化器
# ---------------------------------------------------------
from transformers import AdamW
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

# ---------------------------------------------------------
#           5. 训练 / 验证 / 保存 checkpoint
# ---------------------------------------------------------
def format_time(seconds): return str(datetime.timedelta(seconds=int(round(seconds))))

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

def save_model(epoch, val_loss_hist, train_loss_hist):
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'val_loss_hist':val_loss_hist,
        'train_loss_hist':train_loss_hist
    }, model_save_path)
    print(f"Saved checkpoint at epoch {epoch} to {model_save_path}")

train_loss_hist, val_loss_hist, stats = [], [], []
t0_global = time.time()
best_val = None

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\n======== Epoch {epoch}/{NUM_EPOCHS} ========")
    # ----- TRAIN -----
    model.train()
    tr_loss, tr_samples = 0., 0
    t0 = time.time()
    for step, batch in enumerate(train_loader, 1):
        batch = tuple(t.to(device) for t in batch)
        input_ids, masks, labels = batch
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask=masks, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        tr_loss += loss.item()
        tr_samples += labels.size(0)
        if step % 40 == 0:
            print(f"  step {step}/{len(train_loader)} | "f"avg loss {tr_loss/tr_samples:.4f}")
            elapsed = format_time(time.time() - t0)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"训练峰值显存: {peak_mem:.2f} GB  Batch {step}/{len(train_loader)} | Elapsed {elapsed}")
            torch.cuda.reset_peak_memory_stats()
    avg_tr = tr_loss / tr_samples
    train_loss_hist.append(avg_tr)
    print(f"Train loss: {avg_tr:.4f} | time {format_time(time.time()-t0)}")

    # ----- VALID -----
    model.eval()
    val_loss, val_samples = 0., 0
    t0=time.time()
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, labels = batch
            loss = model(input_ids, attention_mask=masks, labels=labels)
            val_loss += loss.item()
            val_samples += labels.size(0)

    elapsed = format_time(time.time() - t0)     
    print(f" Valid elapsed {elapsed}")
    avg_val = val_loss / val_samples
    val_loss_hist.append(avg_val)
    print(f"Valid loss: {avg_val:.4f}")

    # save best
    if best_val is None or avg_val < best_val:
        best_val = avg_val
        save_model(epoch, val_loss_hist, train_loss_hist)

    stats.append({'epoch':epoch, 'train_loss':avg_tr, 'val_loss':avg_val,
                  'train_time':format_time(time.time()-t0)})

print("\nTraining done in", format_time(time.time()-t0))

# ---------------------------------------------------------
#                 6. 测试 / 评估
# ---------------------------------------------------------
def predict(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, masks, labels = batch
            logits = model(input_ids, attention_mask=masks)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs.tolist())
    return np.asarray(preds)
t0=time.time()
pred_probs = predict(model, test_loader)
elapsed = format_time(time.time() - t0)     
print(f" test elapsed {elapsed}")
auc  = roc_auc_score(test_labels, pred_probs)
print(f"\n=== Test ROC-AUC: {auc:.4f} ===")

# -- 综合评估（含 Top-k）
def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score > threshold).astype(int)
    CM = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("Confusion Matrix:\n", CM)
    print(classification_report(y_true, y_pred, target_names=["Non-vuln","Vuln"]))

    print("AUC :", roc_auc_score(y_true, y_score))
    print("F1  :", f1_score(y_true, y_pred))
    print("BalAcc :", balanced_accuracy_score(y_true, y_pred))
    pq_rq = topk_metrics(y_score, y_true, ks=[50,100,150,200,250,300,350,400])
    for k,(pq,rq) in pq_rq.items():
        print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

evaluate(test_labels, pred_probs)

# ---------------------------------------------------------
#                 7. 把结果写 CSV + 学习曲线
# ---------------------------------------------------------
ListToCSV(pd.DataFrame(list(zip(test_id, pred_probs, test_y)),
                       columns=['Func_id','prob','Label']),
          os.path.join('.', f'RoBERTa_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_result.csv'))

# 学习曲线
pd.DataFrame(stats).set_index('epoch')[['train_loss','val_loss']].plot(
        figsize=(10,6), style=['b-o','g-o'])
plt.title("Training vs Validation Loss (RoBERTa)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
plt.tight_layout()
plt.savefig(f'RoBERTa_learning_curve_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pdf')
plt.show()
