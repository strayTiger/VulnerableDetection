# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import os, time, datetime, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"           
import numpy as np
import pandas as pd
import torch, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler, WeightedRandomSampler)
from torch.nn import BCEWithLogitsLoss
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, f1_score, balanced_accuracy_score,
                             precision_recall_curve, average_precision_score,
                             cohen_kappa_score)
from tqdm import tqdm, trange
from transformers import (GPT2Tokenizer, GPT2Model, AdamW)

from DataLoader import SplitCharacters, ListToCSV

# ----------------- 超参数 ------------------
MAX_LEN       = 512
BATCH_SIZE    = 8
NUM_EPOCHS    = 10
LR            = 2e-5
max_grad_norm = 1.0

result_dir = './result'
model_dir     = r'./gpt2_model'
model_save_path = os.path.join(result_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_gpt2.bin')

# ------------ 1. 读取 .jsonl 数据 ------------
def load_jsonl(path):
    ids, labels, codes = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            ids.append(js['func_name'])
            labels.append(js['target'])
            tokens = [SplitCharacters(tok) for tok in js['func'].split()]
            codes.append(' '.join(tokens))
    return ids, labels, codes

train_id, train_y, train_code = load_jsonl('train.jsonl')
val_id,   val_y,   val_code   = load_jsonl('valid.jsonl')
test_id,  test_y,  test_code  = load_jsonl('test.jsonl')

print(f"Train/Val/Test sizes: {len(train_y)}, {len(val_y)}, {len(test_y)}")

# ------------ 2. Tokenizer & Padding ------------
tokenizer = GPT2Tokenizer.from_pretrained(model_dir, cache_dir=None)
# GPT-2 没有 pad；这里把 eos_token 用作 pad 并扩展词表
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def encode_pad(texts, max_len=MAX_LEN):
    encodings = [tokenizer.encode(t,
                                  add_special_tokens=True,
                                  truncation=True,
                                  max_length=max_len)
                 for t in texts]
    padded = pad_sequences(encodings, maxlen=max_len,
                           dtype='long', truncating='post', padding='post',
                           value=tokenizer.pad_token_id)
    masks = [[float(tok_id != tokenizer.pad_token_id) for tok_id in seq]
             for seq in padded]
    return padded, masks

train_inp, train_msk = encode_pad(train_code)
val_inp,   val_msk   = encode_pad(val_code)
test_inp,  test_msk  = encode_pad(test_code)

train_ds = TensorDataset(torch.tensor(train_inp),
                         torch.tensor(train_msk),
                         torch.tensor(train_y))
val_ds   = TensorDataset(torch.tensor(val_inp),
                         torch.tensor(val_msk),
                         torch.tensor(val_y))
test_ds  = TensorDataset(torch.tensor(test_inp),
                         torch.tensor(test_msk),
                         torch.tensor(test_y))

# 类别失衡采样
class_cnt = np.unique(train_y, return_counts=True)[1]
weights = torch.from_numpy((1. / class_cnt)[train_y]).double()
train_loader = DataLoader(train_ds,
                          sampler=WeightedRandomSampler(weights, len(weights)),
                          batch_size=BATCH_SIZE)
val_loader   = DataLoader(val_ds,
                          sampler=SequentialSampler(val_ds),
                          batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,
                          sampler=SequentialSampler(test_ds),
                          batch_size=BATCH_SIZE)

# ------------ 3. 定义模型 ------------
class GPT2Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_dir)
        self.gpt2.resize_token_embeddings(len(tokenizer))
        self.classifier = torch.nn.Linear(self.gpt2.config.n_embd, 1)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=False)    # last_hidden_state, past
        hidden = outputs[0]                       # [B, L, H]
        pooled = hidden.mean(dim=1)               # mean pool
        logits = self.classifier(pooled).squeeze(-1)  # [B]

        if labels is not None:
            loss = BCEWithLogitsLoss()(logits, labels.float())
            return loss
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GPT2Classifier().to(device)

# ------------ 4. 优化器 ------------
no_decay = ['bias', 'ln_f', 'ln_1', 'ln_2']
optimizer_grouped_parameters = [
    {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay':0.01},
    {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay':0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

# ------------ 5. 训练 + 验证 ------------
def format_sec(s): return str(datetime.timedelta(seconds=int(round(s))))

train_loss_hist, val_loss_hist, stats = [], [], []
best_val = None

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
    # ---- Train ----
    model.train()
    tr_loss, tr_samp = 0., 0
    t0=time.time()
    for step, (ids, msk, lbl) in enumerate(train_loader, 1):
        ids, msk, lbl = ids.to(device), msk.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = model(ids, attention_mask=msk, labels=lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        tr_loss += loss.item()
        tr_samp += lbl.size(0)
        if step % 40 == 0:
            print(f"  step {step:>4}/{len(train_loader)}  "
                  f"avg loss {tr_loss/tr_samp:.4f}")
            elapsed = format_sec(time.time() - t0)
            peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"训练峰值显存: {peak_mem:.2f} GB  Batch {step}/{len(train_loader)} | Elapsed {elapsed}")
            torch.cuda.reset_peak_memory_stats()
    train_loss_hist.append(tr_loss / tr_samp)

    # ---- Valid ----
    model.eval()
    val_loss, val_samp = 0., 0
    t0=time.time()
    with torch.no_grad():
        for ids, msk, lbl in val_loader:
            ids, msk, lbl = ids.to(device), msk.to(device), lbl.to(device)
            loss = model(ids, attention_mask=msk, labels=lbl)
            val_loss += loss.item()
            val_samp += lbl.size(0)

    elapsed = format_sec(time.time() - t0)     
    print(f" Valid elapsed {elapsed}")
    avg_val = val_loss / val_samp
    val_loss_hist.append(avg_val)
    print(f" Valid loss {avg_val:.4f}")

    # save best
    if best_val is None or avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), model_save_path)
        print(" Checkpoint saved")

    stats.append({'epoch':epoch,
                  'train_loss':train_loss_hist[-1],
                  'val_loss':avg_val})

# ------------ 6. 测试 & 评价 ------------
def topk_metrics(pred_probs, labels, ks=(50,100,150,200,250,300,350,400)):
    scores = np.asarray(pred_probs).ravel()
    labels = np.asarray(labels).ravel()
    order  = np.argsort(-scores)
    sorted_lbl = labels[order]
    total_pos = labels.sum()
    out = {}
    for k in ks:
        if k > len(labels): continue
        tp = sorted_lbl[:k].sum()
        out[k] = (tp/k, tp/total_pos if total_pos else 0.)
    return out

def predict(dl):
    model.eval()
    preds = []
    with torch.no_grad():
        for ids, msk, _ in dl:
            ids, msk = ids.to(device), msk.to(device)
            logits = model(ids, attention_mask=msk)
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(preds)

t0=time.time()
pred_probs = predict(test_loader)
elapsed = format_sec(time.time() - t0) 
print(f" test elapsed {elapsed}")

auc = roc_auc_score(test_y, pred_probs)
print(f"\n=== Test ROC-AUC: {auc:.4f} ===")

# 0.5 阈值
y_pred = (pred_probs > 0.5).astype(int)
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred, target_names=['Non-vuln','Vuln']))
print("Balanced Acc :", balanced_accuracy_score(test_y, y_pred))
print("F1 :", f1_score(test_y, y_pred))
print("Kappa :", cohen_kappa_score(test_y, y_pred))

# Top-k
topk = topk_metrics(pred_probs, test_y)
for k,(pq,rq) in topk.items():
    print(f"PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%")

# ------------ 7. 结果保存 ------------
ListToCSV(pd.DataFrame(list(zip(test_id, pred_probs, test_y)),
                       columns=['Func_id','prob','Label']),
          f'GPT2_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_result.csv')

pd.DataFrame(stats).set_index('epoch')[['train_loss','val_loss']].plot(
        figsize=(10,6), style=['b-o','g-o'])
plt.title("GPT-2 Training vs Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
plt.tight_layout()
plt.savefig(f'GPT2_learning_curve_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pdf')
plt.show()
