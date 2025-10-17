# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time, datetime, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler, TensorDataset
from tqdm import trange
from sklearn.metrics import (precision_score, recall_score, precision_recall_curve, f1_score,
                             roc_auc_score, balanced_accuracy_score, average_precision_score,
                             cohen_kappa_score, confusion_matrix, classification_report)
from numpy import trapz
from transformers import XLNetModel, XLNetTokenizer
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from DataLoader import SplitCharacters, ListToCSV

# =========================
# 全局配置
# =========================
MAX_LEN = 768
BATCH_SIZE = 16
max_grad_norm = 1.0
num_epochs = 15
path_model = './xlnet_model'
data_dir = r'./'
model_save_path = r'./result/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_xlnet.bin'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def generateIdCodeLabels(json_file_path):
    id_list, label_list, func_body_list = [], [], []
    with open(json_file_path, encoding='utf-8') as f:
        for line in f:
            js = json.loads(line.strip())
            id_list.append(js['func_name'])
            label_list.append(js['target'])
            code = js['func'].split()
            code = ' '.join(SplitCharacters(tok) for tok in code)
            func_body_list.append(code)
    return id_list, label_list, func_body_list

def tokenize_batch(text_list, tokenizer, max_len=768):
    enc = tokenizer(
        text_list,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt',
        return_attention_mask=True
    )
    return enc['input_ids'], enc['attention_mask']

def topk_metrics(pred_probs, labels, ks=(10,20,30,40,50,100,150,200)):
    scores = np.asarray(pred_probs).flatten()
    labels = np.asarray(labels).flatten()
    total_pos = labels.sum()
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    out = {}
    for k in ks:
        if k > len(sorted_labels): continue
        tp_k = sorted_labels[:k].sum()
        out[k] = (tp_k/k, tp_k/total_pos if total_pos else 0)
    return out

# =========================
# 数据
# =========================
train_id_list, train_label_list, train_func_list = generateIdCodeLabels(data_dir + "ffmpeg_train.jsonl")
validation_id_list, validation_label_list, validation_func_list = generateIdCodeLabels(data_dir + "ffmpeg_val.jsonl")
test_id_list, test_label_list, test_func_list = generateIdCodeLabels(data_dir + "ffmpeg_test.jsonl")

print(f"Train: {len(train_label_list)} (pos={np.count_nonzero(train_label_list)})")
print(f"Valid: {len(validation_label_list)} (pos={np.count_nonzero(validation_label_list)})")
print(f"Test : {len(test_label_list)} (pos={np.count_nonzero(test_label_list)})")

tokenizer = XLNetTokenizer.from_pretrained(path_model, do_lower_case=True)
train_inputs, train_masks = tokenize_batch(train_func_list, tokenizer, MAX_LEN)
validation_inputs, validation_masks = tokenize_batch(validation_func_list, tokenizer, MAX_LEN)
test_inputs, test_masks = tokenize_batch(test_func_list, tokenizer, MAX_LEN)

train_labels = torch.tensor(train_label_list, dtype=torch.long)
validation_labels = torch.tensor(validation_label_list, dtype=torch.long)
test_labels = torch.tensor(test_label_list, dtype=torch.long)

# 类别不平衡采样
class_sample_count = np.unique(train_label_list, return_counts=True)[1]
weight = 1.0 / class_sample_count
samples_weight = torch.from_numpy(weight[train_labels.numpy()]).double()
weighted_train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(TensorDataset(train_inputs, train_masks, train_labels),
                          sampler=weighted_train_sampler, batch_size=BATCH_SIZE)
valid_loader = DataLoader(TensorDataset(validation_inputs, validation_masks, validation_labels),
                          sampler=SequentialSampler(TensorDataset(validation_inputs, validation_masks, validation_labels)),
                          batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(test_inputs, test_masks, test_labels),
                         sampler=SequentialSampler(TensorDataset(test_inputs, test_masks, test_labels)),
                         batch_size=BATCH_SIZE)

# =========================
# 模型
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained(path_model)
        self.classifier = nn.Linear(768, 1)
        torch.nn.init.xavier_normal_(self.classifier.weight)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.last_hidden_state.mean(dim=1)     # [B, H]
        logits = self.classifier(pooled).squeeze(-1)       # [B]
        if labels is not None:
            return BCEWithLogitsLoss()(logits, labels.float())
        return logits
    def freeze_xlnet_decoder(self):
        for p in self.xlnet.parameters(): p.requires_grad = False
    def unfreeze_xlnet_decoder(self):
        for p in self.xlnet.parameters(): p.requires_grad = True

model = Model().to(device)

# 优化器
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

# =========================
# 早停 + 保存/加载
# =========================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False
    def step(self, val_loss):
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return True
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
            return False

def save_model(model, save_path, epoch, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    model_to_save = model.module if hasattr(model, 'module') else model
    ckpt = {
        'epochs': epoch,
        'lowest_eval_loss': lowest_eval_loss,
        'state_dict': model_to_save.state_dict(),
        'train_loss_hist': train_loss_hist,
        'valid_loss_hist': valid_loss_hist,
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"Saved BEST model @ epoch {epoch} (val loss={lowest_eval_loss:.6f})")

def load_model(save_path):
    ckpt = torch.load(save_path, map_location='cpu')
    m = Model()
    m.load_state_dict(ckpt['state_dict'])
    return m, ckpt['epochs'], ckpt['lowest_eval_loss'], ckpt['train_loss_hist'], ckpt['valid_loss_hist']

# =========================
# 训练（含早停 & 保存最优）
# =========================
def train(model, num_epochs, optimizer, train_loader, valid_loader, model_save_path, device=device):
    train_hist, valid_hist, stats = [], [], []
    total_t0 = time.time()
    model.to(device)
    early = EarlyStopping(patience=3, min_delta=1e-4)
    lowest_eval_loss = None
    best_epoch = None

    for ep in trange(num_epochs, desc="Epoch"):
        print(f"\n======== Epoch {ep} / {num_epochs} ========")
        # ---- Train ----
        model.train()
        t0 = time.time()
        tr_loss, ntrain = 0.0, 0
        for step, batch in enumerate(train_loader):
            if step % 500 == 0 and step != 0:
                elapsed = format_time(time.time()-t0)
                if torch.cuda.is_available():
                    print(f"峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                print(f"  Batch {step:>5}/{len(train_loader)} | Elapsed: {elapsed}")
            input_ids, attn_mask, labels = (t.to(device) for t in batch)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            bs = labels.size(0)
            tr_loss += loss.item() * bs
            ntrain += bs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        train_loss = tr_loss / max(1, ntrain)
        train_hist.append(train_loss)
        print(f"  Train Loss: {train_loss:.6f} | Time: {format_time(time.time()-t0)}")

        # ---- Valid ----
        model.eval()
        t0 = time.time()
        ev_loss, nvalid = 0.0, 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, attn_mask, labels = (t.to(device) for t in batch)
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                bs = labels.size(0)
                ev_loss += loss.item() * bs
                nvalid += bs
        valid_loss = ev_loss / max(1, nvalid)
        valid_hist.append(valid_loss)
        print(f"  Valid Loss: {valid_loss:.6f} | Time: {format_time(time.time()-t0)}")

        stats.append({'epoch': ep, 'Training Loss': train_loss, 'Valid. Loss': valid_loss})

        # 保存最优 + 早停
        improved = early.step(valid_loss)
        if lowest_eval_loss is None or valid_loss < lowest_eval_loss:
            lowest_eval_loss = valid_loss
            best_epoch = ep
        if improved:
            save_model(model, model_save_path, ep, lowest_eval_loss, train_hist, valid_hist)
        if early.should_stop:
            print(f"Early stopping triggered @ epoch {ep}. Best val loss = {early.best:.6f}")
            break

    print("\nTraining complete! Total: ", format_time(time.time()-total_t0))
    return train_hist, valid_hist, stats, best_epoch, lowest_eval_loss

train_loss_set, valid_loss_set, training_stats, best_epoch, lowest_eval = train(
    model, num_epochs, optimizer, train_loader, valid_loader, model_save_path, device
)

# =========================
# 可视化
# =========================
df_stats = pd.DataFrame(training_stats).set_index('epoch')
sns.set(style='darkgrid', font_scale=1.2)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(range(len(df_stats)), df_stats['Training Loss'].values, 'b-o', label="Training")
plt.plot(range(len(df_stats)), df_stats['Valid. Loss'].values, 'g-o', label="Validation")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.xticks(range(len(df_stats)))
os.makedirs('test_result', exist_ok=True)
plt.savefig('test_result/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_loss_curve.pdf')
plt.close()

# =========================
# 测试（自动加载最佳模型）
# =========================
def test(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attn_mask, labels = (t.to(device) for t in batch)
            logits = model(input_ids, attention_mask=attn_mask)  # [B]
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(p.tolist())
    return probs

best_model, best_ep_saved, lowest_eval_loss_saved, _, _ = load_model(model_save_path)
best_model = best_model.to(device).eval()
pred_probs = test(best_model, test_loader)
print(f"Using BEST model from epoch (saved): {best_ep_saved}")
print("Mean prob:", float(np.mean(pred_probs)))
print("AUC (BEST): {:.6f}".format(roc_auc_score(test_labels.numpy(), np.asarray(pred_probs))))

def Evaluate(labels_tensor, pred_probs, p=0.5):
    labels = labels_tensor.numpy() if torch.is_tensor(labels_tensor) else np.asarray(labels_tensor)
    probs = np.asarray(pred_probs).flatten()
    preds = (probs > p).astype(int)

    CM = confusion_matrix(labels, preds, labels=[0,1])
    print("Confusion Matrix:\n", CM)
    TN, FP, FN, TP = CM.ravel()
    print(f' TN:{TN}  FP:{FP}  FN:{FN}  TP:{TP}')
    print('Total positive : ', TP + FN)

    auc = roc_auc_score(labels, probs)
    print('AUC :', auc)
    prec = precision_score(labels, preds)
    rec  = recall_score(labels, preds)
    f1   = f1_score(labels, preds)
    print('precision :', prec)
    print('recall :', rec)
    print('f1 :', f1)

    precision_arr, recall_arr, _ = precision_recall_curve(labels, probs)
    pr_auc = trapz(precision_arr, recall_arr)
    print("PR-AUC: %0.4f" % pr_auc)
    ap = average_precision_score(labels, probs)
    print("AP: %0.4f" % ap)

    print('kappa :', cohen_kappa_score(labels, preds))
    print('balanced_accuracy :', balanced_accuracy_score(labels, preds))
    print("\n", classification_report(labels, preds, target_names=["Non-vulnerable", "Vulnerable"]))

    for k, (pq, rq) in topk_metrics(probs, labels, ks=[10,20,30,40,50,100,150,200]).items():
        print(f'PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%')

Evaluate(test_labels, pred_probs, p=0.5)

# =========================
# 导出结果
# =========================
df_out = pd.DataFrame(list(zip(test_id_list, pred_probs, test_labels.tolist())),
                      columns=['Func_id','prob','Label'])
ListToCSV(df_out, './XLNet_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_result.csv')
