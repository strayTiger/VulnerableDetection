# -*- coding: utf-8 -*-
# Code is modified based on:
# http://restanalytics.com/2021-05-04-Fine-Tuning-XLNet-For-Sequence-Classification/
from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import glob
import time
import datetime
import pandas as pd
import datetime
import pickle
import random
import re
import shutil
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns
# pred_probs
import statistics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, WeightedRandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from DataLoader import SplitCharacters, ListToCSV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from numpy import trapz
#from scipy.integrate import simps
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import average_precision_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, roc_auc_score
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import torch
# pytorch 处理数据的包
from torch.utils.data import TensorDataset, DataLoader,RandomSampler,SequentialSampler
from torch.nn import BCEWithLogitsLoss
# 对数据进行补齐功能的包
from keras_preprocessing.sequence import pad_sequences
# 数据分割
from sklearn.model_selection import train_test_split
# 英文分词， 分类
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from torch.cuda.amp import autocast, GradScaler  # 添加 FP16 混合精度
# Adam + L2正则
#from transformers import AdamW
from torch.optim import AdamW

MAX_LEN = 768
# MAX_LEN = 1024
BATCH_SIZE = 16
max_grad_norm = 1.0 
num_epochs = 10
path_model = './xlnet_model'
large_pathmodel = './xlnet_large_model'

from peft import LoraConfig, get_peft_model

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放因子
    target_modules=[
        #"rel_attn.q",
        # "rel_attn.k",
        # "rel_attn.v",
        # "rel_attn.o",
        "ff.layer_1",
        "ff.layer_2",
    ],
    lora_dropout=0.1,  # Dropout 概率
    bias="none"  # 不对偏置参数进行调整
)

def generateIdCodeLabels(json_file_path):
    id_list = []
    label_list = []
    func_body_list = []
    with open(json_file_path) as f:
        for line in f:
            js = json.loads(line.strip())
            id_list.append(js['func_name'])
            label_list.append(js['target'])
            code = js['func'].split()
            new_sub_line = []
            for element in code:
                new_element = SplitCharacters(element)
                new_sub_line.append(new_element)
            code = ' '.join(new_sub_line)
            func_body_list.append(code)
    return id_list, label_list, func_body_list

data_dir = r'./'
model_save_path = r'./result/'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_xlnet.bin'

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

train_id_list, train_label_list, train_func_list = generateIdCodeLabels(data_dir + "ffmpeg_train.jsonl")
validation_id_list, validation_label_list, validation_func_list = generateIdCodeLabels(data_dir + "ffmpeg_val.jsonl")
test_id_list, test_label_list, test_func_list = generateIdCodeLabels(data_dir + "ffmpeg_test.jsonl")

print("The length of the training set is: " + str(len(train_label_list)) + ", there are " + str(
    np.count_nonzero(train_label_list)) + " vulnerable samples.")
print("The length of the validation set is: " + str(len(validation_label_list)) + ", there are " + str(
    np.count_nonzero(validation_label_list)) + " vulnerable samples.")
print("The length of the test set is: " + str(len(test_label_list)) + ", there are " + str(
    np.count_nonzero(test_label_list)) + " vulnerable samples.")

# Load models from Hugging Face community
tokenizer = XLNetTokenizer.from_pretrained(path_model, do_lower_case=True)
#model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)


class Model(nn.Module):

    def __init__(self, num_labels=2):
        super(Model, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained(path_model)
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):

        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids
                                       )
        # pool the outputs into a mean vector
        # last_hidden_state : [4, 768, 768]  4: batch_size  768: sequence_length
        # mean_last_hidden_state : [4, 768]
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        logits = logits[:, 1] - logits[:, 0] # 【2.3，4.6】-》4.6-2.3=2.3>0 为正例，有漏洞
        if labels is not None:

            loss = BCEWithLogitsLoss()(logits, labels.float())
            return loss
        else:
            return logits

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def codeTokenization_Padding(code_list):
    code_sequences = [code + " [SEP] [CLS]" for code in code_list]
    tokenized_code = [tokenizer.tokenize(code) for code in code_sequences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_code]  # 将token转换成id
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # 补0处不参与计算 >0为1 否则为0 转换成浮点数行书
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, attention_masks

train_func_pad, train_masks = codeTokenization_Padding(train_func_list)
valid_func_pad, validation_masks = codeTokenization_Padding(validation_func_list)
test_func_pad, test_masks = codeTokenization_Padding(test_func_list)

# 转换成torch到的数据格式
train_inputs = torch.tensor(train_func_pad).to(torch.int64)
validation_inputs = torch.tensor(valid_func_pad).to(torch.int64)
test_inputs = torch.tensor(test_func_pad).to(torch.int64)

train_labels = torch.tensor(train_label_list)
validation_labels = torch.tensor(validation_label_list)
test_labels = torch.tensor(test_label_list)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)

# Handling data imbalance by assigning more weights to minority class.
class_sample_count = np.unique(train_label_list, return_counts=True)[1]
weight = 1. / class_sample_count
samples_weight = weight[train_label_list]
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
weighted_train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# 训练数据batch
batch_size = BATCH_SIZE #batchsize
train_data = TensorDataset(train_inputs, train_masks, train_labels) # input_ids, attention_mask, labels
#train_sampler = RandomSampler(train_data) # 训练时需要shuffle数据 RandomSampler从迭代器里面随机取样本
train_dataloader = DataLoader(train_data, sampler=weighted_train_sampler, batch_size=batch_size) # 把把数据放入DataLoader迭代器里面

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)# input_ids, attention_mask, labels
validation_sampler = SequentialSampler(validation_data) #测试时不需要shuffle数据
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)# 把数据放入DataLoader迭代器里面

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = Model(num_labels=2)
for name, param in model.named_parameters():
    print(name)

model = get_peft_model(model, lora_config)  

model.cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

param_optimizer = list(model.named_parameters())

# bias:偏置
# gamma,beta:normalization layer参数
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()  # 预测值flatten摊平 [32]
    labels_flat = labels.flatten()  # 真实值摊平
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# ―――――――――――――――――――― 新增开始 ――――――――――――――――――――
def topk_metrics(pred_probs, labels, ks=(10,20,30,40,50,100,150,200)):
    """
    计算 Top-k Precision / Recall
    pred_probs : 1-D array-like，概率或分数（越大越像正类）
    labels     : 1-D array-like，0/1 真值
    ks         : iterable，需要评估的 K 值
    返回 dict  {k: (PQk, RQk)}
    """
    scores = np.asarray(pred_probs).flatten()
    labels = np.asarray(labels).flatten()
    total_pos = labels.sum()                       # 数据集中真实正样本总数
    order = np.argsort(-scores)                    # 分数降序
    sorted_labels = labels[order]

    results = {}
    for k in ks:
        k = int(k)
        if k > len(sorted_labels):
            continue                               # 样本数不足时跳过
        tp_k = sorted_labels[:k].sum()             # 前 k 中有多少真漏洞
        pq_k = tp_k / k
        rq_k = tp_k / total_pos if total_pos else 0
        results[k] = (pq_k, rq_k)
    return results
# ―――――――――――――――――――― 新增结束 ――――――――――――――――――――

# function to save and load the model form a specific epoch
def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs,
                  'lowest_eval_loss': lowest_eval_loss,
                  'state_dict': model_to_save.state_dict(),
                  'train_loss_hist': train_loss_hist,
                  'valid_loss_hist': valid_loss_hist,
                  'optimizer_state_dict': optimizer.state_dict()
                  }

    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs, lowest_eval_loss))
    return


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = Model(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]

    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist
"""
Training loop:

Tell the model to compute gradients by setting the model in train mode
Unpack our data inputs and labels
Load data onto the GPU for acceleration
Clear out the gradients calculated in the previous pass. In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out
Forward pass (feed input data through the network)
Backward pass (backpropagation)
Tell the network to update parameters with optimizer.step()
Track variables for monitoring progress Evalution loop:

Tell the model not to compute gradients by setting th emodel in evaluation mode

Unpack our data inputs and labels
Load data onto the GPU for acceleration
Forward pass (feed input data through the network)
Compute loss on our validation data and track variables for monitoring progress

"""


def train(model, num_epochs,
          optimizer,
          train_dataloader, valid_dataloader,
          model_save_path,
          train_loss_set=[], valid_loss_set=[],
          lowest_eval_loss=None, start_epoch=0,
          device = device
          ):
    """
    Train the model and save the model with the lowest validation loss
    """
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()

    model.to(device)

    # trange is a tqdm wrapper around the python range function
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(actual_epoch, num_epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            #print('开始每个epoch的每个batch的数据')
            if step % 500 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # 转换为 GB
                print(f"训练峰值显存: {peak_memory:.2f} GB")
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Enable gradient clipping.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss / num_train_samples
        train_loss_set.append(epoch_train_loss)

        #     print("Train loss: {}".format(epoch_train_loss))

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(epoch_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Valid loss: {}".format(epoch_eval_loss))

        # Report the final accuracy for this validation run.
        #     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        #     print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        #   avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(epoch_eval_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': actual_epoch,
                'Training Loss': epoch_train_loss,
                'Valid. Loss': epoch_eval_loss,
                #             'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            # save model
            save_model(model, model_save_path, actual_epoch,
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch,
                           lowest_eval_loss, train_loss_set, valid_loss_set)

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
    return model, train_loss_set, valid_loss_set, training_stats


model, train_loss_set, valid_loss_set, training_stats = train(model=model,
                                                              num_epochs=num_epochs,
                                                              optimizer=optimizer,
                                                              train_dataloader=train_dataloader,
                                                              valid_dataloader=validation_dataloader,
                                                              model_save_path=model_save_path,
                                                              device="cuda")

# Display floats with two decimal places.
#pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# Display the table.
df_stats

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([0, 1, 2, 3, 4, 5])
plt.savefig('test_result' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_.pdf')
#plt.show()

### Get Predictions
def test(model, test_dataloader):
    print("Testing...")
    # Testing
    pred_probs = []
    model.eval()
    # Tracking variables

    test_accuracy = 0
    nb_test_steps, nb_test_examples = 0, 0

    # Test data for one epoch
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)  # 不传labels
            # Move logits and labels to CPU
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs.extend(logits.tolist())
    return pred_probs

pred_probs = test(model, test_dataloader)
print ("The mean of predicted probabilities:")
statistics.mean(pred_probs)
auc_value = roc_auc_score(test_labels, np.asarray(pred_probs) >0.5)
print("auc  on test {}".format(auc_value))

def Evaluate(labels, pred_probs, p=0.5):
    predictions=np.asarray(pred_probs) > p
    CM = confusion_matrix(labels, predictions > p)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print(' (True Negatives): {}'.format(TN))
    print(' (False Negatives):  {}'.format(FN))
    print(' (True Positives): {}'.format(TP))
    print('(False Positives):{}'.format(FP))
    print('Total positive : ', np.sum(CM[1]))
    auc = roc_auc_score(labels, predictions)
    prec = precision_score(labels, predictions > 0.5)
    rec = recall_score(labels, predictions > 0.5)
    # calculate F1 score
    f1 = f1_score(labels, predictions > p)
    print('auc :{}'.format(auc))
    print('precision :{}'.format(prec))
    print('recall :{}'.format(rec))
    print('f1 :{}'.format(f1))
    # Compute Precision-Recall and plot curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions > 0.5)
    # use the trapezoidal rule to calculate the area under the precion-recall curve
    area = trapz(recall, precision)

    # area =  simps(recall, precision)
    print("Area Under Precision Recall  Curve(AP): %0.4f" % area)  # should be same as AP?
    average_precision = average_precision_score(labels, predictions > 0.5)
    print("average precision: %0.4f" % average_precision)
    kappa = cohen_kappa_score(labels, predictions > 0.5)
    print('kappa :{}'.format(kappa))
    balanced_accuracy = balanced_accuracy_score(labels, predictions > 0.5)
    print('balanced_accuracy :{}'.format(balanced_accuracy))
    target_names = ["Non-vulnerable", "Vulnerable"]  # non-vulnerable->0, vulnerable->1
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print("\r\n")
    print(classification_report(labels, predictions, target_names=target_names))
    topk_res = topk_metrics(pred_probs, labels, ks=[10,20,30,40,50,100,150,200])
    for k, (pq, rq) in topk_res.items():
        print(f'PQ{k:>3} = {pq*100:5.2f}% | RQ{k:>3} = {rq*100:5.2f}%')
#print("输出预测分数")
#print(pred_probs)
Evaluate(test_labels, pred_probs)

zippedlist = list(zip(test_id_list, pred_probs, test_labels))

result_set = pd.DataFrame(zippedlist, columns=['Func_id', 'prob', 'Label'])
output_dir = '.'

ListToCSV(result_set, output_dir + os.sep + 'XLNet' + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_result.csv')
