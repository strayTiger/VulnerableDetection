import json

data_test = './test.jsonl'
data_train = './train.jsonl'
data_valid = './valid.jsonl'
funcs = []
targets_train = []
targets_valid = []
targets_test = []
with open(data_train, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        funcs.append(data['func'])
        targets_train.append(data['target'])
with open(data_test, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        funcs.append(data['func'])
        targets_test.append(data['target'])
with open(data_valid, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        funcs.append(data['func'])
        targets_valid.append(data['target'])

print(f"Total samples: {len(funcs)}")

from collections import Counter

# 查看训练集类别分布
train_counts = Counter(targets_train)
test_counts = Counter(targets_test)
valid_counts = Counter(targets_valid)
print("Training set class distribution:", train_counts)  #{0: 78078, 1: 1195}
print("Test set class distribution:", test_counts)       #{0: 26030, 1: 395}
print("Valid set class distribution:", valid_counts)     #{0: 26031, 1: 394}
