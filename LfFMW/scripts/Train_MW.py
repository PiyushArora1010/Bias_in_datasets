import os
import sys
os.chdir('..')
sys.path[0] = os.getcwd()


from data.attr_dataset import AttributeDataset_bffhq
from module.loss import GeneralizedCELoss,EMA,MultiDimAverageMeter
from module.models import dic_models
from module.models2 import dic_models_2
from data.util import get_dataset, IdxDataset, ZippedDataset, get_dataset_bffhq
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import random
from numpy.random import RandomState

def set_seed(seed: int) -> RandomState:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state


parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("dataset_in",default="ColoredMNIST-Skewed0.02-Severity4", help="Name of the Dataset")
parser.add_argument("model_in", default="resnet18", help="Name of the model")
parser.add_argument("train_samples", default=1000, type=int,help="Number of training samples")
parser.add_argument("bias_ratio", default=0.02, type = float,help="Bias ratio")
parser.add_argument("seed", default=12, type = int,help="Seed")
args = parser.parse_args()

set_seed(args.seed)

target_attr_idx = 0
bias_attr_idx = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_to_file(filename, text):
    with open(filename, 'a') as f:
        f.write(text)
        f.write('\n')


def evaluate_accuracy(mw_model, test_loader, memory_loader):
  mw_model.eval()
  mw_correct = 0
  mem_iter = iter(memory_loader)
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)
        try:
            indexm,memory_input,_ = next(mem_iter)
        except:
            mem_iter = iter(memory_loader)
            indexm,memory_input,_ = next(mem_iter)
        memory_input = memory_input.to(device)

        mw_outputs = mw_model(data, memory_input)
        mw_pred = mw_outputs.data.max(1, keepdim=True)[1]

        mw_correct += mw_pred.eq(target.data.view_as(mw_pred)).sum().item()
  mw_accuracy = 100.*(torch.true_divide(mw_correct,len(test_loader.dataset))).item()
  mw_model.train()
  return mw_accuracy


train_dataset = get_dataset(
        args.dataset_in,
        data_dir='/home/user/datasets/debias',
        dataset_split="train",
        transform_split="train",
    )
test_dataset = get_dataset(
        args.dataset_in,
        data_dir='/home/user/datasets/debias',
        dataset_split="eval",
        transform_split="eval",
    )
valid_dataset = get_dataset(
        args.dataset_in,
        data_dir='/home/user/datasets/debias',
        dataset_split="train",
        transform_split="train",
        add = True
)


indices_train_biased = train_dataset.attr[:,0] == train_dataset.attr[:,1]

indices_train_biased = indices_train_biased.nonzero().squeeze()

nums_train_biased = np.random.choice(indices_train_biased, int(args.train_samples - args.bias_ratio * args.train_samples) , replace=False)


indices_train_unbiased = train_dataset.attr[:,0] != train_dataset.attr[:,1]

indices_train_unbiased = indices_train_unbiased.nonzero().squeeze()

nums_train_unbiased = np.random.choice(indices_train_unbiased, int(args.bias_ratio * args.train_samples) , replace=False)

nums_train = np.concatenate((nums_train_biased, nums_train_unbiased))

if args.dataset_in != 'bffhq':
    nums_valid_unbiased = []
    while len(nums_valid_unbiased) < 1000:
        i = np.random.randint(0, len(valid_dataset))
        if valid_dataset.attr[i,0] != valid_dataset.attr[i,1] and i not in nums_train:
            nums_valid_unbiased.append(i)
    
    nums_valid_unbiased = np.array(nums_valid_unbiased)
    
    valid_dataset.attr = valid_dataset.attr[nums_valid_unbiased]
    valid_dataset.data = valid_dataset.data[nums_valid_unbiased]
    valid_dataset.__len__ = 1000
    valid_dataset.query_attr = valid_dataset.attr[:, torch.arange(2)]

    train_dataset.attr = train_dataset.attr[nums_train]
    train_dataset.data = train_dataset.data[nums_train]
    train_dataset.__len__ = args.train_samples
    train_dataset.query_attr = train_dataset.attr[:, torch.arange(2)]
    del indices_train_biased, indices_train_unbiased, nums_train_biased, nums_train_unbiased, nums_train, nums_valid_unbiased

else:
    train_dataset.data = [train_dataset.data[i] for i in nums_train]
    train_dataset.attr = train_dataset.attr[nums_train]
    train_dataset.__len__ = args.train_samples
    train_dataset.query_attr = train_dataset.attr[:, torch.arange(2)]
    del indices_train_biased, indices_train_unbiased, nums_train_biased, nums_train_unbiased, nums_train

print("[Size of the Dataset]["+str(len(train_dataset))+"]")
print("[Conflicting Samples in Training Data]["+str(len(train_dataset.attr[train_dataset.attr[:,0] != train_dataset.attr[:,1]]))+"]")
print("[Conflicting Samples in Validation Data]["+str(len(valid_dataset.attr[valid_dataset.attr[:,0] != valid_dataset.attr[:,1]]))+"]")
print("[Conflicting Samples in Test Data]["+str(len(test_dataset.attr[test_dataset.attr[:,0] != test_dataset.attr[:,1]]))+"]")

train_target_attr = train_dataset.attr[:, target_attr_idx]
train_bias_attr = train_dataset.attr[:, bias_attr_idx]




attr_dims = []
attr_dims.append(torch.max(train_target_attr).item() + 1)
num_classes = attr_dims[0]


train_dataset = IdxDataset(train_dataset)
valid_dataset = IdxDataset(valid_dataset)    
test_dataset = IdxDataset(test_dataset)
 
train_loader = DataLoader(
        train_dataset,
        batch_size=250,
        shuffle=True,
        drop_last=True
    )

valid_loader = DataLoader(
        valid_dataset,
        batch_size=250,
        shuffle=False,
        drop_last=False
    )

test_loader = DataLoader(
        test_dataset,
        batch_size=250,
        shuffle=False,
        drop_last=False
    )
mem_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        drop_last=True
)



try:
    model_d = dic_models[args.model_in](num_classes).to(device)
except:
    model_d = dic_models_2[args.model_in](num_classes).to(device)


if 'MNIST' in args.dataset_in:
    optimizer_d = torch.optim.Adam(model_d.parameters(),lr= 0.002, weight_decay=0.0)
    schedulerd = MultiStepLR(optimizer_d, milestones=[300], gamma=0.5)
elif args.dataset_in == 'bffhq':
    optimizer_d = torch.optim.Adam(model_d.parameters(),lr= 0.001, weight_decay=0.0)
    schedulerd = MultiStepLR(optimizer_d, milestones=[300], gamma=0.5)
else:
    optimizer_d = torch.optim.SGD(model_d.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
    schedulerd = MultiStepLR(optimizer_d, milestones=[150,225], gamma=0.1)


criterion = nn.CrossEntropyLoss()

if 'CIFAR' in args.dataset_in:
    main_num_steps = 300
elif 'MNIST' in args.dataset_in:
    main_num_steps = 200
else:
    main_num_steps = 150


test_accuracy = -1.0
test_cheat = -1.0
test_accuracy_epoch = -1.0
valid_accuracy_best = -1.0
mem_iter = None

for step in range(1, main_num_steps+1):

    for ix, (index,data,attr) in enumerate(train_loader):

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]

        try:
            _,datam,_ = next(mem_iter)
        except:
            mem_iter = iter(mem_loader)
            _,datam,_ = next(mem_iter)
        
        datam = datam.to(device)

        logit_d = model_d(data, datam)

        loss = criterion(logit_d, label)

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()
    
    schedulerd.step()

    train_accuracy_epoch = evaluate_accuracy(model_d, train_loader, mem_loader)
    prev_valid_accuracy = valid_accuracy_best
    valid_accuracy_epoch = evaluate_accuracy(model_d, valid_loader, mem_loader)
    valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)

    print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")

    test_accuracy_epoch = evaluate_accuracy(model_d, test_loader, mem_loader)

    test_cheat = max(test_cheat, test_accuracy_epoch)

    print("[Test Accuracy cheat][%.4f]"%test_cheat)

    if valid_accuracy_best > prev_valid_accuracy:
        test_accuracy = test_accuracy_epoch

    print('[Best Test Accuracy]', test_accuracy)


write_to_file('results_text/results_MW_'+args.dataset_in.split('-')[0]+'_'+str(args.train_samples)+'_'+str(args.bias_ratio)+'.txt','[Best Test Accuracy]'+str(test_accuracy)+"[Final Epoch Test Accuracy]"+str(test_accuracy_epoch)+ '[Best Cheat Test Accuracy]'+str(test_cheat))

