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


def weights_loss(model_d, model_b, indexm,datam, labelm, datam1):
        logit_b_mem = model_b(datam)
     
        logit_d_mem = model_d(datam, datam1,'Error',False )
      
        loss_b_mem = criterion(logit_b_mem, labelm).cpu().detach()
        loss_d_mem = criterion(logit_d_mem, labelm).cpu().detach()

        label_cpu_mem = labelm.cpu()

        for c in range(num_classes):
            class_index = np.where(label_cpu_mem == c)[0]
            max_loss_b_mem = sample_loss_ema_b_mem.max_loss(c)
            max_loss_d_mem = sample_loss_ema_d_mem.max_loss(c)
            loss_b_mem[class_index] /= max_loss_b_mem
            loss_d_mem[class_index] /= max_loss_d_mem
   
        loss_weight_mem = loss_b_mem / (loss_b_mem + loss_d_mem + 1e-8)
        loss_weight_mem = loss_weight_mem.detach()
        return loss_weight_mem


def evaluate_accuracy_mem(mw_model, model_b, test_loader, memory_loader, mem_loader_1):
  mw_model.eval()
  mw_correct = 0

  mem_iter = iter(memory_loader)
  mem_iter_ = iter(mem_loader_1)
  with torch.no_grad():
    for _, data, target in test_loader:
        data = data.to(device)
        target = target[:,target_attr_idx]
        target = target.to(device)
        try:
            _,data_m_1, _ = next(mem_iter_)
        except:
            mem_iter_ = iter(mem_loader_1)
            _,data_m_1, _ = next(mem_iter_)
        try:
            indexm,memory_input,attrm = next(mem_iter)
            _,data_m_1, _ = next(mem_iter_)
        except:
            mem_iter = iter(memory_loader)
            indexm,memory_input,attrm = next(mem_iter)
            mem_iter_ = iter(mem_loader_1)
            _,data_m_1, _ = next(mem_iter_)
        data_m_1 = data_m_1.to(device)
        labelm = attrm[:,target_attr_idx].to(device)
        memory_input = memory_input.to(device)
        weights_mul = weights_loss(mw_model, model_b, indexm, memory_input, labelm, data_m_1)
 
        weights_mul = weights_mul.to(device)
        weights_mul = weights_mul.view(weights_mul.size(0), 1)
        mw_outputs  = mw_model(data,memory_input, weights_mul, True)
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

mem_loader_1 = DataLoader(
    train_dataset,
    batch_size=100,
    shuffle=True,
    drop_last=True
)

try:
    model_d = dic_models[args.model_in+'_weights'](num_classes).to(device)
    model_b = dic_models[args.model_in](num_classes).to(device)
except:
    model_d = dic_models_2[args.model_in+'_weights'](num_classes).to(device)
    model_b = dic_models_2[args.model_in](num_classes).to(device)

if 'MNIST' in args.dataset_in:
    optimizer_b = torch.optim.Adam(model_b.parameters(),lr= 0.002, weight_decay=0.0)
    optimizer_d = torch.optim.Adam(model_d.parameters(),lr= 0.002, weight_decay=0.0)
    schedulerd = MultiStepLR(optimizer_d, milestones=[300], gamma=0.5)
    schedulerb = MultiStepLR(optimizer_b, milestones=[300], gamma=0.5)
elif args.dataset_in == 'bffhq':
    optimizer_b = torch.optim.Adam(model_b.parameters(),lr= 0.001, weight_decay=0.0)
    optimizer_d = torch.optim.Adam(model_d.parameters(),lr= 0.001, weight_decay=0.0)
    schedulerd = MultiStepLR(optimizer_d, milestones=[300], gamma=0.5)
    schedulerb = MultiStepLR(optimizer_b, milestones=[300], gamma=0.5)
else:
    optimizer_b = torch.optim.SGD(model_b.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
    optimizer_d = torch.optim.SGD(model_d.parameters(),lr= 0.1, weight_decay=5e-4, momentum = 0.9, nesterov = True)
    schedulerd = MultiStepLR(optimizer_d, milestones=[150,225], gamma=0.1)
    schedulerb = MultiStepLR(optimizer_b, milestones=[150,225], gamma=0.1)


criterion = nn.CrossEntropyLoss(reduction='none')
bias_criterion = GeneralizedCELoss()


sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

sample_loss_ema_b_mem = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
sample_loss_ema_d_mem = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

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
mem_iter_1 = None

for step in range(1, main_num_steps+1):

    for ix, (index,data,attr) in enumerate(train_loader):
        
        try:
            _,datam1,_ = next(mem_iter_1)
        except:
            mem_iter_1 = iter(mem_loader_1)
            _,datam1,_ = next(mem_iter_1)
        try:
            indexm,datam, attrm = next(mem_iter)
            _,datam1,_ = next(mem_iter_1)
        except:
            mem_iter = iter(mem_loader)
            indexm,datam, attrm = next(mem_iter)
            _,datam1,_ = next(mem_iter_1)

        datam1 = datam1.to(device)
        datam = datam.to(device)
        attrm = attrm.to(device)

        labelm = attrm[:,target_attr_idx]

        data = data.to(device)
        attr = attr.to(device)
        label = attr[:, target_attr_idx]

        bias_label = attr[:, bias_attr_idx]

        logit_b_mem = model_b(datam)

        logit_d_mem = model_d(datam, datam1, 'Error', False)
      
        loss_b_mem = criterion(logit_b_mem, labelm).cpu().detach()
        loss_d_mem = criterion(logit_d_mem, labelm).cpu().detach()

        loss_per_sample_b_mem = loss_b_mem
        loss_per_sample_d_mem = loss_d_mem

        sample_loss_ema_b_mem.update(loss_b_mem, indexm)
        sample_loss_ema_d_mem.update(loss_d_mem, indexm)

        loss_b_mem = sample_loss_ema_b_mem.parameter[indexm].clone().detach()
        loss_d_mem = sample_loss_ema_d_mem.parameter[indexm].clone().detach()

        label_cpu_mem = labelm.cpu()

        for c in range(num_classes):
            class_index = np.where(label_cpu_mem == c)[0]
            max_loss_b_mem = sample_loss_ema_b_mem.max_loss(c)
            max_loss_d_mem = sample_loss_ema_d_mem.max_loss(c)
            loss_b_mem[class_index] /= max_loss_b_mem
            loss_d_mem[class_index] /= max_loss_d_mem
   

        loss_weight_mem = loss_b_mem / (loss_b_mem + loss_d_mem + 1e-8)
        loss_weight_mem = loss_weight_mem.detach()
        logit_b = model_b(data)
        loss_weight_mem = loss_weight_mem.to(device)
        loss_weight_mem = loss_weight_mem.view(loss_weight_mem.size(0), 1)
        logit_d = model_d(data, datam,loss_weight_mem, True)
        
        loss_b = criterion(logit_b, label).cpu().detach()
        loss_d = criterion(logit_d, label).cpu().detach()

        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d

        sample_loss_ema_b.update(loss_b, index)
        sample_loss_ema_d.update(loss_d, index)

        loss_b = sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = sample_loss_ema_d.parameter[index].clone().detach()

        label_cpu = label.cpu()
        
        for c in range(num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d
   
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)

        loss_b_update = bias_criterion(logit_b, label)

        loss_d_update = criterion(logit_d, label) * loss_weight.to(device)

        loss = loss_b_update.mean() + loss_d_update.mean()

        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()
    
    train_accuracy_epoch = evaluate_accuracy_mem(model_d, model_b, train_loader, mem_loader, mem_loader_1)
    prev_valid_accuracy = valid_accuracy_best
    valid_accuracy_epoch = evaluate_accuracy_mem(model_d, model_b, valid_loader, mem_loader, mem_loader_1)
    valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)

    print("[Epoch "+str(step)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")

    test_accuracy_epoch = evaluate_accuracy_mem(model_d, model_b, test_loader, mem_loader, mem_loader_1)

    test_cheat = max(test_cheat, test_accuracy_epoch)

    print("[Test Accuracy cheat][%.4f]"%test_cheat)

    if valid_accuracy_best > prev_valid_accuracy:
        test_accuracy = test_accuracy_epoch

    print('[Best Test Accuracy]', test_accuracy)


write_to_file('results_text/results_weights_'+args.dataset_in.split('-')[0]+'_'+str(args.train_samples)+'_'+str(args.bias_ratio)+'.txt','[Best Test Accuracy]'+str(test_accuracy)+"[Final Epoch Test Accuracy]"+str(test_accuracy_epoch)+ '[Best Cheat Test Accuracy]'+str(test_cheat))