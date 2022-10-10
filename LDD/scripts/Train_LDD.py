import os
import sys
os.chdir('..')
sys.path[0] = os.getcwd()


from module.loss import GeneralizedCELoss
from util import EMA
from module.models import dic_models
from module.models2 import dic_models_2
from data.util import get_dataset, IdxDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torch.optim.lr_scheduler import MultiStepLR
import random
from numpy.random import RandomState
import torchvision.transforms.functional as TF
import cv2 as cv

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


def evaluate_accuracy(model_b, model_l, data_loader, model='label'):
        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0

        for index, data, attr in data_loader:
            label = attr[:, 0]

            data = data.to(device)
            label = label.to(device)


            with torch.no_grad():

                try:
                    z_l, z_b = [], []
                    hook_fn = model_l.avgpool.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()
                    
                    z_l = z_l[0]
                    hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                except:
                    z_l, z_b = [], []
                    hook_fn = model_l.layer4.register_forward_hook(concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()
                    
                    z_l = z_l[0]
                    hook_fn = model_b.layer4.register_forward_hook(concat_dummy(z_b))
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]    

                z_origin = torch.cat((z_l, z_b), dim=1)

                if model == 'bias':
                    pred_label = model_b.fc(z_origin)
                else:
                    pred_label = model_l.fc(z_origin)

                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = 100*total_correct/float(total_num)
        model_b.train()
        model_l.train()

        return accs.item()

def concat_dummy(z):
    def hook(model, input, output):
        z.append(output.squeeze())
        return torch.cat((output, torch.zeros_like(output)), dim=1)
    return hook

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
        drop_last=True,

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

try:
    model_d = dic_models[args.model_in](num_classes).to(device)
    model_b = dic_models[args.model_in](num_classes).to(device)
except:
    model_d = dic_models_2[args.model_in](num_classes).to(device)
    model_b = dic_models_2[args.model_in](num_classes).to(device)

model_d.fc = nn.Linear(model_d.fc.in_features*2, num_classes).to(device)
model_b.fc = nn.Linear(model_b.fc.in_features*2, num_classes).to(device)

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


sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=num_classes, alpha=0.9)
sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), num_classes=num_classes, alpha=0.9)

if 'CIFAR' in args.dataset_in:
    main_num_steps = 300
    lambda_swap_align = 5.0
    lambda_dis_align = 5.0
    lambda_swap_ = 1.0
elif 'MNIST' in args.dataset_in:
    main_num_steps = 200
    lambda_swap_align = 10.0
    lambda_dis_align = 10.0
    lambda_swap_ = 1.0
else:
    main_num_steps = 150
    lambda_swap_align = 2.0
    lambda_dis_align = 2.0
    lambda_swap_ = 0.5

test_accuracy = -1.0
test_cheat = -1.0
test_accuracy_epoch = -1.0
valid_accuracy_best = -1.0

print("[Length Train Loader][{}]".format(len(train_loader)))

for epoch in range(1, main_num_steps+1):
    for ix, (index, data, attr) in enumerate(train_loader):

        data = data
        label = attr[:,target_attr_idx]

        data = data.to(device)
        label = label.to(device)
    
        try:
            z_b = []
            hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_b))
            _ = model_b(data)
            hook_fn.remove()
            z_b = z_b[0]
            
            z_d = []
            hook_fn = model_d.avgpool.register_forward_hook(concat_dummy(z_d))
            _ = model_d(data)
            hook_fn.remove()
            z_d = z_d[0]

            if epoch == 1 and ix == 0:
                print("[Average Pool layer Selected]")
        except:
            z_b = []
            hook_fn = model_b.layer4.register_forward_hook(concat_dummy(z_b))
            _ = model_b(data)
            hook_fn.remove()
            z_b = z_b[0]
            
            z_d = []
            hook_fn = model_d.layer4.register_forward_hook(concat_dummy(z_d))
            _ = model_d(data)
            hook_fn.remove()
            z_d = z_d[0]
            if epoch == 1 and ix == 0:
                print("[Layer 4 Selected]")

        z_conflict = torch.cat((z_d, z_b.detach()), dim=1)
        z_align = torch.cat((z_d.detach(), z_b), dim=1)

       
        pred_conflict = model_d.fc(z_conflict)
        pred_align = model_b.fc(z_align)
        loss_dis_conflict = criterion(pred_conflict, label)
        loss_dis_align = criterion(pred_align, label)
        
        loss_dis_conflict = loss_dis_conflict.detach()
        loss_dis_align = loss_dis_align.detach()
        
        sample_loss_ema_d.update(loss_dis_conflict, index)
        sample_loss_ema_b.update(loss_dis_align, index)

        loss_dis_conflict = sample_loss_ema_d.parameter[index].clone().detach()
        loss_dis_align = sample_loss_ema_b.parameter[index].clone().detach()

        loss_dis_conflict = loss_dis_conflict.to(device)
        loss_dis_align = loss_dis_align.to(device)

        for c in range(num_classes):
            class_index = torch.where(label == c)[0].to(device)
            max_loss_conflict = sample_loss_ema_d.max_loss(c)
            max_loss_align = sample_loss_ema_b.max_loss(c)
            loss_dis_conflict[class_index] /= max_loss_conflict
            loss_dis_align[class_index] /= max_loss_align

        loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)  
        loss_weight = loss_weight.to(device)
        
        loss_dis_conflict = criterion(pred_conflict, label) * loss_weight           
        loss_dis_align = bias_criterion(pred_align, label)
  

        if epoch >= 30:
            indices = np.random.permutation(z_b.size(0))
            z_b_swap = z_b[indices]         # z tilde
            label_swap = label[indices]     # y tilde

            z_mix_conflict = torch.cat((z_d, z_b_swap.detach()), dim=1)
            z_mix_align = torch.cat((z_d.detach(), z_b_swap), dim=1)

            pred_mix_conflict = model_d.fc(z_mix_conflict)
            pred_mix_align = model_b.fc(z_mix_align)
            loss_swap_conflict = criterion(pred_mix_conflict, label) * loss_weight     
            loss_swap_align = bias_criterion(pred_mix_align, label_swap)                               
            lambda_swap = lambda_swap_    
        else:
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda_swap = 0
        
        
        loss_dis  = loss_dis_conflict.mean() + lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
        loss_swap = loss_swap_conflict.mean() + lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
        loss = loss_dis + lambda_swap * loss_swap 

        optimizer_d.zero_grad()
        optimizer_b.zero_grad()
        loss.backward()
        optimizer_d.step()
        optimizer_b.step()

    schedulerb.step()
    schedulerd.step()

    train_accuracy_epoch = evaluate_accuracy(model_b,model_d, train_loader)
    prev_valid_accuracy = valid_accuracy_best
    valid_accuracy_epoch = evaluate_accuracy(model_b,model_d, valid_loader)
    valid_accuracy_best = max(valid_accuracy_best, valid_accuracy_epoch)

    print("[Epoch "+str(epoch)+"][Train Accuracy", round(train_accuracy_epoch,4),"][Validation Accuracy",round(valid_accuracy_epoch,4),"]")

    test_accuracy_epoch = evaluate_accuracy(model_b, model_d, test_loader)

    test_cheat = max(test_cheat, test_accuracy_epoch)

    print("[Test Accuracy cheat][%.4f]"%test_cheat)

    if valid_accuracy_best > prev_valid_accuracy:
        test_accuracy = test_accuracy_epoch

    print('[Best Test Accuracy]', test_accuracy)


write_to_file('results_text/results_Vanilla_LDD_'+args.dataset_in.split('-')[0]+'_'+str(args.train_samples)+'_'+str(args.bias_ratio)+'.txt','[Best Test Accuracy]'+str(test_accuracy)+"[Final Epoch Test Accuracy]"+str(test_accuracy_epoch)+ '[Best Cheat Test Accuracy]'+str(test_cheat))

