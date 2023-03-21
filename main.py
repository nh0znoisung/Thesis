# Import dependencies
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder
import os
from os import path
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.model_selection import KFold
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
# from torch.nn import functional as F
import torchvision
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from sklearn.metrics import f1_score, recall_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

import socket
import sys
import time
from datetime import timedelta, datetime
import errno
import random
import numbers

def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        acc = correct.float().sum().mul_(1.0 / batch_size)
    return acc, pred

def calc_metrics(y_pred, y_true, y_scores):
    metrics = {}
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()
    y_scores = torch.cat(y_scores).cpu().numpy()
    classes = unique_labels(y_true, y_pred)

    # recall score
    metrics['rec'] = recall_score(y_true, y_pred, average='macro')

    # f1 score
    f1_scores = f1_score(y_true, y_pred, average=None, labels=unique_labels(y_pred))
    metrics['f1'] = f1_scores.sum() / classes.shape[0]

    # AUC PR
    Y = label_binarize(y_true, classes=classes.astype(int).tolist())
    metrics['aucpr'] = average_precision_score(Y, y_scores, average='macro')

    # AUC ROC
    metrics['aucroc'] = roc_auc_score(Y, y_scores, average='macro')

    return metrics

# Custom Classifier
class CustomClassifier(nn.Module):
    def __init__(self, in_feature: int, out_feature: int = 3):
        super(CustomClassifier, self).__init__()
        
        self.dense1 = nn.Linear(in_feature, 64) #2040
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(64, out_feature)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        # x = self.softmax(x)
        return x

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def write_log(writer, losses, acc, metrics, e, tag='set'):
    # tensorboard
    writer.add_scalar(f'loss/{tag}', losses.avg, e)
    writer.add_scalar(f'acc/{tag}', acc.avg, e)
    writer.add_scalar(f'rec/{tag}', metrics['rec'], e)
    writer.add_scalar(f'f1/{tag}', metrics['f1'], e)
    writer.add_scalar(f'aucpr/{tag}', metrics['aucpr'], e)
    writer.add_scalar(f'aucroc/{tag}', metrics['aucroc'], e)


def save_checkpoint(epoch, model, tag):
    # 'model_state_dict': model.state_dict(),
    target_path = os.path.join(target_dir, current_id, model_name + f"_{tag}.pt") # "/content/drive/My Drive/Thesis_Final/models/Affectnet-WM-RR-sample/enet_mtl_acc.pt"
    os.makedirs(os.path.join(target_dir, current_id) , exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model
    }, target_path)
    print(f"-------Save model with tag {tag} at epoch {epoch} at {target_path}")

class Logger(object):
    console = sys.stdout

    def __init__(self, fpath=None):
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class RandomFiveCrop(object):

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        # randomly return one of the five crops
        return F.five_crop(img, self.size)[random.randint(0, 4)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


### Global config
num_epoch = 30
ft_num_epoch = 5

# define R1 and R2 regularization coefficients
lambda1 = 0.001
lambda2 = 0.001

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


batch_size = 32 # 16 vs 32 vs 64 vs 128
img_size = 256 # ??? (320, 135) vs (368, 368) vs (240,240)    



# transformations = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.GaussianBlur(kernel_size=(61, 61), sigma=(0.1, 2.0)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
# ])
normalize = transforms.Normalize(mean=[0.5752, 0.4495, 0.4012],
                                     std=[0.2086, 0.1911, 0.1827])

train_transformations=transforms.Compose([
    transforms.Resize(256),
    RandomFiveCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

valid_transformations=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# base_path = "/content/drive/My Drive/Thesis_Final"
base_path = ""
pretrained_base_path = os.path.join(base_path, "pretrained_models") # "/content/drive/My Drive/Thesis_Final/pretrained_models"
model_base_path = os.path.join(base_path, "models") # "/content/drive/My Drive/Thesis_Final/models"
log_base_path = os.path.join(base_path, "logs") # "/content/drive/My Drive/Thesis_Final/logs"
dataset_base_path = os.path.join(base_path, "datasets") #"/content/drive/My Drive/Thesis_Final/datasets"


def freeze_model(model):
    print("----- Freeze base architecture for fine tuning ------")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_model(model):
    print("----- Unfreeze whole architecture for fine tuning ------")
    for param in model.parameters():
        param.requires_grad = True


def train(train_loader, model, criterion, optimizer, epoch, num_epoch):
    y_pred, y_true, y_scores = [], [], []
    accs = AverageMeter()
    losses = AverageMeter()



    # switch to train mode
    model.train()

    # Train model with no log
    # print("----- Starting epoch {}".format(epoch))

    y_pred, y_true, y_scores = [], [], []
    with tqdm(total=int(len(train_loader.dataset) / batch_size)) as pbar:
        for _, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Compute output
            output = model(images)
            loss = criterion(output,labels)

            # add R1 regularization
            l1_regularization = torch.tensor(0.0).to(device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, p=1)
            loss += lambda1 * l1_regularization
            
            # add R2 regularization
            l2_regularization = torch.tensor(0.0).to(device)
            for param in model.parameters():
                l2_regularization += torch.norm(param, p=2)
            loss += lambda2 * l2_regularization

            
            # measure accuracy and record loss
            acc, pred = accuracy(output, labels)
            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(labels)
            y_scores.append(output.data)


            # compute grads + opt step
            optimizer.zero_grad() #
            loss.backward() #
            optimizer.step() #
            # print("Batch: ", i+1, end="   ")


            # progressbar
            pbar.set_description(f'TRAINING [{epoch:02d}/{num_epoch}]')
            pbar.set_postfix({'L': losses.avg,
                              'acc': accs.avg})
            pbar.update(1)
        
    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] TRAIN [{epoch:02d}/{num_epoch}] | '
        f'loss={losses.avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)
    write_log(writer, losses, accs, metrics, epoch, tag='train')

def validate(valid_loader, model, criterion, epoch, num_epoch):
    y_pred, y_true, y_scores = [], [], []
    accs = AverageMeter()
    losses = AverageMeter()


    
    # switch to train mode
    model.eval()

    # print("----- Starting epoch {}".format(epoch))

    y_pred, y_true, y_scores = [], [], []
    with tqdm(total=int(len(valid_loader.dataset) / batch_size)) as pbar:
      with torch.no_grad():
        for _, (images, labels) in enumerate(valid_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Compute output
            output = model(images)
            loss = criterion(output,labels)

            # add R1 regularization
            l1_regularization = torch.tensor(0.0).to(device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, p=1)
            loss += lambda1 * l1_regularization
            
            # add R2 regularization
            l2_regularization = torch.tensor(0.0).to(device)
            for param in model.parameters():
                l2_regularization += torch.norm(param, p=2)
            loss += lambda2 * l2_regularization

            
            # measure accuracy and record loss
            acc, pred = accuracy(output, labels)
            losses.update(loss.item(), images.size(0))
            accs.update(acc.item(), images.size(0))

            # collect for metrics
            y_pred.append(pred)
            y_true.append(labels)
            y_scores.append(output.data)


            # progressbar
            pbar.set_description(f'VALIDATING [{epoch:03d}/{num_epoch}]')
            pbar.update(1)
            
        
    metrics = calc_metrics(y_pred, y_true, y_scores)
    progress = (
        f'[-] VALIDATING [{epoch:02d}/{num_epoch}] | '
        f'loss={losses.avg:.4f} | '
        f'acc={accs.avg:.4f} | '
        f'rec={metrics["rec"]:.4f} | '
        f'f1={metrics["f1"]:.4f} | '
        f'aucpr={metrics["aucpr"]:.4f} | '
        f'aucroc={metrics["aucroc"]:.4f}'
    )
    print(progress)

    # save model checkpoints for best valid
    if accs.avg > best_valid['acc']:
        save_checkpoint(epoch, model, 'acc')
    if metrics['rec'] > best_valid['rec']:
        save_checkpoint(epoch, model, 'rec')

    best_valid['acc'] = max(best_valid['acc'], accs.avg)
    best_valid['rec'] = max(best_valid['rec'], metrics['rec'])
    best_valid['f1'] = max(best_valid['f1'], metrics['f1'])
    best_valid['aucpr'] = max(best_valid['aucpr'], metrics['aucpr'])
    best_valid['aucroc'] = max(best_valid['aucroc'], metrics['aucroc'])
    write_log(writer, losses, accs, metrics, epoch, tag='valid')


def main(input_feature: str, model_id: str, dataset: str):
    """
    input_feature = 'WM' # 'WM' or 'TH'
    model_name = "enet_mtl"
    dataset = "Affectnet-WM-RR-sample"
    """
    # define logger
    global current_id, model_name, target_dir, writer, best_valid
    model_name = model_id
    current_id = datetime.now().strftime('%b%d_%H-%M-%S') + socket.gethostname()
    log_dir = os.path.join(log_base_path, input_feature, model_name, current_id) # "/content/drive/My Drive/Thesis_Final/logs/WM/enet28"
    os.makedirs(log_dir, exist_ok = True)
    writer = SummaryWriter(log_dir)
    sys.stdout = Logger(os.path.join(log_dir, 'log.log'))
    start = time.time()

    # -----------------------------
    
    print(f">>>>    Running on input feature {input_feature} \ model name {model_name} \ dataset {dataset}      >>>>>")
    print("******************************************************")

    print(">> Loading base dataset")
    base_dir = os.path.join(dataset_base_path, dataset) # "/content/drive/My Drive/Thesis_Final/datasets/Affectnet-WM-RR-sample"
    target_dir = os.path.join(model_base_path, dataset) #"/content/drive/My Drive/Thesis_Final/models/Affectnet-WM-RR-sample"
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)


    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    train_set = ImageFolder(root=train_dir, transform=train_transformations)
    val_set = ImageFolder(root=val_dir, transform=valid_transformations)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)


    # training and evaluation
    # -----------------------
    best_valid = dict.fromkeys(['acc', 'rec', 'f1', 'aucpr', 'aucroc'], 0.0)

    #### Loading model
    print(">> Loading model")
    # Load torch.hub: IMAGENET1K_V2 load
    model_path = os.path.join(pretrained_base_path, input_feature, model_name + '.pt') # /content/drive/My Drive/Thesis_Final/pretrained_models/WM/enet_mtl.pt"
    model = None
    if os.path.isfile(model_path):
      model = torch.load(model_path, map_location=device)
      print(f"Load model {model_path} with pre-trained weights with deivce")
    else:
      try:
        model = torchvision.models.get_model(model_name, weights="IMAGENET1K_V2").to(device)
        print(f"Load model {model_name} with IMAGENET1K_V2 pre-trained weights with deivce")
      except:
        try:
          model = torchvision.models.get_model(model_name, weights="IMAGENET1K_V1").to(device)
          print(f"Load model {model_name} with IMAGENET1K_V1 pre-trained weights with device")
        except:
          raise Exception("Cannot load the model")

    # Change model with new classifier
    try:
      model.fc = CustomClassifier(model.fc.in_features)
      print("Change new classifier at fc layer")
    except:
      model.classifier = CustomClassifier(model.classifier[0].in_features)
      print("Change new classifier at classifier layer")

    
    
    
    #### Setup vairable
    print(">> Loading optimizer, criterition, logger")
    # define optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Adam 
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, min_lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # define criterion
    criterion = nn.CrossEntropyLoss()
    

    print('[>] Begin Fine tuning '.ljust(64, '-'))
    freeze_model(model)
    for epoch in range(1, ft_num_epoch + 1):

        start_ = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, ft_num_epoch)
        # validate for one epoch
        validate(val_loader, model, criterion, epoch, ft_num_epoch)

        # progress
        end_ = time.time()
        progress = (
            f'[*] epoch time = {end_ - start_:.2f}s | '
            f'lr = {optimizer.param_groups[0]["lr"]}\n'
        )
        print(progress)

        # lr step
        scheduler.step()


    print('[>] Begin Training '.ljust(64, '-'))
    unfreeze_model(model)
    for epoch in range(1, num_epoch + 1):

        start_ = time.time()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, num_epoch)
        # validate for one epoch
        validate(val_loader, model, criterion, epoch, num_epoch)

        # progress
        end_ = time.time()
        progress = (
            f'[*] epoch time = {end_ - start_:.2f}s | '
            f'lr = {optimizer.param_groups[0]["lr"]}\n'
        )
        print(progress)

        # lr step
        scheduler.step()

    # best valid info
    # ---------------
    print('[>] Best Valid '.ljust(64, '-'))
    stat = (
        f'[+] acc={best_valid["acc"]:.4f}\n'
        f'[+] rec={best_valid["rec"]:.4f}\n'
        f'[+] f1={best_valid["f1"]:.4f}\n'
        f'[+] aucpr={best_valid["aucpr"]:.4f}\n'
        f'[+] aucroc={best_valid["aucroc"]:.4f}'
    )
    print(stat)



    # -----------------------------
    end = time.time()

    print('\n[*] Fini! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()



import argparse
parser = argparse.ArgumentParser(description='DACL for FER in the wild')
parser.add_argument('--ft', required=True, type=str, help="input feature including WM for without margin or TH for top half face")
parser.add_argument('--mdn', required=True, type=str, help="model name including enet_mtl or vgg16")
parser.add_argument('--ds', required=True, type=str, help="dataset name such as Affectnet-WM-RR")

args = parser.parse_args()

# input_feature: str, model_id: str, dataset: str
main(args.ft, args.mdn, args.ds)  #1
# main("WM", "enet_mtl", "Affectnet-WM-RR")  #1
# python3 main.py --ft WM --mdn enet_mtl --ds Affectnet-WM-RR