import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import torch.nn.functional as F
from torch import nn
import config

config = config.config

def calculate_mse_rmse(outputs, target):
    outputs_softmax = F.softmax(outputs, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=2).squeeze()
    mse_loss = F.mse_loss(outputs_softmax, target_one_hot.float())
    rmse = torch.sqrt(mse_loss)
    return mse_loss.item(), rmse.item()



def random_flip(img, horizontal_prob=0.5, vertical_prob=0.5):
    # Flip the image horizontally
    if np.random.rand() < horizontal_prob:
        img = img[:, :, ::-1]
    # Flip pattern vertically
    if np.random.rand() < vertical_prob:
        img = img[:, ::-1, :]

    return img

# Draw AUC curve
def drawAUC_TwoClass(y_true, y_score, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc * 100

    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', linestyle='-', linewidth=2,
             label=('CNN (' + str(path).split('.')[0] + ' = %0.2f %%)' % roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(direction='in', top=True, bottom=True, left=True, right=True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # print("AUC:",roc_auc)
    plt.savefig(path, format='png')
    plt.close()


def calculate_f1_score(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def calculate_auc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr) * 100
    return roc_auc


def plot_and_save(path, train_acc_list, train_loss_list, train_f1_list, val_acc_list, val_loss_list, val_f1_list):
    epochs = range(1, len(train_acc_list) + 1)

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, train_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Train Acc')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc')
    plt.savefig(os.path.join(path, 'train_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Train Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.savefig(os.path.join(path, 'train_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Train F1')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Train F1')
    plt.savefig(os.path.join(path, 'train_F1.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, val_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Val Acc')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Val Acc')
    plt.savefig(os.path.join(path, 'val_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Val Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.savefig(os.path.join(path, 'val_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Val F1')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Val F1')
    plt.savefig(os.path.join(path, 'val_F1.png'), format='png')


def plot_save_lsm(path, probs):
    probs = probs.reshape((config["height"], config["width"]))
    # 数据可视化
    plt.figure(dpi=300)  
    plt.imshow(probs, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(path, 'AlexNet_LSM.png'), format='png', dpi=300)  
    plt.show()