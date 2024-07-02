import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import label_binarize
import random

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def get_all_emb(folder):
    embd = [np.load(os.path.join(folder, f_name ), allow_pickle=True) for f_name in os.listdir(folder)]
    return [np.mean(patch,axis=0) for patch in tqdm(embd)]

def get_emb_av(values, folder):
    embd_V = [np.load(os.path.join(folder, f_name + ".npy"), allow_pickle=True) for f_name in tqdm(values)]
    return [np.mean(patch,axis=0) for patch in tqdm(embd_V)]

def loader(features, dom, DEVICE, set_='train'):
    features = torch.Tensor(features)
    dataset = TensorDataset(features,dom)
    if set_=='train':        
        data_loader=DataLoader(dataset, batch_size=32, shuffle=True)
    else:
        data_loader=DataLoader(dataset, batch_size=32, shuffle=False)
    
    return data_loader

def calculate_metrics(y_test, score):
    fpr, tpr, _ = roc_curve(y_test, score)
    return auc(fpr, tpr)

def evaluate_anomaly_classifier(scores, bin_labels, true_labels):
    # Initialize variables to store AUC scores and fpr/tpr values for each subclass
    auc_scores = []

    # Evaluate the AUC for each subclass
    for i in [0,2]:
        target_labels = bin_labels[true_labels!=i]
        target_scores = scores[true_labels!=i]
        # Compute ROC curve and AUC for the current subclass
        fpr, tpr, _ = roc_curve(target_labels, target_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        print('AUC for Subclass {}: {:.2f}'.format(i, roc_auc))
    
    for i in [1,3,4,5]:
        indexes = np.where(np.logical_or(np.logical_or(true_labels == 0, true_labels == 2), true_labels==i))[0]
        target_labels = bin_labels[indexes]
        target_scores = scores[indexes]
        # Compute ROC curve and AUC for the current subclass
        fpr, tpr, _ = roc_curve(target_labels, target_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        print('AUC for Subclass {}: {:.2f}'.format(i, roc_auc))

    return auc_scores


def avg_std(list1, list2):
    return np.mean(list1), np.std(list2)

def print_nbs(A_outlier_1, A_outlier_0, B_outlier_1, B_outlier_0):
    print("[INFO] Dataset from A")
    print('Nb test samples anomalies: ')
    print(len(A_outlier_1))
    print('Nb test samples normals: ')
    print(len(A_outlier_0))
    print("[INFO] Dataset from B")
    print('Nb test samples anomalies: ')
    print(len(B_outlier_1))
    print('Nb test samples normals: ')
    print(len(B_outlier_0))


def plot_data_(plt_name, save_name, percentages, metrics, hosp):
    fmt_ = {'AE': 'x-', 'DAUD': 'o-'}
    for model_name in ['AE', 'DAUD']:
        plt.errorbar(percentages, metrics[model_name]['B'][0], yerr=metrics[model_name]['B'][1]*0.61, fmt=fmt_[model_name], label=model_name+'-B')  # Coral
        plt.errorbar(percentages, metrics[model_name]['A'][0], yerr=metrics[model_name]['A'][1]*0.61, fmt=fmt_[model_name], label=model_name+'-A')  # Gold
    
    if hosp=='A':
        hosp2='B'
    elif hosp=='B':
        hosp2='A'
    else:
        raise("Error")
    
    plt.title('Training with Hosp. ' + hosp, fontsize=16)
    plt.xlabel('Percentage of samples from Hosp. ' + hosp2, fontsize=16)
    plt.ylabel('AUC values')
    plt.grid(True)
    plt.legend(fontsize=11)
    plt_name = 'hosp' + hosp + '.pdf'
    plt.savefig(plt_name, format='pdf')
    plt.show()


def save_data(percentages, metrics, save_name):
    np.save('results/percentages', percentages)
    np.save('results/' + save_name + 'metrics.npy', metrics)