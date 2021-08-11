import os 

import numpy as np
import pandas as pd
from sklearn.utils import shuffle 

import torch 
from torch.utils.data import DataLoader, dataset
from sklearn import metrics
from sklearn.model_selection import train_test_split

import config 
from dataset import G2Dataset
from model import Model
import engine 
import npy2image


if __name__ == "__main__":
    # read the training labels and submission file
    training_labels = pd.read_csv(config.data_dir + 'training_labels.csv')
    sample_submission = pd.read_csv(config.data_dir + 'sample_submission.csv')

    training_labels['path'] = training_labels['id'].apply(npy2image.idx2path)
    train_path, val_path, train_y, val_y = train_test_split(training_labels['path'], training_labels['target'], test_size=0.2)
    train_path, train_y = train_path.reset_index(drop=True), train_y.reset_index(drop=True)
    val_path, val_y = val_path.reset_index(drop=True), val_y.reset_index(drop=True)
    train_dataset = G2Dataset(train_path, train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = G2Dataset(val_path, val_y)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_path = sample_submission['id'].apply(npy2image.idx2path)
    test_dataset = G2Dataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 10

    model = Model()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-3)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        engine.train(train_loader, model, optimizer=optimizer, device=device)
        train_pred, train_target = engine.evaluate(train_loader, model, device=device)
        val_pred, val_target = engine.evaluate(val_loader, model, device=device)
        roc_auc_train = metrics.roc_auc_score(train_target, train_pred)
        roc_auc_val = metrics.roc_auc_score(val_target, val_pred)
        print(f"Epoch{epoch} Train Score: {roc_auc_train} Val score: {roc_auc_val}")
