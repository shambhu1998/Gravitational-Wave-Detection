import numpy as np
import torch
from nnAudio.Spectrogram import CQT1992v2
from PIL import Image 
from sklearn.preprocessing import MinMaxScaler
import config


def idx2path(idx, is_train=True):
    path = config.data_dir
    
    if is_train:
        path += 'train/' + idx[0] +'/'+ idx[1] +'/'+ idx[2] +'/'+ idx +'.npy'
    else:
        path += 'test/' + idx[0] +'/'+ idx[1] +'/'+ idx[2] +'/'+ idx +'.npy'
    return path 

def increase_dimension(path,transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)): # in order to use efficientnet we need 3 dimension images
    data = np.load(path)
    d1 = torch.from_numpy(data[0]).float()
    d2 = torch.from_numpy(data[1]).float()
    d3 = torch.from_numpy(data[2]).float()
    d1 = transform(d1)
    d2 = transform(d2)
    d3 = transform(d3)
    img = np.zeros([d1.shape[1], d1.shape[2], 3], dtype=np.uint8)
    scaler = MinMaxScaler()
    img[:,:,0] = 255*scaler.fit_transform(d1.reshape(d1.shape[1],d1.shape[2]))
    img[:,:,1] = 255*scaler.fit_transform(d2.reshape(d2.shape[1],d2.shape[2]))
    img[:,:,2] = 255*scaler.fit_transform(d3.reshape(d3.shape[1],d3.shape[2]))
    return Image.fromarray(img).rotate(90, expand=1).resize((256,256))
    