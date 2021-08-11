import torch
from torch._C import dtype 
import torch.nn as nn


def train(data_loader, model, optimizer, device):
    # put the model in train mode 
    model.train()

    # go over every batch in data loader
    for data in data_loader:
        # we have image and target in data loader
        inputs = data['image']
        targets = data['target']
        # move inputs and targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        # zero grad the optimizer
        optimizer.zero_grad()
        # do the forward steps of model
        outputs = model(inputs)
        # calculate the loss 
        loss = nn.BCEWithLogitsLoss()(outputs,targets)
        # backward step the loss
        loss.backward()
        # step optimzer
        optimizer.step()
    
def evaluate(data_loader, model, device):
    # put the model into evaluation mode 
    model.eval()
    # init the list to targets and outputs
    final_targets = []
    final_outputs = []

    # we use no_grad context 
    with torch.no_grad():

        for data in data_loader:
            inputs = data['image']
            targets = data['target']
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # do the forward stepd to generate prediction
            output  = model(inputs)

            # convert targets and outputs to lists
            targets = targets.detach().cpu().numpy().tolist()
            output = output.output.detach().cpu().numpy().tolist()

            # extend the original list
            final_targets.extend(targets)
            final_outputs.extend(output)

    return final_outputs, final_targets
