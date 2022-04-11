import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def get_logits(model, x_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        output = model(x.cuda())
    
    return output.cpu().numpy()

def get_logits_norm(model, x_nat):
    
    # ============================
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ============================
    
    # step 1: transpose/permute dimension
    # input must be [n,C,W,H] due to normalization and model's input is in that format
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float() # input [n,W,H,C] => [n,C,W,H]
    #x = torch.from_numpy(x_nat).permute(0, 2, 3, 1).float()
    
    # step 2: normalize input to fit model trained under normalized data
    with torch.no_grad():
        norm_img = normalize(x[0])
        x = torch.unsqueeze(norm_img, 0)
        output = model(x.cuda())
    
    return output.cpu().numpy()

def get_predictions(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)
    with torch.no_grad():
        output = model(x.cuda())
    
    return (output.cpu().max(dim=-1)[1] == y).numpy()

def get_predictions_norm(model, x_nat, y_nat,mu,std):
    # ============================
    #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize(mu, std)
    # ============================
    y = torch.from_numpy(y_nat)
    #print(x_nat.shape)
    
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    #x = torch.from_numpy(x_nat).permute(0, 2, 3, 1).float()
    #print(x.shape)
    
    with torch.no_grad():
        norm_img = normalize(x[0])
        x = torch.unsqueeze(norm_img, 0)
        
        output = model(x.cuda())
    #print('adv label:',output.cpu().max(dim=-1)[1])
    return (output.cpu().max(dim=-1)[1] == y).numpy()

def get_predictions_norm_target(model, x_nat, y_nat,mu,std):
    # ============================
    #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize(mu, std)
    # ============================
    y = torch.from_numpy(y_nat)
    #print(x_nat.shape)
    
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    #x = torch.from_numpy(x_nat).permute(0, 2, 3, 1).float()
    #print(x.shape)
    
    with torch.no_grad():
        norm_img = normalize(x[0])
        x = torch.unsqueeze(norm_img, 0)
        
        output = model(x.cuda())
    #print('adv label:',output.cpu().max(dim=-1)[1])
    return (output.cpu().max(dim=-1)[1] != y).numpy()

def get_predictions_and_gradients(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)

    with torch.enable_grad():
        output = model(x.cuda())
        loss = nn.CrossEntropyLoss()(output, y.cuda())

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()

    return pred, grad

def get_predictions_and_gradients_norm(model, x_nat, y_nat,mu,std):
    
    # ============================
    #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize(mu, std)
    # ============================
    
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    #x = torch.from_numpy(x_nat).permute(0, 2, 3, 1).float()
    
    with torch.no_grad():
        norm_img = normalize(x[0])
        x = torch.unsqueeze(norm_img, 0)
    
    x.requires_grad_()
    y = torch.from_numpy(y_nat)

    with torch.enable_grad():
        output = model(x.cuda())
        loss = nn.CrossEntropyLoss()(output, y.cuda())
    
    #print('loss:',loss.item())
    
    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()
    #grad = grad.detach().numpy()
    #print('loss:',loss.item(),'; Marginal:',(output.detach().max().item()-output[:,y].item()),'; top pred:',output.detach().cpu().max(dim=-1)[1])

    pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()

    return pred, grad

def get_predictions_and_gradients_norm_target(model, x_nat, y_nat,mu,std):
    
    # ============================
    #normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    normalize = transforms.Normalize(mu, std)
    # ============================
    
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    #x = torch.from_numpy(x_nat).permute(0, 2, 3, 1).float()
    
    with torch.no_grad():
        norm_img = normalize(x[0])
        x = torch.unsqueeze(norm_img, 0)
    
    x.requires_grad_()
    y = torch.from_numpy(y_nat)
    with torch.enable_grad():
        output = model(x.cuda())
        #loss = nn.CrossEntropyLoss()(output, y.cuda())
        loss = 2*output[:,y].sum() - output.sum()
    
    #print('loss:',loss.item(),'target class:',y,output.shape)
    #print('loss:',loss.item(),'; Marginal:',(output.detach().max().item()-output[:,y].item()),'; top pred:',output.detach().cpu().max(dim=-1)[1])
    
    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()
    #grad = grad.detach().numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] != y).numpy() #if still different from target one

    return pred, grad


