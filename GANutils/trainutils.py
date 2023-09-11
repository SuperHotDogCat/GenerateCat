import torch
from torch import nn
from torch import optim 
#入力zを制作する関数
def create_input(batch_size, z_size, mode = "uniform"):
    if mode == "uniform":
        input_z = torch.rand(batch_size,z_size) * 2 - 1 #[-1,1]の値がでてくる
    else:
        input_z = torch.randn(batch_size,z_size)
    return input_z

def trainGAN(z_size: int, real_data: torch.Tensor, G: nn.Module, D: nn.Module, optimizerG: optim.Optimizer, optimizerD: optim.Optimizer,device):
    #まず識別器のモデルを改善した後に生成器のモデルを改善する
    criterion = nn.BCELoss()
    optimizerD.zero_grad()
    batch_size = real_data.size(0)
    real_label = torch.ones(batch_size,1).to(device)
    real_proba = D(real_data) #データの形状が合わない時はforward内部で入力ように整形変更することを想定して制作。
    D_loss_real = criterion(real_proba, real_label)
    input_z = create_input(batch_size,z_size).to(device)
    fake_data = G(input_z)
    fake_proba = D(fake_data)
    fake_label = torch.zeros(batch_size,1).to(device)
    D_loss_fake = criterion(fake_proba,fake_label)
    D_loss = D_loss_fake + D_loss_real
    D_loss.backward()
    optimizerD.step()
    #D_lossのところでGenerate Modelのパラメータの勾配も計算されている為、Generate Modelに関する勾配はここで初期化を行う
    optimizerG.zero_grad()
    input_z = create_input(batch_size,z_size).to(device)
    fake_data = G(input_z)
    fake_proba = D(fake_data)
    real_label = torch.ones(batch_size,1).to(device) #ラベルが1になるように(騙すように)訓練するので、ラベルは1を予測するように作る
    G_loss = criterion(fake_proba,real_label)
    G_loss.backward()
    optimizerG.step()
    return D_loss.detach().item(), G_loss.detach().item()

def trainWGAN(z_size: int, real_data: torch.Tensor, G: nn.Module, D: nn.Module, optimizerG: optim.Optimizer, optimizerD: optim.Optimizer,lambda_gp,device):
    #まず識別器のモデルを改善した後に生成器のモデルを改善する
    optimizerD.zero_grad()
    batch_size = real_data.size(0)
    real_data = real_data
    real_proba = D(real_data)
    D_loss_real = -real_proba.mean()
    input_z = create_input(batch_size,z_size).to(device)
    fake_data = G(input_z)
    fake_proba = D(fake_data)
    D_loss_fake = fake_proba.mean()
    #勾配ペナルティの計算
    alpha = torch.rand(*real_data.shape, requires_grad=True, device = device)
    interpolated_data = alpha * real_data + (1-alpha) * fake_data
    interpolated_proba = D(interpolated_data)
    grad_D = torch.autograd.grad(outputs=interpolated_proba, inputs=interpolated_data, grad_outputs=torch.ones(interpolated_proba.size(),device=device)\
        ,create_graph=True, retain_graph=True)[0]
    grad_D = grad_D.view(batch_size,-1)
    D_loss_gradient = lambda_gp*((grad_D.norm(2,dim=1))**2).mean()
    D_loss = D_loss_fake + D_loss_real + D_loss_gradient
    D_loss.backward()
    optimizerD.step()

    
    #D_lossのところでGenerate Modelのパラメータの勾配も計算されている為、Generate Modelに関する勾配はここで初期化を行う
    optimizerG.zero_grad()
    input_z = create_input(batch_size,z_size).to(device)
    fake_data = G(input_z)
    fake_proba = D(fake_data)
    G_loss = -fake_proba.mean()
    G_loss.backward()
    optimizerG.step()
    return D_loss.detach().item(), G_loss.detach().item()