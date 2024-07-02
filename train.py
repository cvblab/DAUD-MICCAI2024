import torch
from torch.optim import Adam, SGD
import torch.nn as nn

def loss_function_DAUD(x, x_hat, mu1, mu2, logvar1, logvar2):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
    KLD1      = - 0.5 * torch.sum(1+ logvar1 - mu1.pow(2) - logvar1.exp())
    KLD2      = - 0.5 * torch.sum(1+ logvar2 - mu2.pow(2) - logvar2.exp())

    return reproduction_loss + KLD1 + KLD2


def training_loop(model, train_loader, model_name, DEVICE):
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(200):
        overall_loss = 0
        for x, dom in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()

            if model_name == 'AE':
                x_hat, _ = model(x)
                loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
            
            elif model_name == 'DAUD':
                x_hat, mu1, mu2, logvar1, logvar2, _ = model(x,dom)
                loss = loss_function_DAUD(x, x_hat, mu1, mu2, logvar1, logvar2)
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
    
    return model