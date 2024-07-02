import torch
import numpy as np    

def evaluate(model, model_name, test, DEVICE):
    model.eval()

    recon = []
    with torch.no_grad():
        for x, dom in test:
            x = x.to(DEVICE)

            if model_name == 'AE':
                x_hat, _ = model(x)

            elif model_name == 'DAUD':
                x_hat, _, _, _, _, _  = model(x, dom)
            
            scores = torch.mean((x-x_hat)**2, axis=1)
            recon.extend(scores.detach().cpu().numpy())

    return np.array(recon)
    
def representation(model, model_name, test, DEVICE):
    model.eval()

    rep = []
    with torch.no_grad():
        for x, dom in test:
            x = x.to(DEVICE)
            
            if model_name == 'VAE':
                x_hat, _, _ = model(x)

            elif model_name == 'AE':
                x_hat, z = model(x)

            elif model_name == 'DAUD':
                x_hat, _, _, _, _, z  = model(x, dom)
            
            
            rep.extend(z.detach().cpu().numpy())

    return np.array(rep)
