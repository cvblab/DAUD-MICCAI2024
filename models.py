import torch
import torch.nn as nn


from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.anogan import AnoGAN
from pyod.models.vae import VAE

    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_input3  = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.ReLU()
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))

        return self.FC_input3(h_)
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = self.FC_output(h)
        return x_hat
    

class AE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(AE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        z = self.Encoder(x)        
        return self.Decoder(z), z

# Implementation utilized for the experiments of the MICCAI paper
class DAUD(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(DAUD, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.latent_dim = 16

        self.mu1 = torch.nn.Parameter(torch.ones(self.latent_dim))
        self.logvar1 = torch.nn.Parameter(torch.zeros(self.latent_dim))

        self.mu2 = torch.nn.Parameter(torch.ones(self.latent_dim))
        self.logvar2 = torch.nn.Parameter(torch.zeros(self.latent_dim))

        self.zeros = torch.zeros(self.latent_dim).to('cuda')
        
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
                
    def forward(self, x, i):
        z = self.Encoder(x)
        
        dom1 = self.reparameterization(self.mu1, torch.exp(0.5 * self.logvar1))
        dom2 = self.reparameterization(self.mu2, torch.exp(0.5 * self.logvar2))
        
        dom11 = self.reparameterization(self.zeros, self.zeros)
        x_hat11 = self.Decoder(z+dom11)
        x_hat1           = self.Decoder(z+dom1)
        x_hat2            = self.Decoder(z+dom2)
        x_hat = torch.stack([aa if cc==1 else bb if cc==0 else dd for aa, bb, cc, dd in zip(x_hat1, x_hat2, i, x_hat11)])
        x_hat = x_hat.to('cuda')
        return x_hat, self.mu1, self.mu2, self.logvar1, self.logvar2, z

# New implementation to generalize to an arbitrary number of domains (not fully tested)
class DAUD_v2(nn.Module):
    def __init__(self, Encoder, Decoder, num_domains=2, latent_dim=16):
        super(DAUD_v2, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.latent_dim = latent_dim

        self.domain_parameters = nn.ModuleList([
            nn.Parameter(torch.ones(self.latent_dim)),
            nn.Parameter(torch.zeros(self.latent_dim))
        ] * num_domains)

        self.zeros = torch.zeros(self.latent_dim).to('cuda')

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda')        
        z = mean + var * epsilon
        return z

    def forward(self, x, i):
        z = self.Encoder(x)

        domain_latents = [self.reparameterization(mu, torch.exp(0.5 * logvar)) for mu, logvar in self.domain_parameters]

        domain_latents.insert(0, self.reparameterization(self.zeros, self.zeros))

        x_hats = [self.Decoder(z + dom) for dom in domain_latents]

        # Convert domain indices to boolean masks
        masks = [i == domain_index for domain_index in range(len(x_hats))]

        x_hat = x_hats[0]
        for j in range(1, len(x_hats)):
            x_hat = torch.where(masks[j].unsqueeze(1).unsqueeze(2), x_hat, x_hats[j])

        x_hat = x_hat.to('cuda')

        return x_hat, [mu for mu in self.domain_parameters], z

def select_model(model_name):
    # Custom

    if model_name == 'AE':
        encoder = Encoder(input_dim=512, hidden_dim=64, latent_dim=16)
        decoder = Decoder(latent_dim=16, hidden_dim = 32, output_dim = 512)
      
        model = AE(Encoder=encoder, Decoder=decoder)
    
    elif model_name=='DAUD':
        encoder = Encoder(input_dim=512, hidden_dim=64, latent_dim=16)
        decoder = Decoder(latent_dim=16, hidden_dim = 64, output_dim = 512)

        model = DAUD(Encoder=encoder, Decoder=decoder)

    # PyOD
        
    elif model_name=='ECOD':
        model = ECOD()

    
    elif model_name=='DeepSVDD':
        model = DeepSVDD(512)
    
    elif model_name=='AnoGAN':
        model = AnoGAN()

    elif model_name=='VAE':
        model = VAE()

    elif model_name=='beta-VAE':
        model = VAE(beta=0.5)


    else:
        raise('Model not evaluatefound/implemented')

    return model