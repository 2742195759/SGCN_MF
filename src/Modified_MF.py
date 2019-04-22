import torch
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

class Modified_MF(torch.nn.Module):
    def __init__(self , args , number_node) :
        super(Modified_MF, self).__init__()
        self.args = args
        self.Y = Parameter(torch.Tensor(number_node, self.args.ydivx * self.args.dimnode))

    def forward(self , Z , interaction) : 
        """
        Model Modified_MF , 
        :param interaction , the array of (userid ,itemid , rating)
        """
        self.latent = torch.cat([Z , self.Y] ,dim = 1) ; 
        #import pdb
        #pdb.set_trace()
        u = interaction[:,0]
        i = interaction[:,1]
        r = torch.from_numpy(interaction[:,2]).type(torch.float)
        r_hat = torch.sum(self.latent[u,:] * self.latent[i,:] , dim=1)
        loss = torch.sum((r - r_hat) ** 2) + torch.sum(self.Y ** 2)
        return loss
        
        
    def get_topk(self) : 
        score = torch.matmul(self.latent , self.latent.t())
        _ , indices = torch.sort(score , -1 , descending=False)
        return indices
