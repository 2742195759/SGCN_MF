import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Modified_MF(torch.nn.Module):
    def __init__(self , args , nu , ni , nf) :
        """
        Init Function : 
        :parameter args 
        :parameter nu , the number of user
        :parameter ni , the number of item
        :parameter nf , the number of features
        """
        super(Modified_MF, self).__init__()
        self.args = args
        self.uY = Parameter(torch.Tensor(nu, self.args.ydivx * self.args.dimEmbedding))
        self.iY = Parameter(torch.Tensor(ni, self.args.ydivx * self.args.dimEmbedding))
        init.uniform_(self.uY , 0 , 1)
        init.uniform_(self.iY , 0 , 1)
        self.nu = nu 
        self.ni = ni

    def forward(self , Z , interaction) : 
        """
        Model Modified_MF , 
        :param Z , the embedding of the SGCN
        :param interaction , the array of (userid , itemid , rating)  [verified]
        :note all the userid / itemid must start with zero seperate  
        """
        nu = self.nu 
        ni = self.ni
        import pdb
        pdb.set_trace()
        self.latentu = torch.cat([Z[0:nu] , self.uY] ,dim = 1) ; 
        self.latenti = torch.cat([Z[nu:ni+nu] , self.iY] ,dim = 1) ; 

        u = interaction[:,0]
        i = interaction[:,1]
        r = torch.from_numpy(interaction[:,2]).type(torch.float)
        r_hat = torch.sum(self.latentu[u,:] * self.latenti[i,:] , dim=1)
        loss = torch.mean((r - r_hat) ** 2) 
        return loss
        
    def get_topk(self) : 
        score = torch.matmul(self.latentu , self.latenti.t())
        _ , indices = torch.sort(score , -1 , descending=True)
        return indices
