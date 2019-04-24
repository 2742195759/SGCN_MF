import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from sgcn_mf import Evaluate

class MF(torch.nn.Module):
    def __init__(self , args , nu , ni) :
        """
        Init Function : 
        :parameter args 
        :parameter nu , the number of user
        :parameter ni , the number of item
        :parameter nf , the number of features
        """
        super(MF, self).__init__()
        self.args = args
        self.uY = Parameter(torch.Tensor(nu, self.args.mf_lfmdim))
        self.iY = Parameter(torch.Tensor(ni, self.args.mf_lfmdim))
        init.uniform_(self.uY , 0 , 1)
        init.uniform_(self.iY , 0 , 1)
        self.nu = nu 
        self.ni = ni
        self.evaluate = Evaluate(args)

    def forward(self , interaction) : 
        """
        Model Modified_MF , 
        :param interaction , the array of (userid , itemid , rating)  [verified]
        :note all the userid / itemid must start with zero seperate  
        """
        nu = self.nu 
        ni = self.ni
        self.latentu = self.uY
        self.latenti = self.iY

        u = interaction[:,0]
        i = interaction[:,1]
        r = interaction[:,2].type(torch.float)
        r_hat = torch.sum(self.latentu[u,:] * self.latenti[i,:] , dim=1)
        loss = torch.mean((r - r_hat) ** 2) 
        return loss
        
    def get_topk(self) : 
        score = torch.matmul(self.latentu , self.latenti.t())
        _ , indices = torch.sort(score , -1 , descending=True)
        return indices

    def score(self , train , test) : 
        return self.evaluate.accurate(self , train , test , retain=True)
