# coding=utf8
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

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
        self.neg2posratio = 1
        mf_lfmdim = self.args.mf_lfmdim
        init.uniform_(self.uY , 0 , 1.0)
        init.uniform_(self.iY , 0 , 1.0)
        self.nu = nu 
        self.ni = ni
        print (self.nu , self.ni)

    #def predict(self , u , i) : 
        #return torch.reshape(torch.sum(self.uY[u,:] * self.iY[ipos,:] , dim=1) , [-1,1])
    
    def forward(self , Tu , Ti) : 
        """
        Model Modified_MF , 
        :param Tu , the Tensorable of userid [ [ userid ] ...]
        :param Ti , the Tensorable of userid [ [ itemid ] ...]
        """
        if not isinstance(Tu , torch.LongTensor) : 
            Tu = torch.LongTensor(Tu)
        if not isinstance(Ti , torch.LongTensor) : 
            Ti = torch.LongTensor(Ti)
        return torch.sum(self.uY[Tu,:] * self.iY[Ti,:] , dim=1)

    def getloss(self , interaction) : 
        nu , ni = self.nu , self.ni
        u , ipos , ineg = interaction[:,0] , interaction[:,1] , interaction[:,2]
        loss = -torch.sum(torch.log(torch.sigmoid(self(u,ipos)-self(u,ineg))))
        loss = loss + (torch.sum(self.uY**2) + torch.sum(self.iY**2)) * self.args.mf_lambda
        return loss


class Modified_MF(torch.nn.Module):
    
    """
    args.dimEmbedding = layers[-1] * 3
    Modified_MF : num of latent factor is (1+self.args.ydivx) * args.dimEmbedding
    """

    def __init__(self , args , nu , ni) :
        """
        Init Function : 
        :parameter args 
        :parameter nu , the number of user
        :parameter ni , the number of item
        :parameter nf , the number of features
        """
        super(Modified_MF, self).__init__()
        self.args = args
        print( "dim Y: ", self.args.dimEmbedding*self.args.ydivx )
        print( "dim Z: ", self.args.dimEmbedding)
        self.uY = Parameter(torch.Tensor(nu, int(self.args.ydivx * self.args.dimEmbedding)))
        self.iY = Parameter(torch.Tensor(ni, int(self.args.ydivx * self.args.dimEmbedding)))
        self.neg2posratio = 1
        init.uniform_(self.uY , 0 , 1.0)
        init.uniform_(self.iY , 0 , 1.0)
        self.nu = nu 
        self.ni = ni

    def forward(self , Z , Tu , Ti) : 
        """
        Model Modified_MF , 
        :param Tu , the Tensorable of userid [ [ userid ] ...]
        :param Ti , the Tensorable of userid [ [ itemid ] ...] 
        NOTE :
            the index of Tu , Ti is from 0 respectly
        """
        if not isinstance(Tu , torch.LongTensor) : 
            Tu = torch.LongTensor(Tu)
        if not isinstance(Ti , torch.LongTensor) : 
            Ti = torch.LongTensor(Ti)
        cu = torch.cat([Z[0:self.nu] , self.uY] ,dim = 1) ; 
        ci = torch.cat([Z[self.nu:self.ni+self.nu] , self.iY] ,dim = 1) ; 
        return torch.sum(cu[Tu,:] * ci[Ti,:] , dim=1)

    def getloss(self , Z , interaction) : 
        """
        parameter Z : the embedding from the sgcn
        parameter inter : the Tensor from the sampled interaction [ [u , posi , negi] ...]
        """
        nu , ni = self.nu , self.ni
        u , ipos , ineg = interaction[:,0] , interaction[:,1] , interaction[:,2]
        loss = -torch.sum(torch.log(torch.sigmoid(self(Z , u ,ipos)-self(Z , u,ineg))))
        loss = loss + (torch.sum(self.uY**2) + torch.sum(self.iY**2)) * self.args.mf_lambda
        return loss

