import json
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from utils import calculate_auc , scipy_coo2torch_coo
import scipy
from sklearn.model_selection import train_test_split
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from sgcn import SignedGraphConvolutionalNetwork
from utils import gather_2dim_list
from trainer import Trainer
from evaluate import EvaluateModifiedMF
from MF import Modified_MF
from sample import NegativeSample3
import os
from scipy import sparse
        
class SGCN_MF(Trainer):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, traingraph , testgraph):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param graph: Edge data structure with positive and negative graph separated.
        #pdb.set_trace()
        """
        super(SGCN_MF , self).__init__(args , traingraph, testgraph)
        
    def init(self , args , traingraph , testgraph) : 
        # MACRO DEFINED, EMURATE
        self.USER=0
        self.ITEM=1
        self.FEATURE=2
        ## 
        self.traingraph = traingraph 
        self.testgraph = testgraph 
        self.args = args
        self.nu = self.args.encoder['nu']
        self.ni = self.args.encoder['ni']
        self.batchsize = self.args.sgcn_mf_batchsize
        self.rawtrainset = [[item[0] , item[1] , 1] for item in traingraph['interaction']]
        self.rawtestset = [[item[0] , item[1] , 1] for item in testgraph['interaction']]
        self.evaluater = EvaluateModifiedMF(args , self.nu , self.ni)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logs()
        self.graph = self.traingraph

        self.setup_dataset()

        self.sgcn = SignedGraphConvolutionalNetwork(self.device, self.args, self.X, self.nu, self.ni).to(self.device)
        self.sgcn.train()
        print ('modified')
        self.mod_mf = Modified_MF(self.args , self.nu , self.ni).to(self.device)
        self.mod_mf.train()

    def getoptimizer(self) : 
        self.optimizer = torch.optim.Adam(self.getparameter() , lr=self.args.learning_rate) # XXX parameters() ?
        return self.optimizer

    def pre_epoch(self , rawtrainset) : 
        hyper_edge = np.array(self.graph['hyper_edge'])
        np.random.shuffle(hyper_edge)
        self.hyper_edge = hyper_edge.tolist()
        self.trainset = self.hyper_edge
        self.interaction = [[item[0] , item[1]-self.nu , 1] for item in self.hyper_edge]
        
        #print(self.nu)
        #print(self.interaction[1:10] ,'\n\n\n', self.hyper_edge[1:10])

        return None# normal use the step to train the set . 

    def getparameter(self) : 
        return [i for i in self.sgcn.parameters()]+[i for i in self.mod_mf.parameters()]

    def getregulazationloss(self) : 
        ps = self.getparameter()
        loss = torch.Tensor([0]).squeeze().to(self.device)
        for p in ps : 
            loss = loss + (p**2).sum()
        return loss

    def getloss(self , st , ed) : 
        #print (st , ed)
        mu = self.args.super_mu
        reg_mu = self.args.weight_decay
        sample = NegativeSample3(self.nu , self.ni , self.args.ratio_neg_pos)
        self.Z = self.sgcn(self.positive_edges , self.negative_edges, self.pos_adj, self.neg_adj)
        #loss1 = self.sgcn.calculate_loss_function(self.Z , torch.LongTensor(self.hyper_edge[st:ed]))
        loss1 = self.sgcn.calculate_loss_xk_func(self.Z , torch.LongTensor(self.hyper_edge[st:ed]))
        loss2 = self.mod_mf.getloss(self.Z , torch.LongTensor(sample.sample(self.interaction[st:ed])))
        #print (loss1, loss2)

        #print (loss1*mu , loss2 , self.getregulazationloss() * reg_mu)

        loss = loss1*mu + loss2 + self.getregulazationloss() * reg_mu
        return loss
           
    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] =  vars(self.args)
        self.logs["performance"] = [["Epoch","AUC","F1"]]
        self.logs["training_time"] = [["Epoch","Seconds"]]

    def idtype(self, node_id):
        if node_id < self.nu: return self.USER
        if node_id < self.nu+self.ni: return self.ITEM
        return self.FEATURE

    def vote_and_norm(self, adj): #TODO(xiongkun)
        t = adj.sum(axis=1)
        t[t==0] = 1
        t = 1.0 / t
        sparse_t = scipy.sparse.diags(np.array(t).reshape((-1,)))
        adj = sparse_t.dot(adj).dot(sparse_t)
        return adj

    def setup_dataset(self):
        """
        Creating train and test split.
        """
        self.positive_edges = self.traingraph["positive_edges"]
        self.negative_edges = self.traingraph["negative_edges"]
        self.ecount = len(self.positive_edges + self.negative_edges)

        
        self.X = torch.rand((self.graph['ncount'] , self.args.dimnode)) # use the X to embedding the result XXX
        # use the X to be a node feature

        iu_neg_adj = sparse.dok_matrix((self.X.shape[0] , self.X.shape[0]))
        iu_pos_adj = sparse.dok_matrix((self.X.shape[0] , self.X.shape[0]))
        if_neg_adj = sparse.dok_matrix((self.X.shape[0] , self.X.shape[0]))
        if_pos_adj = sparse.dok_matrix((self.X.shape[0] , self.X.shape[0]))
        for edge in self.positive_edges : 
            choosed_adj = None
            #because only have USER-ITEM or ITEM-FEATURE
            f, s = self.idtype(edge[0]), self.idtype(edge[1])
            if f == self.USER or s == self.USER:
                choosed_adj = iu_pos_adj
            elif f == self.ITEM or s == self.ITEM:
                choosed_adj = if_pos_adj
            else: raise RuntimeError("Choosed_adj must be one of the two")
            choosed_adj[edge[0],edge[1]] = 1 
            choosed_adj[edge[1],edge[0]] = 1 
        for edge in self.negative_edges : 
            choosed_adj = None
            #because only have USER-ITEM or ITEM-FEATURE
            f, s = self.idtype(edge[0]), self.idtype(edge[1])
            if f == self.USER or s == self.USER:
                choosed_adj = iu_neg_adj
            elif f == self.ITEM or s == self.ITEM:
                choosed_adj = if_neg_adj
            else: raise RuntimeError("Choosed_adj must be one of the two")
            choosed_adj[edge[0],edge[1]] = 1 
            choosed_adj[edge[1],edge[0]] = 1 
        ### delete the conflict edge -- the vote method
        self.vote_and_norm(iu_neg_adj)
        self.vote_and_norm(iu_pos_adj)
        self.vote_and_norm(if_neg_adj)
        self.vote_and_norm(if_pos_adj)

        self.pos_adj = [None , None]
        self.neg_adj = [None , None]
        self.neg_adj[0], self.neg_adj[1] = scipy_coo2torch_coo(iu_neg_adj).to(self.device), scipy_coo2torch_coo(if_neg_adj).to(self.device)
        self.pos_adj[0], self.pos_adj[1] = scipy_coo2torch_coo(iu_pos_adj).to(self.device), scipy_coo2torch_coo(if_pos_adj).to(self.device)

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)

    def score(self , train , test) : 
        self.evaluater.set_Z(self.Z)
        print (self.Z[0] , self.mod_mf.uY[0])
        print (self.mod_mf.iY[0])
        return self.evaluater.accurate(self.mod_mf , self.rawtrainset , self.rawtestset , retain=True)

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        print("\nEmbedding is saved.\n")
        self.train_z = self.train_z.cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate([np.array(range(self.train_z.shape[0])).reshape(-1,1),self.train_z],axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns = embedding_header)
        self.train_z.to_csv(self.args.embedding_path, index = None)
        print("\nRegression weights are saved.\n")
        self.regression_weights = self.model.regression_weights.cpu().detach().numpy().T
        regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])]
        self.regression_weights = pd.DataFrame(self.regression_weights, columns = regression_header)
        self.regression_weights.to_csv(self.args.regression_weights_path, index = None)     

### for test
if __name__ == '__main__' : 
    import operator as op
    print('start_test')
    recmap = Evaluate.dataset2recmap(None , {'interaction':[[0,1,5],[0,2,3],[1,0,1]]})
    assert(op.eq(recmap[0],[1,2]))
    assert(op.eq(recmap[1],[0]))
    assert(2 not in recmap)
    print('successful')
    os.exit(-1)
