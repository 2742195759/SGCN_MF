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
from utils import calculate_auc
from sklearn.model_selection import train_test_split
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from sgcn import SignedGraphConvolutionalNetwork
from utils import gather_2dim_list
from trainer import Trainer
from evaluate import EvaluateModifiedMF
from MF import Modified_MF
from sample import NegativeSample3
import os
        

class SGCN_MF(Trainer):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, traingraph , testgraph):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param graph: Edge data structure with positive and negative graph separated.
        """
        super(SGCN_MF , self).__init__(args , traingraph, testgraph)

    def init(self , args , traingraph , testgraph) : 
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

        self.sgcn = SignedGraphConvolutionalNetwork(self.device, self.args, self.X).to(self.device)
        self.sgcn.train()
        self.mod_mf = Modified_MF(self.args , self.nu , self.ni)
        self.mod_mf.train()

    def getoptimizer(self) : 
        self.optimizer = torch.optim.SGD(self.getparameter() , lr=self.args.learning_rate) # XXX parameters() ?
        return self.optimizer

    def pre_epoch(self , rawtrainset) : 
        self.hyper_edge = self.graph['hyper_edge']
        sample = NegativeSample3(self.nu , self.ni , 1)
        self.trainset = sample.sample(rawtrainset)
        return 1

    def getparameter(self) : 
        return [i for i in self.sgcn.parameters()]+[i for i in self.mod_mf.parameters()]

    def getregulazationloss(self) : 
        ps = self.getparameter()
        loss = torch.Tensor([0]).squeeze()
        for p in ps : 
            loss = loss + (p**2).sum()
        return loss

    def getloss(self , st , ed) : 
        mu = self.args.super_mu
        reg_mu = self.args.weight_decay
        self.Z = self.sgcn(self.positive_edges , self.negative_edges , self.pos_adj , self.neg_adj)
        loss1 = self.sgcn.calculate_loss_function(self.Z , torch.LongTensor(self.hyper_edge))
        loss2 = self.mod_mf.getloss(self.Z , torch.LongTensor(self.trainset))

        print (loss1*mu , loss2 , self.getregulazationloss() * reg_mu)

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


    def setup_dataset(self):
        """
        Creating train and test split.
        """
        self.positive_edges = self.traingraph["positive_edges"]
        self.negative_edges = self.traingraph["negative_edges"]
        self.ecount = len(self.positive_edges + self.negative_edges)

        
        self.X = torch.rand((self.graph['ncount'] , self.args.dimnode)) # use the X to embedding the result XXX
        # use the X to be a node feature
        neg_adj = np.zeros((self.X.shape[0] , self.X.shape[0]) , dtype='Float32')
        pos_adj = np.zeros((self.X.shape[0] , self.X.shape[0]) , dtype='Float32')
        for edge in self.positive_edges : 
            pos_adj[edge[0]][edge[1]] = 1 
            pos_adj[edge[1]][edge[0]] = 1 
        for edge in self.negative_edges : 
            neg_adj[edge[0]][edge[1]] = 1 
            neg_adj[edge[1]][edge[0]] = 1 
        self.neg_adj = torch.from_numpy(neg_adj)
        self.pos_adj = torch.from_numpy(pos_adj)

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)

    def score(self , train , test) : 
        self.evaluater.set_Z(self.Z)
        print (self.Z[0] , self.mod_mf.uY[0])
        print (self.mod_mf.iY[0])
        return self.evaluater.accurate(self.mod_mf , self.rawtrainset , self.rawtestset , retain=True)

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.sgcn = SignedGraphConvolutionalNetwork(self.device, self.args, self.X).to(self.device)

        # XXX MF and BPR
        args = self.args
        self.second_model = Modified_MF(self.args , args.encoder['nu'], args.encoder['ni'] , args.encoder['nf'])
        self.optimizer = torch.optim.Adam([i for i in self.sgcn.parameters()]+[i for i in self.second_model.parameters()], lr=self.args.learning_rate, weight_decay=self.args.weight_decay) # XXX parameters() ?
        self.sgcn.train()
        self.second_model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        hyper_edge = self.graph['hyper_edge']
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            Z = self.sgcn(self.positive_edges, self.negative_edges , self.pos_adj , self.neg_adj)
            #import pdb 
            #pdb.set_trace()
            loss1 = self.sgcn.calculate_loss_function(Z, torch.LongTensor(hyper_edge)) 
            loss2 =  (1-self.args.super_mu)*self.second_model(self.sgcn.z , np.array(self.graph["interaction"] , dtype='int32'))
            loss = self.regular_loss() + loss1 + loss2
            loss.backward()
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1,time.time()-start_time])
            if self.args.test_size > 0 and epoch % 20 == 0:
                print (self.score_model(epoch))
            self.epochs.set_description("SGCN (Loss=%g)" % round(loss.item(),4))

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
