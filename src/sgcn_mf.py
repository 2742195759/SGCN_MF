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
from utils import calculate_auc, setup_features
from sklearn.model_selection import train_test_split
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule
from sgcn import SignedGraphConvolutionalNetwork
from Modified_MF import Modified_MF
from evaluate import TopkEvaluate
from utils import gather_2dim_list
import os


class Evaluate(TopkEvaluate) :
    def dataset2recmap(self , dataset) : 
        '''
            dataset is { } contain 'interaction'
        '''
        inte = dataset['interaction']
        gath = gather_2dim_list(inte , 0)
        for k,v in gath.items() : 
            v = [item[1] for item in v]
            gath[k] = v
        return gath

        

class SGCN_MF(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, traingraph , testgraph):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param graph: Edge data structure with positive and negative graph separated.
        """
        self.args = args
        self.traingraph = traingraph 
        self.testgraph = testgraph 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logs()
        self.graph = self.traingraph

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
        self.evaluate = Evaluate(self.args)

        
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


    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number. 
        """
        pre , recall = self.evaluate.accurate(self.second_model , self.traingraph , self.testgraph , retain=True)
        self.logs["performance"].append([epoch+1, pre , recall])
        return pre , recall

    def regular_loss(self ) : #TODO
        return 0

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
            loss  = self.regular_loss() + loss1 + loss2
            loss.backward()
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1,time.time()-start_time])
            if self.args.test_size >0:
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
