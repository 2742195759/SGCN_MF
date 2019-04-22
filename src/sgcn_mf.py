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


class Evaluate(TopkEvaluate) :
    def dataset2recmap(self , dataset) : 
        inte = dataset['interaction']
        return gather_2dim_list(inte , 0)

        

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

        
        # use the X to embedding the result XXX
        self.X = torch.rand((self.graph['ncount'] , self.args.dimnode) , requires_grad=True)
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

        self.y = np.array([0]*len(self.positive_edges)+[1]*len(self.negative_edges)+[2]*(self.ecount))
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)

        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)


    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number. 
        """
        pre , racall = self.evaluate.accurate(self.second_model , self.traingraph , self.testgraph , retain=True)
        self.logs["performance"].append([epoch+1, pre , recall])

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.sgcn = SignedGraphConvolutionalNetwork(self.device, self.args, self.X).to(self.device)

        # XXX MF and BPR
        self.second_model = Modified_MF(self.args , self.graph['ncount'])

        self.optimizer = torch.optim.Adam(self.sgcn.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.sgcn.train()
        self.second_model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            loss, _ = self.sgcn(self.positive_edges, self.negative_edges, self.y ,self.pos_adj , self.neg_adj)
            loss = loss * self.args.super_mu + (1-self.args.super_mu)*self.second_model(self.sgcn.z , np.array(self.graph["interaction"] , dtype='int32'))
            loss.backward()
            self.epochs.set_description("SGCN (Loss=%g)" % round(loss.item(),4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1,time.time()-start_time])
            if self.args.test_size >0:
                self.score_model(epoch)

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
