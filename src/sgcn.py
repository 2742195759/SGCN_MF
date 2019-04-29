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
from collections import OrderedDict

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, device, args, X):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = Parameter(X)
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers and Regression Parameters if the model is not a single layer model.


        visualize the parameter of every layer , and no problem.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1,self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        deeplayer = OrderedDict()
        assert(isinstance(self.args.deep_neurons , list))
        assert(self.args.deep_neurons[-1] == 1)
        self.args.dimEmbedding = self.args.layers[-1]*3
        self.args.deep_neurons.insert(0 , self.args.dimEmbedding*2) # output of one Linear Layer
        for i in range(1,len(self.args.deep_neurons)) :
            linear = torch.nn.Linear(self.args.deep_neurons[i-1] , self.args.deep_neurons[i]).to(self.device)
            init.xavier_normal_(linear.weight)
            deeplayer[str(i)] = linear 
            deeplayer[str(i)+'tanh'] = torch.nn.Tanh()
        self.deep = torch.nn.Sequential(deeplayer)
            
 
    def calculate_loss_function(self, z, hyper_edge): 
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param hyper_edge: the Hyperedge of train_set , use the gathered id from 0  [ [uid , iid , fid , +/-1] ]
        """
        # customed loss function
        label = hyper_edge[:,-1].type(torch.float)
        label[label<0] = 0
        deep_input = torch.cat((z[hyper_edge[:,0].type(torch.long),:] , z[hyper_edge[:,1].type(torch.long),:] , z[hyper_edge[:,2].type(torch.long),:]) , dim=-1)
        posi = torch.squeeze(self.deep(deep_input) / 2 + 0.5) # make it between [0,1] XXX ASK

        loss = torch.log(posi * label + (1-posi)*(1-label))
        loss = -loss.sum()
        return loss


    def forward(self, positive_edges, negative_edges, pos_adj , neg_adj):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [] , []

        connected = neg_adj.numpy() + pos_adj.numpy() + np.eye((self.X.shape[0]))
        self.noconnected = {}
        for i , vec in enumerate(connected) : 
            self.noconnected[i] = np.where(vec<0.5)[0].tolist()
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges , pos_adj)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges , neg_adj)))
        for i in range(1,self.layers):
            t_pos = torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1],self.h_neg[i-1], positive_edges, negative_edges , pos_adj , neg_adj))
            t_neg = torch.tanh(self.negative_aggregators[i-1](self.h_neg[i-1],self.h_pos[i-1], positive_edges, negative_edges , pos_adj , neg_adj))
            self.h_pos.append(t_pos)
            self.h_neg.append(t_neg)
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        return self.z
