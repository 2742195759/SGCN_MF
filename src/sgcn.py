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
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule, ItemActivate
from collections import OrderedDict

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, device, args, X, nu, ni):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        self.device = device
        self.X = Parameter(X)
        self.nu = nu
        self.ni = ni
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers and Regression Parameters if the model is not a single layer model.


        visualize the parameter of every layer , and no problem.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = [None, None]
        self.negative_base_aggregator = [None, None]
        for j in range(0,2):
            self.positive_base_aggregator[j] = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
            self.negative_base_aggregator[j] = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
        self.positive_aggregators = [[],[]] # 0 -- iu , 1 -- if
        self.negative_aggregators = [[],[]] # 0 -- iu , 1 -- if
        self.item_activaters = []           # item activate
        self.item_activaters.append(ItemActivate(self.neurons[0], self.neurons[0]).to(self.device))
        for i in range(1,self.layers):
            for j in range(0,2):
                self.positive_aggregators[j].append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
                self.negative_aggregators[j].append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
            self.item_activaters.append(ItemActivate(self.neurons[i], self.neurons[i]).to(self.device))

        for j in range(0,2):
            self.positive_aggregators[j] = ListModule(*self.positive_aggregators[j])
            self.negative_aggregators[j] = ListModule(*self.negative_aggregators[j])
        self.item_activaters = ListModule(*self.item_activaters)

        deeplayer = OrderedDict()
        assert(isinstance(self.args.deep_neurons , list))
        assert(self.args.deep_neurons[-1] == 1)
        self.args.dimEmbedding = self.args.layers[-1]*3
        self.args.deep_neurons.insert(0 , self.args.dimEmbedding*2) # output of one Linear Layer
        for i in range(1,len(self.args.deep_neurons)) :
            linear = torch.nn.Linear(self.args.deep_neurons[i-1] , self.args.deep_neurons[i]).to(self.device)
            init.xavier_normal_(linear.weight)
            deeplayer[str(i)] = linear 
            deeplayer[str(i)+'tanh'] = torch.nn.Tanh().to(self.device)
        self.deep = torch.nn.Sequential(deeplayer).to(self.device)
            
    def calculate_loss_xk_func (self, z, hyper_edge):
        """ calculate the loss by xk version

            this version use the dot product as the similarity, and then the similarity is used to 
            predict the edge type. the more similarity, the more like and then the edge will be a 
            positive edge, vice versa
        """ 
        label = hyper_edge[:,-1].type(torch.float).to(self.device)  #XXX label may be too small?
        label[label<0] = 0
        user_vec = z[hyper_edge[:,0].type(torch.long),:]
        item_vec = z[hyper_edge[:,1].type(torch.long),:]
        posi = torch.sigmoid(torch.sum(user_vec * item_vec, dim=1)).to(self.device)
        #posi = torch.squeeze(self.deep(deep_input) / 2 + 0.5) # make it between [0,1] XXX ASK

        loss = torch.log(posi * label + (1-posi)*(1-label))
        loss = -loss.sum()
        return loss

 
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

    def _get_true_embedding(self, iu_emb, if_emb, index):
        ''' A proxy function that calculate the item embedding using the 2 sub_graph embedding
            
            because the iu_emb, and if_emb is not correct for item, because item need another process
        '''
        nu, ni = self.nu, self.ni
        item_emb = self.item_activaters[index](if_emb[nu:nu+ni,:], iu_emb[nu:nu+ni,:])
        assert(item_emb.shape[0] == self.ni)
        assert(item_emb.shape[1] == iu_emb.shape[1])
        _iu_emb = torch.cat([iu_emb[0:self.nu,:], item_emb, if_emb[self.nu+self.ni:,:]], dim=0) # the same 
        _if_emb = torch.cat([iu_emb[0:self.nu,:], item_emb, if_emb[self.nu+self.ni:,:]], dim=0) # the same
        return _iu_emb, _if_emb

    def _get_final_embedding(self, true_iu_emb, true_if_emb):
        ''' the item embedding is the same.
        '''
        nu, ni = self.nu, self.ni
        return torch.cat([true_iu_emb[0:nu+ni,:], true_if_emb[nu+ni:,:]], dim=0)

    def forward(self, positive_edges, negative_edges, pos_adj, neg_adj):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param pos_adj , neg_adj : is the adjacency matrix of the graph . TYPE : torch.sparse.FloatTensor()
        :return self.z: Hidden vertex representations.
        """
        assert(isinstance(pos_adj, list))
        assert(isinstance(neg_adj, list))
        assert(len(neg_adj)==2)
        assert(len(pos_adj)==2)
        self.h_pos, self.h_neg = [[],[]] , [[],[]]  # 0 -- nu , 1 -- nf

        for j in range(0,2): 
            before_active_embedding_pos=self.positive_base_aggregator[j](self.X, positive_edges, pos_adj[j])
            before_active_embedding_neg=self.negative_base_aggregator[j](self.X, negative_edges, neg_adj[j])
            self.h_pos[j].append(before_active_embedding_pos)
            self.h_neg[j].append(before_active_embedding_neg)
        self.h_pos[0][0], self.h_pos[1][0] = self._get_true_embedding(self.h_pos[0][0], self.h_pos[1][0], 0)
        self.h_neg[0][0], self.h_neg[1][0] = self._get_true_embedding(self.h_neg[0][0], self.h_neg[1][0], 0)
        for j in range(0,2):
            self.h_pos[j][0] = torch.tanh(self.h_pos[j][0])
            self.h_neg[j][0] = torch.tanh(self.h_neg[j][0])

        for i in range(1,self.layers):
            for j in range(0,2):
                t_pos = (self.positive_aggregators[j][i-1](self.h_pos[j][i-1],self.h_neg[j][i-1], positive_edges, negative_edges , pos_adj[j] , neg_adj[j]))
                t_neg = (self.negative_aggregators[j][i-1](self.h_neg[j][i-1],self.h_pos[j][i-1], positive_edges, negative_edges , pos_adj[j] , neg_adj[j]))
                self.h_pos[j].append(t_pos)
                self.h_neg[j].append(t_neg)

            self.h_pos[0][i], self.h_pos[1][i] = self._get_true_embedding(self.h_pos[0][i], self.h_pos[1][i], i)
            self.h_neg[0][i], self.h_neg[1][i] = self._get_true_embedding(self.h_neg[0][i], self.h_neg[1][i], i)
            for j in range(0,2):
                self.h_pos[j][i] = torch.tanh(self.h_pos[j][i])
                self.h_neg[j][i] = torch.tanh(self.h_neg[j][i])

        self.z = torch.cat([
            self._get_final_embedding(self.h_pos[0][-1], self.h_pos[1][-1]), 
            self._get_final_embedding(self.h_neg[0][-1], self.h_neg[1][-1])
            ], 1).to(self.device)

        assert(self.z.shape == (self.X.shape[0], self.neurons[-1]*2)) # only for this testcase

        return self.z
