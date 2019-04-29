from sample import NegativeSample3
from MF import MF
from tqdm import trange
from utils import tab_printer, read_dataset_split_bytime, score_printer, save_logs , build_graph
from evaluate import Evaluate
import torch


class Trainer (object) : 
    '''
        response for the store of input , output
        response for the epoch , step 
        expose a train() method to the main function

        property : 
            rawtrainset , rawtestset : the set passed by main
            trainset , testset       : the preprocessed set
        
        function : 
            preprocessing : the function that called every epoch
    '''
    def __init__ (self, args , trainset , testset) : 
        #torch.manual_seed(self.args.seed)
        self.init(args , trainset , testset)


########################################Train Method
    def train(self) : 
        epochs = trange(self.args.epochs, desc="Loss")
        optimizer = self.getoptimizer()
        for epoch in epochs : 
            self.epoch(epoch)
            tot_loss = 0
            while(self.nextstep()) : 
                #print (self.cntstep)
                optimizer.zero_grad()
                loss = self.step()
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            epochs.set_description("SGCN (Loss=%g)" % round(tot_loss,4))
            if self.args.test_size >0 :
                print (self.score(self.rawtrainset , self.rawtestset))

    def epoch(self , cntepoch , trainset=None) : 
        '''
            trainset must be list [[test]] 
        '''
        self.cntepoch = cntepoch
        if trainset == None : 
            trainset = self.rawtrainset
        self.numstep = self.pre_epoch(trainset) 
        if self.numstep == None : 
            self.numstep = (len(self.trainset) + self.batchsize - 1) // self.batchsize
        self.cntstep = 0

    def nextstep(self) : 
        return self.cntstep != self.numstep

    def step(self) : 
        st = self.cntstep * self.batchsize
        ed = min(st + self.batchsize , len(self.trainset))
        self.cntstep = self.cntstep + 1
        res = self.getloss(st , ed) 
        return res

#################################For Modify
    def score(self , train , test) : 
        return self.evaluater.accurate(self.model , self.rawtrainset , self.rawtestset , retain=True)

    def getloss(self , st , ed) : 
        return self.model.getloss(torch.LongTensor(self.trainset[st:ed]))

    def pre_epoch(self , trainset) : 
        '''
        preprocessing the trainset and then return the trainset after process
        side effect : must add a self.trainset in the self
        return numstep 
        '''
        self.sample = NegativeSample3(self.nu , self.ni , 1)
        self.trainset = self.sample.sample(trainset)
        return None

    def getoptimizer(self) : 
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.mf_learnrate) 
        return optimizer

    def init(self , args , trainset , testset) : 
        self.args = args
        self.batchsize = args.mf_batchsize
        self.nu = self.args.encoder['nu']
        self.ni = self.args.encoder['ni']
        self.rawtrainset = [[item[0] , item[1] , 1] for item in trainset['interaction']]
        self.rawtestset = [[item[0] , item[1] , 1] for item in testset['interaction']]
        self.model = MF(self.args , self.nu , self.ni)
        self.evaluater =Evaluate(self.args , self.nu , self.ni)



