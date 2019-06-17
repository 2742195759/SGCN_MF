import torch
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from utils import gather_2dim_list

class TopkEvaluate(object):
    """TopkEvaluate : to Evaluate the accurate or other coefficient of the model
        must override the preprocess_testset
    """
    def __init__(self , args , nu , ni) :
        super(TopkEvaluate, self).__init__()
        self.args = args
        self.topk = 20 
        if self.args.topk : 
            self.topk = self.args.topk
        print ("[WARN] evaluate topk = %d" % self.topk)
        self.retain = {}
        self.nu = nu
        self.ni = ni

    def dataset2recmap(self , dataset) : 
        """
            function to preprocess the dataset to a np.array
        """
        raise NotImplementedError("TopkEvaluate have to be implement the preprocess_testset")

    def preprocess_set(self , test_set , key , retain=True) : 
        if key in self.retain : return self.retain[key]
        ans = self.dataset2recmap(test_set)
        if retain == True : 
            self.retain[key] = ans
        return ans

    def clear(self) : 
        self.retain = {}

    def numpy2recmap(self , arr) : 
        ans = {}
        for i in range(arr.shape[0]) : 
            ans[i] = arr[i].tolist()
        return ans

    def accurate(self , module , train_set , test_set , retain=True) : 
        """ 
            calculate the accurate of the recsys
            parameter module : the module which have a get_topk method to call
            parameter train_set : the trainset of the task
            parameter test_set : the test_set of the task
        """
        test = self.preprocess_set(test_set , 'test' , retain)
        train = self.preprocess_set(train_set ,'train' , retain)
        #import pdb
        #pdb.set_trace()
        rec = {}
        recset = {}
        for ui in range(self.nu) : 
            rec[ui] = self.modulereclist(module , ui)
        
        for u,l in rec.items() : 
            tmp = set()
            trainset = set(train[u]) if u in train else set()
            cnt = 1 
            for i in l : 
                if cnt > self.topk : break
                if i in trainset : continue
                cnt += 1
                tmp.add(i)
            recset[u] = tmp
            
        prec = 0.0
        recall = 0.0 
        len_pre = 0.0
        len_call = 0.0
        for t_u , t_list in test.items() : 
            s_predict = recset[t_u]
            s_testset = set(t_list)
            assert(isinstance(s_predict , set))
            prec += len (s_predict.intersection(s_testset)) * 1.0 
            len_pre += len(s_predict)
            recall += len (s_predict.intersection(s_testset)) * 1.0
            len_call += len(s_testset)
        prec /= len_pre
        recall /= len_call
        f1 = 2 * (prec * recall) / (prec + recall)

        return prec , recall , f1

class Evaluate(TopkEvaluate) :
    def dataset2recmap(self , dataset) : 
        '''
            dataset is [ items,items,... ]
        '''
        inte = dataset
        gath = gather_2dim_list(inte , 0)
        for k,v in gath.items() : 
            v = [item[1] for item in v]
            gath[k] = v
        return gath

    def modulereclist(self , module , uid) : 
        arr = module([uid]*self.ni , range(self.ni)).detach().numpy()
        score = [[i , s] for i , s in enumerate(arr.tolist())]
        score = sorted(score , key=lambda x:x[1] , reverse=True)
        return [item[0] for item in score]

class EvaluateModifiedMF(Evaluate) :

    def modulereclist(self , module , uid) : 
        arr = module(self.Z , [uid]*self.ni , range(self.ni)).cpu().detach().numpy()
        score = [[i , s] for i , s in enumerate(arr.tolist())]
        score = sorted(score , key=lambda x:x[1] , reverse=True)
        return [item[0] for item in score]

    def set_Z(self , Z) : 
        self.Z = Z
